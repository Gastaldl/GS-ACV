"""Baixa fotos reais de chamados de luminária do serviço 311 de São Francisco (DataSF).

É o mesmo fluxo do Luz Segura: o cidadão fotografa a luminária e abre um chamado. São dois
conjuntos, sem nenhum chamado nem foto em comum:

  - AVALIAÇÃO (padrão): teste de ponta a ponta com relatos reais, nunca usado no treino.
      noturnos: service_details = 'light_burnt_out', aberto entre 19h e 5h (hora local), com foto;
      diurnos (regra de desperdício): 'light_on_all_the_time', aberto entre 10h e 15h, com foto.
    O manifesto passa por deduplicar_manifestos.py (fotos repetidas saem).
  - POOL DE TREINO (--treino): fotos para o autotreinamento do modelo v2 (montar_dataset_v2.py):
      chamados noturnos de luminária queimada abertos ANTES do primeiro chamado NOTURNO da avaliação,
      a mais de 25 m (haversine) de qualquer chamado avaliado e sem foto idêntica a uma foto
      avaliada. Exige a avaliação já baixada.

Reprodutibilidade: se o manifesto já existe, o script baixa de novo exatamente as linhas dele (os
mesmos chamados e fotos); no pool, as exclusões são conferidas outra vez. --nova-amostra consulta a
API de novo; a avaliação nova fica limitada a chamados abertos antes de --ate.

Fonte: DataSF, "311 Cases" (https://data.sf.gov/resource/vw6y-z8j6), licença ODC PDDL
(domínio público), atribuição "San Francisco 311". Consulta pela API Socrata, sem chave.

Saídas (fotos reduzidas para no máximo 1280 px):
  data/raw/sf311/<grupo>/<service_request_id>.jpg          e data/manifests/sf311.csv
  data/raw/sf311_treino/<grupo>/<service_request_id>.jpg   e data/manifests/sf311_treino.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import requests

from deduplicar_manifestos import deduplicar_manifesto  # script vizinho (scripts/dados)

API = "https://data.sf.gov/resource/vw6y-z8j6.json"
UA = "LuzSeguraDataset/1.0 (projeto academico FIAP)"
GRUPOS = {
    "noturno_queimada": ("light_burnt_out",
                         "(date_extract_hh(requested_datetime) >= 19 OR date_extract_hh(requested_datetime) <= 5)"),
    "diurno_acesa": ("light_on_all_the_time",
                     "(date_extract_hh(requested_datetime) >= 10 AND date_extract_hh(requested_datetime) <= 15)"),
}
CAMPOS = ["arquivo", "grupo", "service_request_id", "aberto_em", "detalhe", "lat", "lon", "url_foto"]
GRUPO_TREINO = "noturno_queimada"
DISTANCIA_MIN_TESTE_M = 25.0  # chamado do pool mais perto que isto de um avaliado pode ser o mesmo poste
RAIO_TERRA_M = 6_371_008.8
PAGINA = 1000  # registros por consulta do pool (as exclusões pedem mais chamados que --n-treino)
LOTE = 60  # fotos do pool baixadas por rodada: só o que falta para --n-treino, na ordem da consulta


def onde(detalhe: str, filtro_hora: str, antes_de: str) -> str:
    """Cláusula $where: chamados de luminária com foto, na faixa de horário e abertos antes de antes_de."""
    return (f"service_name='Streetlights' AND service_details='{detalhe}' "
            f"AND media_url.url like '%cloudinary%' AND {filtro_hora} AND requested_datetime < '{antes_de}'")


def consultar(sessao: requests.Session, where: str, limite: int, deslocamento: int = 0) -> list[dict]:
    params = {
        "$select": "service_request_id,requested_datetime,service_details,lat,long,media_url",
        "$where": where,
        "$order": "requested_datetime DESC, service_request_id DESC",  # ordem total: páginas estáveis
        "$limit": str(limite),
        "$offset": str(deslocamento),
    }
    r = sessao.get(API, params=params, timeout=60)
    r.raise_for_status()
    return r.json()


def paginas(sessao: requests.Session, where: str):
    """Percorre o resultado inteiro da consulta, página a página, só quando for preciso."""
    deslocamento = 0
    while True:
        registros = consultar(sessao, where, PAGINA, deslocamento)
        yield from registros
        if len(registros) < PAGINA:
            return
        deslocamento += len(registros)


def preparar(conteudo: bytes, lado_max: int) -> bytes | None:
    """Decodifica, reduz para no máximo lado_max px e recodifica em JPEG.

    Avaliação e pool passam pelo mesmo processo: a mesma foto vira os mesmos bytes, e o pool exclui
    fotos idênticas às avaliadas comparando o SHA-256 dos arquivos gravados.
    """
    img = cv2.imdecode(np.frombuffer(conteudo, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None
    h, w = img.shape[:2]
    escala = lado_max / max(h, w)
    if escala < 1:
        img = cv2.resize(img, (round(w * escala), round(h * escala)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return buf.tobytes() if ok else None


def baixar(sessao: requests.Session, url: str, destino: Path, lado_max: int,
           proibidos: frozenset[str] = frozenset()) -> tuple[str, str]:
    """Baixa, reduz e grava a foto. Devolve (status, SHA-256 do arquivo).

    Foto cujo SHA-256 está em `proibidos` (as fotos da avaliação) não é gravada.
    """
    if destino.exists() and destino.stat().st_size > 0:
        h = hashlib.sha256(destino.read_bytes()).hexdigest()
        if h in proibidos:
            destino.unlink()
            return "identica_a_avaliacao", h
        return "existente", h
    for _ in range(3):
        try:
            r = sessao.get(url, timeout=60)
        except requests.RequestException:
            time.sleep(2)
            continue
        if r.status_code != 200:
            return f"http {r.status_code}", ""
        dados = preparar(r.content, lado_max)
        if dados is None:
            return "nao_e_imagem", ""
        h = hashlib.sha256(dados).hexdigest()
        if h in proibidos:
            return "identica_a_avaliacao", h
        destino.parent.mkdir(parents=True, exist_ok=True)
        tmp = destino.with_name(destino.name + ".part")
        tmp.write_bytes(dados)
        tmp.replace(destino)  # um download interrompido não deixa foto truncada passando por "existente"
        return "ok", h
    return "falhou", ""


def linha(reg: dict, grupo: str, url: str) -> dict:
    return {"arquivo": f"{grupo}/{reg['service_request_id']}.jpg", "grupo": grupo,
            "service_request_id": reg["service_request_id"], "aberto_em": reg.get("requested_datetime", ""),
            "detalhe": reg.get("service_details", ""), "lat": reg.get("lat", ""), "lon": reg.get("long", ""),
            "url_foto": url}


def registro(r: dict) -> dict:
    """Linha de manifesto de volta ao formato da API, para baixar de novo o mesmo chamado."""
    return {"service_request_id": r["service_request_id"], "requested_datetime": r["aberto_em"],
            "service_details": r["detalhe"], "lat": r["lat"], "long": r["lon"], "media_url": {"url": r["url_foto"]}}


def ler(manifesto: Path) -> list[dict]:
    with manifesto.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def gravar(manifesto: Path, campos: list[str], linhas: list[dict]) -> None:
    manifesto.parent.mkdir(parents=True, exist_ok=True)
    with manifesto.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(sorted(linhas, key=lambda x: x["arquivo"]))


def baixar_avaliacao(sessao: requests.Session, args: argparse.Namespace) -> int:
    manifesto = args.manifestos / "sf311.csv"
    fixos = None
    if manifesto.exists() and not args.nova_amostra:
        fixos = {}
        for r in ler(manifesto):
            fixos.setdefault(r["grupo"], []).append(registro(r))
        print(f"repetindo a amostra versionada em {manifesto} (--nova-amostra consulta a API de novo)")
    linhas, falhas = [], []
    for grupo, (detalhe, filtro) in GRUPOS.items():
        if fixos is not None:
            registros = fixos.get(grupo, [])
        else:
            n = args.n_noturno if grupo.startswith("noturno") else args.n_diurno
            registros = consultar(sessao, onde(detalhe, filtro, args.ate), n)
        print(f"{grupo}: {len(registros)} chamados com foto", flush=True)
        with ThreadPoolExecutor(max_workers=6) as ex:
            tarefas = {}
            for reg in registros:
                url = (reg.get("media_url") or {}).get("url")
                if not url:
                    continue
                destino = args.saida / grupo / f"{reg['service_request_id']}.jpg"
                tarefas[ex.submit(baixar, sessao, url, destino, args.lado_max)] = (reg, url)
            for fut in as_completed(tarefas):
                reg, url = tarefas[fut]
                status = fut.result()[0]
                if status in ("ok", "existente"):
                    linhas.append(linha(reg, grupo, url))
                else:
                    falhas.append(f"{reg['service_request_id']} ({status})")
    print(f"baixadas: {len(linhas)}", flush=True)
    if not linhas:  # nada baixado (ex.: sem rede): o manifesto existente fica intacto
        print("erro: nenhuma foto baixada" + (f" (ex.: {min(falhas)})" if falhas else "")
              + f"; {manifesto} não foi alterado", file=sys.stderr)
        return 1
    if fixos is not None and falhas:
        # Só ficam no manifesto as fotos que existem em disco: a avaliação não pode ler arquivo ausente.
        print(f"aviso: {len(falhas)} foto(s) do manifesto não foram baixadas e saem dele: "
              + ", ".join(sorted(falhas)[:10]) + (" ..." if len(falhas) > 10 else ""), file=sys.stderr)
    gravar(manifesto, CAMPOS, linhas)
    antes, depois = deduplicar_manifesto(manifesto, args.saida, "grupo", args.manifestos / "duplicatas_removidas.csv")
    if len(depois) < len(antes):
        print(f"fotos repetidas removidas do manifesto: {len(antes) - len(depois)} (ver duplicatas_removidas.csv)")
    return 0


def coordenadas(lat, lon) -> tuple[float, float] | None:
    try:
        la, lo = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    return (la, lo) if la and lo else None


def distancia_min_m(lat: float, lon: float, pontos: np.ndarray) -> float:
    """Menor distância haversine (m) de (lat, lon) até os pontos [[lat, lon], ...]."""
    la1, lo1 = np.radians(lat), np.radians(lon)
    la2, lo2 = np.radians(pontos[:, 0]), np.radians(pontos[:, 1])
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return float(2 * RAIO_TERRA_M * np.arcsin(np.sqrt(a)).min())


def triar(reg: dict, limite: datetime, pontos: np.ndarray) -> tuple[str, float]:
    """Motivo de exclusão do pool ('' quando o chamado entra) e distância ao chamado avaliado mais próximo."""
    if not (reg.get("media_url") or {}).get("url"):
        return "sem_foto", 0.0
    try:
        aberto = datetime.fromisoformat(reg["requested_datetime"])
    except (KeyError, TypeError, ValueError):
        return "sem_data", 0.0
    if aberto >= limite:
        return "aberto_depois_do_inicio_da_avaliacao", 0.0
    ponto = coordenadas(reg.get("lat"), reg.get("long"))
    if ponto is None:
        return "sem_coordenadas", 0.0  # sem posição não dá para garantir os 25 m
    dist = distancia_min_m(*ponto, pontos)
    if dist <= DISTANCIA_MIN_TESTE_M:
        return "perto_de_chamado_avaliado", dist
    return "", dist


def coletar_pool(sessao: requests.Session, registros: Iterable[dict], n: int, limite: datetime,
                 pontos: np.ndarray, proibidos: frozenset[str], args: argparse.Namespace) -> tuple[list[dict], Counter]:
    """Baixa, na ordem, os chamados que passam nas exclusões até juntar n fotos."""
    registros = iter(registros)
    linhas, motivos, ids, vistos = [], Counter(), set(), set()
    while len(linhas) < n:
        lote = []
        for reg in registros:
            motivo, dist = triar(reg, limite, pontos)
            if not motivo and reg["service_request_id"] in ids:
                motivo = "chamado_repetido"
            if motivo:
                motivos[motivo] += 1
                continue
            ids.add(reg["service_request_id"])
            lote.append((reg, dist))
            if len(lote) >= min(LOTE, n - len(linhas)):
                break
        if not lote:
            break
        destinos = [args.saida_treino / GRUPO_TREINO / f"{reg['service_request_id']}.jpg" for reg, _ in lote]
        with ThreadPoolExecutor(max_workers=6) as ex:
            resultados = list(ex.map(lambda c, d: baixar(sessao, c[0]["media_url"]["url"], d, args.lado_max, proibidos),
                                     lote, destinos))
        for (reg, dist), destino, (status, h) in zip(lote, destinos, resultados):
            if status not in ("ok", "existente"):
                motivos[status] += 1
            elif h in vistos:  # a mesma foto em dois chamados do pool: fica a primeira
                destino.unlink(missing_ok=True)
                motivos["foto_repetida_no_pool"] += 1
            else:
                vistos.add(h)
                # Truncada, não arredondada: o relatório afirma "nenhum a menos de N m" com o mínimo desta
                # coluna, e 25,97 m arredondado para 26,0 tornaria a frase falsa.
                linhas.append({**linha(reg, GRUPO_TREINO, reg["media_url"]["url"]),
                               "distancia_min_teste_m": f"{int(dist * 10) / 10:.1f}"})
    return linhas, motivos


def baixar_pool(sessao: requests.Session, args: argparse.Namespace) -> int:
    manifesto_aval = args.manifestos / "sf311.csv"
    if not manifesto_aval.exists():
        print(f"erro: {manifesto_aval} não existe; baixe antes a avaliação (sem --treino)", file=sys.stderr)
        return 1
    aval = ler(manifesto_aval)
    ausentes = [r["arquivo"] for r in aval if not (args.saida / r["arquivo"]).is_file()]
    if ausentes:
        print(f"erro: {len(ausentes)} foto(s) da avaliação ausentes em {args.saida} (ex.: {ausentes[0]}); "
              "baixe antes a avaliação (sem --treino)", file=sys.stderr)
        return 1
    sem_posicao = sum(coordenadas(r["lat"], r["lon"]) is None for r in aval)
    if sem_posicao:
        print(f"aviso: {sem_posicao} chamado(s) da avaliação sem coordenadas ficam fora da regra dos 25 m",
              file=sys.stderr)
    pontos = np.array([p for r in aval if (p := coordenadas(r["lat"], r["lon"]))], dtype=np.float64).reshape(-1, 2)
    if not len(pontos):
        print("erro: nenhum chamado da avaliação tem coordenadas; a regra dos 25 m não pode ser aplicada",
              file=sys.stderr)
        return 1
    # Corte no chamado NOTURNO de avaliação mais antigo: o pool só tem chamados noturnos de luminária
    # queimada, e a regra dos 25 m vale contra todos os chamados avaliados (inclusive os diurnos).
    limite = min(datetime.fromisoformat(r["aberto_em"]) for r in aval if r["grupo"] == "noturno_queimada")
    proibidos = frozenset(hashlib.sha256((args.saida / r["arquivo"]).read_bytes()).hexdigest() for r in aval)

    manifesto = args.manifestos / "sf311_treino.csv"
    repetindo = manifesto.exists() and not args.nova_amostra
    if repetindo:
        fixos = ler(manifesto)
        registros, n = [registro(r) for r in fixos], len(fixos)
        print(f"repetindo o pool versionado em {manifesto}: {n} chamados (--nova-amostra consulta a API de novo)")
    else:
        detalhe, filtro = GRUPOS[GRUPO_TREINO]
        registros = paginas(sessao, onde(detalhe, filtro, limite.isoformat(timespec="milliseconds")))
        n = args.n_treino
        print(f"pool de treino: chamados noturnos de luminária queimada abertos antes de {limite.isoformat()} "
              f"(primeiro chamado noturno da avaliação), a mais de {DISTANCIA_MIN_TESTE_M:.0f} m dos avaliados", flush=True)
    linhas, motivos = coletar_pool(sessao, registros, n, limite, pontos, proibidos, args)
    print(f"pool de treino: {len(linhas)} fotos | fora do pool: "
          + (", ".join(f"{k} {v}" for k, v in motivos.most_common()) or "nenhum"), flush=True)
    if not linhas:  # nada baixado: um manifesto existente fica intacto
        print(f"erro: nenhuma foto no pool (motivos acima); {manifesto} não foi alterado", file=sys.stderr)
        return 1
    if len(linhas) < n:
        print(f"aviso: {len(linhas)} de {n} fotos"
              + (": linhas do manifesto saíram pelas exclusões ou falhas acima" if repetindo
                 else ": a consulta acabou, não há mais chamados que passem nas exclusões"), file=sys.stderr)
    gravar(manifesto, CAMPOS + ["distancia_min_teste_m"], linhas)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saida", type=Path, default=Path("data/raw/sf311"), help="fotos da avaliação")
    ap.add_argument("--saida-treino", type=Path, default=Path("data/raw/sf311_treino"), help="fotos do pool de treino")
    ap.add_argument("--manifestos", type=Path, default=Path("data/manifests"))
    ap.add_argument("--treino", action="store_true", help="baixa o pool de treino do v2 em vez da avaliação")
    ap.add_argument("--nova-amostra", action="store_true",
                    help="consulta a API de novo em vez de repetir o manifesto existente")
    ap.add_argument("--ate", default="2026-11-18T00:00:00",
                    help="avaliação nova: só chamados abertos antes disto (hora local de São Francisco)")
    ap.add_argument("--n-noturno", type=int, default=400)
    ap.add_argument("--n-diurno", type=int, default=60)
    ap.add_argument("--n-treino", type=int, default=1200)
    ap.add_argument("--lado-max", type=int, default=1280,
                    help="mesmo valor nos dois conjuntos: a exclusão de fotos idênticas compara os arquivos gravados")
    args = ap.parse_args()

    sessao = requests.Session()
    sessao.headers["User-Agent"] = UA
    return baixar_pool(sessao, args) if args.treino else baixar_avaliacao(sessao, args)


if __name__ == "__main__":
    sys.exit(main())
