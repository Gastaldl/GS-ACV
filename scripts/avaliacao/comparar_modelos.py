"""Escolhe os pesos finais entre o v1 e o v2 pela regra fixada antes de avaliar o v2.

Regra: o v2 substitui o v1 só se
  1. a média entre o F1 do sistema em Chennai e a fração de chamados noturnos do SF 311 em que o
     sistema acusou a falha for estritamente maior que a do v1; e
  2. o mAP50 no teste do Open Images (todos os recortes) não cair mais de 0,03 em relação ao v1.
Em qualquer outro caso, inclusive sem resultados do v2, ficam os pesos do v1 (--sem-v2 deixa o v2 de fora).

Entradas: results/avaliacao/imagens_v1.json e deteccao_v1.json (obrigatórios), imagens_v2.json e
deteccao_v2.json (se o v2 foi treinado), models/luzsegura_yolo26n_v1.pt e _v2.pt. Cada JSON precisa ter
sido gerado com os pesos da sua versão (confere o SHA-256 gravado pelos scripts de avaliação). Para
descrever o treino do v2: results/treino_v2/resumo_treino.json, data/manifests/pseudo_rotulos_v2.csv e
data/dataset_v2/train.txt.

Saídas:
  - results/modelo_final.json (versão escolhida, pesos, SHA-256, regra, métricas das duas versões e o
    resumo do treino do v2);
  - models/luzsegura_yolo26n.pt: cópia dos pesos escolhidos (os que a API, o app e os vídeos usam);
  - os resultados da versão escolhida copiados sem o sufixo, os nomes lidos pela demonstração e pelo
    relatório (imagens_v1.json -> imagens.json, imagens_chennai_v1.csv -> imagens_chennai.csv,
    deteccao_v1.json -> deteccao.json, deteccao_v1_noite/ -> deteccao_noite/ etc.).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

TOLERANCIA_MAP50 = 0.03
REGRA = ("Regra fixada antes de avaliar o v2: o v2 substitui o v1 só se (1) a média entre o F1 do sistema em "
         "Chennai e a fração de chamados noturnos do SF 311 em que o sistema acusou a falha for estritamente "
         "maior que a do v1 e (2) o mAP50 no teste do Open Images não cair mais de 0,03 em relação ao v1. "
         "Em qualquer outro caso, inclusive sem resultados do v2, ficam os pesos do v1.")
METRICAS = ("chennai_f1_sistema", "chennai_precisao_sistema", "chennai_revocacao_sistema",
            "sf311_noite_acusou_falha_sistema", "sf311_noite_chamado_automatico", "sf311_dia_acusou_acesa_de_dia",
            "oi_teste_map50", "oi_teste_map50_95", "oi_noite_map50")


def relativo(caminho: Path) -> str:
    """Caminho relativo ao repositório (sem a pasta local de quem rodou); fora dele, só o nome."""
    if not caminho.is_absolute():
        return caminho.as_posix()
    try:
        return caminho.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return caminho.name


def sha256(caminho: Path) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def ler_json(caminho: Path) -> dict | None:
    return json.loads(caminho.read_text(encoding="utf-8")) if caminho.exists() else None


def metricas(img: dict, det: dict) -> dict:
    """As métricas comparadas (contrato de results/modelo_final.json), dos JSONs de avaliação de uma versão."""
    return {
        "chennai_f1_sistema": float(img["chennai"]["sistema"]["f1"]),
        "chennai_precisao_sistema": float(img["chennai"]["sistema"]["precisao"]),
        "chennai_revocacao_sistema": float(img["chennai"]["sistema"]["revocacao"]),
        "sf311_noite_acusou_falha_sistema": float(img["sf311_noite"]["acusou_falha_sistema"]),
        "sf311_noite_chamado_automatico": float(img["sf311_noite"]["chamado_automatico"]),
        "sf311_dia_acusou_acesa_de_dia": float(img["sf311_dia"]["acusou_acesa_de_dia"]),
        "oi_teste_map50": float(det["todos"]["mAP50"]),
        "oi_teste_map50_95": float(det["todos"]["mAP50_95"]),
        "oi_noite_map50": float(det["noite"]["mAP50"]),
    }


def criterio(m: dict) -> float:
    # Arredondado: as entradas têm 4 casas, e um erro de ponto flutuante não pode decidir um empate.
    return round((m["chennai_f1_sistema"] + m["sf311_noite_acusou_falha_sistema"]) / 2, 6)


def escolher(v1: dict, v2: dict | None) -> tuple[str, str]:
    """Aplica a regra; devolve a versão e o motivo."""
    if v2 is None:
        return "v1", "sem resultados completos do v2 (ou --sem-v2)"
    if criterio(v2) <= criterio(v1):
        return "v1", f"o v2 não melhorou o critério ({criterio(v2):.4f} contra {criterio(v1):.4f} do v1)"
    queda = round(v1["oi_teste_map50"] - v2["oi_teste_map50"], 6)
    if queda > TOLERANCIA_MAP50:
        return "v1", (f"o v2 melhorou o critério, mas o mAP50 no Open Images caiu {queda:.4f} "
                      f"(limite {TOLERANCIA_MAP50:.2f})")
    return "v2", f"o v2 melhorou o critério ({criterio(v2):.4f} contra {criterio(v1):.4f}) e manteve o mAP50"


def carregar(versao: str, avaliacao: Path, modelos: Path) -> tuple[dict | None, list[str], list[str]]:
    """Lê os resultados e confere os pesos de uma versão. Devolve (dados, faltando, divergências)."""
    arquivos = {"imagens": avaliacao / f"imagens_{versao}.json", "deteccao": avaliacao / f"deteccao_{versao}.json",
                "pesos": modelos / f"luzsegura_yolo26n_{versao}.pt"}
    faltando = [str(p) for p in arquivos.values() if not p.exists()]
    if faltando:
        return None, faltando, []
    img, det = ler_json(arquivos["imagens"]), ler_json(arquivos["deteccao"])
    h = sha256(arquivos["pesos"])
    # Uma avaliação feita com outros pesos não pode decidir qual arquivo é publicado.
    divergencias = []
    for k, d in (("imagens", img), ("deteccao", det)):
        if not d.get("sha256_pesos"):
            print(f"aviso: {arquivos[k]} não registra o SHA-256 dos pesos; não deu para conferir", file=sys.stderr)
        elif d["sha256_pesos"] != h:
            divergencias.append(f"{arquivos[k]} foi gerado com outros pesos (SHA-256 {d['sha256_pesos'][:12]}...), "
                                f"não com {arquivos['pesos']} ({h[:12]}...)")
    return {"imagens": img, "deteccao": det, "pesos": arquivos["pesos"], "sha256": h}, [], divergencias


def descrever_treino_v2(resultados: Path, manifestos: Path, dataset_v2: Path, anterior: dict | None) -> dict | None:
    """Pool de fotos do SF 311 usado no autotreinamento do v2 (o relatório precisa declarar)."""
    pseudo = manifestos / "pseudo_rotulos_v2.csv"
    if not ((resultados / "treino_v2" / "resumo_treino.json").exists() and pseudo.exists()):
        return None
    with pseudo.open(encoding="utf-8", newline="") as f:
        linhas = list(csv.DictReader(f))  # uma linha por foto do pool, aceita ou descartada
    aceitas = [r for r in linhas if int(r["aceita"])]
    lista = dataset_v2 / "train.txt"
    if lista.exists():
        n_lista = sum(1 for l in lista.read_text(encoding="utf-8").splitlines() if l.strip())
    else:
        # data/dataset_v2 fica fora do git: sem a lista (clone novo com --pular-treino), vale o número já registrado.
        n_lista = ((anterior or {}).get("treino_v2") or {}).get("lista_treino")
    return {"fotos_pool": len(linhas), "aceitas": len(aceitas),
            "caixas_acesas": sum(int(r["acesas"]) for r in aceitas),
            "caixas_apagadas": sum(int(r["apagadas"]) for r in aceitas),
            "lista_treino": n_lista}


def publicar_resultados(avaliacao: Path, sufixo: str) -> list[str]:
    """Copia os resultados com o sufixo da versão escolhida para os nomes sem sufixo."""
    copiados = []
    for p in sorted(avaliacao.iterdir()):
        if not p.name.startswith(("imagens", "deteccao")):
            continue
        if p.is_file() and p.suffix in (".json", ".csv") and p.stem.endswith(sufixo):
            destino = p.with_name(p.stem[: -len(sufixo)] + p.suffix)
            shutil.copy2(p, destino)
        elif p.is_dir() and p.name.startswith(f"deteccao{sufixo}_"):
            destino = avaliacao / ("deteccao_" + p.name[len(f"deteccao{sufixo}_"):])
            if destino.exists():
                shutil.rmtree(destino)  # sem sobras de gráficos da outra versão
            shutil.copytree(p, destino)
        else:
            continue
        copiados.append(f"{p.name} -> {destino.name}")
    return copiados


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resultados", type=Path, default=Path("results"))
    ap.add_argument("--modelos", type=Path, default=Path("models"))
    ap.add_argument("--manifestos", type=Path, default=Path("data/manifests"))
    ap.add_argument("--dataset-v2", type=Path, default=Path("data/dataset_v2"))
    ap.add_argument("--sem-v2", action="store_true",
                    help="deixa o v2 fora da escolha mesmo que existam resultados dele (fica o v1)")
    args = ap.parse_args()
    avaliacao = args.resultados / "avaliacao"

    versoes, erros = {}, []
    for versao in ("v1",) if args.sem_v2 else ("v1", "v2"):
        dados, faltando, divergencias = carregar(versao, avaliacao, args.modelos)
        erros += divergencias
        if versao == "v1" and faltando:
            erros.append("faltam os resultados ou os pesos do v1: " + ", ".join(faltando)
                         + " (rode as etapas treino e avaliacao_v1 de scripts/reproduzir.py)")
        elif faltando and len(faltando) < 3:
            print("aviso: v2 incompleto, fica fora da comparação; falta: " + ", ".join(faltando), file=sys.stderr)
        if dados:
            try:
                dados["metricas"] = metricas(dados["imagens"], dados["deteccao"])
            except KeyError as e:
                erros.append(f"resultados do {versao} sem a chave {e}: rode de novo a avaliação do {versao}")
            versoes[versao] = dados
    if erros:
        print("erro: " + "\n  ".join(erros), file=sys.stderr)
        return 1

    v1, v2 = versoes["v1"]["metricas"], versoes.get("v2", {}).get("metricas")
    versao, motivo = escolher(v1, v2)
    escolhida = versoes[versao]
    saida = args.resultados / "modelo_final.json"
    try:
        anterior = ler_json(saida)
    except ValueError:
        anterior = None  # registro anterior ilegível (ex.: conflito de merge): só servia para repetir lista_treino
    try:  # antes de copiar qualquer coisa: um erro aqui não deixa pesos e registro desencontrados
        treino_v2 = None if args.sem_v2 else descrever_treino_v2(args.resultados, args.manifestos, args.dataset_v2,
                                                                 anterior)
    except (KeyError, ValueError) as e:
        print(f"erro: {args.manifestos / 'pseudo_rotulos_v2.csv'} fora do formato esperado ({e!r}); "
              "rode de novo scripts/dados/montar_dataset_v2.py", file=sys.stderr)
        return 1
    destino_pesos = args.modelos / "luzsegura_yolo26n.pt"
    shutil.copy2(escolhida["pesos"], destino_pesos)
    copiados = publicar_resultados(avaliacao, f"_{versao}")

    if versao == "v2" and treino_v2 is None:
        print("aviso: o v2 foi escolhido, mas faltam results/treino_v2/resumo_treino.json ou "
              "pseudo_rotulos_v2.csv; o relatório precisa declarar o pool de treino do v2", file=sys.stderr)
    final = {"versao": versao, "pesos": relativo(escolhida["pesos"]), "sha256": escolhida["sha256"], "regra": REGRA,
             "comparacao": {"v1": v1, "v2": v2}, "treino_v2": treino_v2}
    saida.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"{'métrica':36} {'v1':>8} {'v2':>8}")
    for k in METRICAS:
        print(f"{k:36} {v1[k]:8.4f} " + (f"{v2[k]:8.4f}" if v2 else f"{'-':>8}"))
    print(f"{'critério (média F1 Chennai e SF 311)':36} {criterio(v1):8.4f} " + (f"{criterio(v2):8.4f}" if v2 else f"{'-':>8}"))
    if treino_v2:
        print("treino v2: " + ", ".join(f"{k} {v}" for k, v in treino_v2.items()))
    print(f"escolhido: {versao} ({motivo})")
    print(f"pesos: {escolhida['pesos']} -> {destino_pesos} (SHA-256 {escolhida['sha256'][:12]}...)")
    print("resultados publicados: " + ("; ".join(copiados) or "nenhum"))
    print(f"registro: {saida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
