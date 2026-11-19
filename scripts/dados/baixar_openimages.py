"""Baixa do Open Images V7 as imagens que têm caixas da classe "Street light" (/m/033rq4).

Entradas (baixadas por scripts/dados/baixar_anotacoes_openimages.py):
  data/raw/openimages/validation-annotations-bbox.csv
  data/raw/openimages/test-annotations-bbox.csv
  data/raw/openimages/train-streetlight-bbox.csv            (linhas filtradas, sem cabeçalho)
  data/raw/openimages/*-images-with-rotation.csv             (metadados e licenças)
  data/raw/openimages/train-streetlight-images.csv          (linhas filtradas, sem cabeçalho)

Saídas:
  data/raw/openimages/images/<subset>/<ImageID>.jpg
  data/manifests/openimages_imagens.csv   (uma linha por imagem, com autor, licença e URL de origem)
  data/manifests/openimages_caixas.csv    (uma linha por caixa, coordenadas normalizadas)

Regras de filtragem:
  - descarta imagens com caixa IsDepiction=1 (desenhos, brinquedos, reflexos);
  - descarta imagens com caixa IsGroupOf=1 (uma caixa cobrindo várias luminárias);
  - descarta imagens com rotação diferente de zero nos metadados.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

CLASSE_STREET_LIGHT = "/m/033rq4"
URL_IMAGEM = "https://open-images-dataset.s3.amazonaws.com/{subset}/{image_id}.jpg"
COLUNAS_CAIXA = ["ImageID", "Source", "LabelName", "Confidence", "XMin", "XMax", "YMin", "YMax",
                 "IsOccluded", "IsTruncated", "IsGroupOf", "IsDepiction", "IsInside"]
COLUNAS_IMAGEM = ["ImageID", "Subset", "OriginalURL", "OriginalLandingURL", "License",
                  "AuthorProfileURL", "Author", "Title", "OriginalSize", "OriginalMD5",
                  "Thumbnail300KURL", "Rotation"]


def ler_caixas(caminho: Path, tem_cabecalho: bool, subset: str) -> list[dict]:
    caixas = []
    with caminho.open(newline="", encoding="utf-8") as f:
        leitor = csv.reader(f)
        if tem_cabecalho:
            next(leitor)
        for linha in leitor:
            reg = dict(zip(COLUNAS_CAIXA, linha[: len(COLUNAS_CAIXA)]))
            if reg.get("LabelName") != CLASSE_STREET_LIGHT:
                continue
            reg["subset"] = subset
            caixas.append(reg)
    return caixas


def ler_metadados(caminho: Path, tem_cabecalho: bool, ids: set[str]) -> dict[str, dict]:
    meta = {}
    with caminho.open(newline="", encoding="utf-8") as f:
        leitor = csv.reader(f)
        if tem_cabecalho:
            next(leitor)
        for linha in leitor:
            reg = dict(zip(COLUNAS_IMAGEM, linha))
            if reg.get("ImageID") in ids:
                meta[reg["ImageID"]] = reg
    return meta


def baixar(sessao: requests.Session, url: str, destino: Path, tentativas: int = 3) -> str:
    if destino.exists() and destino.stat().st_size > 0:
        return "existente"
    for _ in range(tentativas):
        try:
            r = sessao.get(url, timeout=30)
            if r.status_code == 200 and r.content[:2] == b"\xff\xd8":
                tmp = destino.with_suffix(".part")
                tmp.write_bytes(r.content)
                tmp.replace(destino)
                return "ok"
            if r.status_code in (403, 404):
                return f"http {r.status_code}"
        except requests.RequestException:
            continue
    return "falhou"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/raw/openimages"))
    ap.add_argument("--manifestos", type=Path, default=Path("data/manifests"))
    ap.add_argument("--max-imagens", type=int, default=0, help="0 = todas")
    ap.add_argument("--paralelo", type=int, default=32)
    args = ap.parse_args()

    raw: Path = args.raw
    caixas = (
        ler_caixas(raw / "validation-annotations-bbox.csv", True, "validation")
        + ler_caixas(raw / "test-annotations-bbox.csv", True, "test")
        + ler_caixas(raw / "train-streetlight-bbox.csv", False, "train")
    )
    por_imagem: dict[str, list[dict]] = defaultdict(list)
    for c in caixas:
        por_imagem[c["ImageID"]].append(c)

    ids = set(por_imagem)
    meta = {}
    meta.update(ler_metadados(raw / "validation-images-with-rotation.csv", True, ids))
    meta.update(ler_metadados(raw / "test-images-with-rotation.csv", True, ids))
    meta.update(ler_metadados(raw / "train-streetlight-images.csv", False, ids))

    descartes = defaultdict(int)
    selecionadas = []
    for image_id, cx in sorted(por_imagem.items()):
        m = meta.get(image_id)
        if m is None:
            descartes["sem metadados"] += 1
            continue
        if any(c["IsDepiction"] == "1" for c in cx):
            descartes["IsDepiction"] += 1
            continue
        if any(c["IsGroupOf"] == "1" for c in cx):
            descartes["IsGroupOf"] += 1
            continue
        rot = (m.get("Rotation") or "0").strip()
        if rot not in ("", "0", "0.0"):
            descartes["rotacionada"] += 1
            continue
        selecionadas.append(image_id)

    if args.max_imagens:
        selecionadas = selecionadas[: args.max_imagens]
    print(f"caixas lidas: {len(caixas)} | imagens: {len(por_imagem)} | selecionadas: {len(selecionadas)}")
    for motivo, n in sorted(descartes.items()):
        print(f"  descartadas por {motivo}: {n}")

    args.manifestos.mkdir(parents=True, exist_ok=True)
    resultados = {}
    sessao = requests.Session()
    sessao.headers["User-Agent"] = "luzsegura-dataset/1.0"
    tarefas = {}
    with ThreadPoolExecutor(max_workers=args.paralelo) as ex:
        for image_id in selecionadas:
            subset = meta[image_id]["Subset"]
            destino = raw / "images" / subset / f"{image_id}.jpg"
            destino.parent.mkdir(parents=True, exist_ok=True)
            url = URL_IMAGEM.format(subset=subset, image_id=image_id)
            tarefas[ex.submit(baixar, sessao, url, destino)] = image_id
        for i, fut in enumerate(as_completed(tarefas), 1):
            resultados[tarefas[fut]] = fut.result()
            if i % 500 == 0:
                print(f"  {i}/{len(tarefas)} processadas", flush=True)

    ok = {k for k, v in resultados.items() if v in ("ok", "existente")}
    falhas = {k: v for k, v in resultados.items() if k not in ok}
    print(f"baixadas: {len(ok)} | falhas: {len(falhas)}")

    with (args.manifestos / "openimages_imagens.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "subset", "arquivo", "autor", "perfil_autor", "titulo", "licenca",
                    "url_original", "pagina_original"])
        for image_id in sorted(ok):
            m = meta[image_id]
            w.writerow([image_id, m["Subset"], f"images/{m['Subset']}/{image_id}.jpg", m["Author"],
                        m["AuthorProfileURL"], m["Title"], m["License"], m["OriginalURL"],
                        m["OriginalLandingURL"]])

    with (args.manifestos / "openimages_caixas.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "subset", "xmin", "ymin", "xmax", "ymax", "ocluida", "truncada", "origem"])
        for image_id in sorted(ok):
            for c in por_imagem[image_id]:
                w.writerow([image_id, c["subset"], c["XMin"], c["YMin"], c["XMax"], c["YMax"],
                            c["IsOccluded"], c["IsTruncated"], c["Source"]])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
