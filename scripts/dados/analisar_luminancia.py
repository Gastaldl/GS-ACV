"""Mede a luminosidade de todas as imagens e caixas baixadas.

Para cada imagem do Open Images: luminosidade da cena e do piso e, para cada caixa
de "Street light", brilho, núcleo luminoso, contraste com o entorno e estado pela
regra de luminosidade (acesa / apagada / indefinido). Para o dataset de Chennai,
só a cena (as caixas ainda não existem nesta etapa).

Saídas:
  data/interim/luminancia_openimages_caixas.csv
  data/interim/luminancia_openimages_imagens.csv
  data/interim/luminancia_chennai_imagens.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

from luzsegura.luminancia import luma, medir_caixa, medir_cena


def _medir_oi(tarefa):
    caminho, caixas = tarefa
    img = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if img is None:
        return None
    y = luma(img)
    h, w = y.shape
    cena = medir_cena(y)
    linhas = []
    for c in caixas:
        xyxy = (float(c["xmin"]) * w, float(c["ymin"]) * h, float(c["xmax"]) * w, float(c["ymax"]) * h)
        m = medir_caixa(y, xyxy)
        bw, bh = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
        linhas.append({
            "image_id": c["image_id"], "subset": c["subset"],
            "xmin": c["xmin"], "ymin": c["ymin"], "xmax": c["xmax"], "ymax": c["ymax"],
            "largura_px": round(bw, 1), "altura_px": round(bh, 1),
            "fracao_area": round(bw * bh / (w * h), 6), "proporcao_h_w": round(bh / max(bw, 1e-6), 3),
            "media": round(m.media, 1), "pico": round(m.pico, 1), "pixels_nucleo": m.pixels_nucleo,
            "fracao_nucleo": round(m.fracao_nucleo, 5), "maior_mancha": m.maior_mancha,
            "contraste": round(m.contraste, 3), "estado": m.estado,
        })
    img_linha = {"image_id": caixas[0]["image_id"], "subset": caixas[0]["subset"], "largura": w, "altura": h,
                 "cena_media": round(cena.media, 1), "piso_media": round(cena.piso, 1),
                 "fracao_escura": round(cena.fracao_escura, 4), "fracao_saturada": round(cena.fracao_saturada, 5),
                 "n_caixas": len(caixas)}
    return img_linha, linhas


def _medir_cena(caminho):
    img = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if img is None:
        return None
    y = luma(img)
    cena = medir_cena(y)
    return {"arquivo": caminho.parent.name + "/" + caminho.name, "estado_pasta": caminho.parent.name,
            "largura": y.shape[1], "altura": y.shape[0], "cena_media": round(cena.media, 1),
            "piso_media": round(cena.piso, 1), "fracao_escura": round(cena.fracao_escura, 4),
            "fracao_saturada": round(cena.fracao_saturada, 5)}


def _gravar(caminho: Path, linhas: list[dict]) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with caminho.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--manifestos", type=Path, default=Path("data/manifests"))
    ap.add_argument("--saida", type=Path, default=Path("data/interim"))
    ap.add_argument("--processos", type=int, default=14)
    args = ap.parse_args()

    caixas_por_imagem = defaultdict(list)
    with (args.manifestos / "openimages_caixas.csv").open(encoding="utf-8") as f:
        for c in csv.DictReader(f):
            caixas_por_imagem[c["image_id"]].append(c)
    arquivos = {}
    with (args.manifestos / "openimages_imagens.csv").open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            arquivos[r["image_id"]] = args.raw / "openimages" / r["arquivo"]

    tarefas = [(arquivos[i], cx) for i, cx in caixas_por_imagem.items() if i in arquivos]
    imgs, caixas = [], []
    with ProcessPoolExecutor(max_workers=args.processos) as ex:
        for res in ex.map(_medir_oi, tarefas, chunksize=32):
            if res:
                imgs.append(res[0])
                caixas.extend(res[1])
    _gravar(args.saida / "luminancia_openimages_imagens.csv", imgs)
    _gravar(args.saida / "luminancia_openimages_caixas.csv", caixas)
    print(f"Open Images: {len(imgs)} imagens, {len(caixas)} caixas")

    chennai = sorted((args.raw / "chennai").glob("*/*.jpg"))
    with ProcessPoolExecutor(max_workers=args.processos) as ex:
        linhas = [r for r in ex.map(_medir_cena, chennai, chunksize=16) if r]
    _gravar(args.saida / "luminancia_chennai_imagens.csv", linhas)
    print(f"Chennai: {len(linhas)} imagens")
    return 0


if __name__ == "__main__":
    sys.exit(main())
