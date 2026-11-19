"""Gera um mosaico de conferência visual: imagens com as caixas coloridas pelo estado.

Uso:
  python scripts/dados/mosaico_amostras.py --filtro noite --n 16 --saida data/outputs/qa/mosaico_noite.jpg
Filtros: noite (cena < 55), crepusculo (55-105), dia (>= 105), apagada, indefinido.

O mosaico mostra fotos brutas do Open Images (pessoas sem desfoque, sem créditos por foto): é só
para conferência interna e fica em data/outputs/, fora do git.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

CORES = {"acesa": (0, 200, 255), "apagada": (255, 0, 255), "indefinido": (255, 255, 255)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--filtro", default="noite")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--lado", type=int, default=360)
    ap.add_argument("--semente", type=int, default=7)
    ap.add_argument("--saida", type=Path, default=Path("data/outputs/qa/mosaico.jpg"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw/openimages"))
    ap.add_argument("--interim", type=Path, default=Path("data/interim"))
    args = ap.parse_args()

    imgs = {r["image_id"]: r for r in csv.DictReader(open(args.interim / "luminancia_openimages_imagens.csv", encoding="utf-8"))}
    caixas = defaultdict(list)
    for c in csv.DictReader(open(args.interim / "luminancia_openimages_caixas.csv", encoding="utf-8")):
        caixas[c["image_id"]].append(c)

    def ok(i):
        m = float(imgs[i]["cena_media"])
        if args.filtro == "noite":
            return m < 55
        if args.filtro == "crepusculo":
            return 55 <= m < 105
        if args.filtro == "dia":
            return m >= 105
        if args.filtro in ("apagada", "indefinido"):
            return m < 70 and any(c["estado"] == args.filtro for c in caixas[i])
        return True

    ids = sorted(i for i in imgs if ok(i))
    random.Random(args.semente).shuffle(ids)
    ids = ids[: args.n]
    tiles = []
    for i in ids:
        r = imgs[i]
        img = cv2.imread(str(args.raw / "images" / r["subset"] / f"{i}.jpg"))
        h, w = img.shape[:2]
        for c in caixas[i]:
            p1 = (int(float(c["xmin"]) * w), int(float(c["ymin"]) * h))
            p2 = (int(float(c["xmax"]) * w), int(float(c["ymax"]) * h))
            cv2.rectangle(img, p1, p2, CORES[c["estado"]], max(2, w // 300))
        s = args.lado / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)))
        tile = np.zeros((args.lado, args.lado, 3), np.uint8)
        tile[: img.shape[0], : img.shape[1]] = img
        cv2.putText(tile, f"{i[:8]} cena={float(r['cena_media']):.0f}", (4, args.lado - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    linhas = [np.hstack(tiles[k:k + 4]) for k in range(0, len(tiles), 4)]
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.saida), np.vstack(linhas), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(args.saida, len(ids))
    return 0


if __name__ == "__main__":
    sys.exit(main())
