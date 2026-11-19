"""Dataset v2: adaptação ao domínio das fotos de cidadãos por autotreinamento (pseudo-rótulos).

O modelo v1 foi treinado com fotos do Flickr (Open Images) e noites simuladas. As fotos que o
Luz Segura recebe na prática são de celular, muitas vezes apontadas para cima, para o poste. Para
aproximar os domínios sem anotar nada à mão:

1. As fotos vêm só do POOL DE TREINO do SF 311 (scripts/dados/baixar_sf311.py --treino,
   data/manifests/sf311_treino.csv): chamados noturnos de luminária queimada abertos antes do
   primeiro chamado noturno da avaliação, a mais de 25 m de qualquer chamado avaliado e sem foto idêntica
   a uma foto avaliada. Chennai e o SF 311 de avaliação nunca entram no treino: o script confere
   essa separação antes de montar o dataset e para se ela falhar.
2. O modelo v1 (confiança >= 0,05) propõe as caixas, e o estado de cada uma vem da regra de brilho:
   - caixa forte (confiança >= 0,15) vira rótulo; com brilho indefinido, usa a classe do modelo
     se a confiança for >= 0,40, senão a foto é descartada;
   - candidata fraca (0,05 a 0,15) que não coincide com uma caixa forte pode ser uma luminária que
     o v1 quase perdeu: sem rótulo, ela viraria fundo no treino. A região dela é coberta de cinza
     na cópia de treino da foto (região ignorada: nem rótulo, nem fundo com cara de luminária);
   - o chamado é de luminária queimada: a foto precisa de ao menos uma caixa apagada.
   Fotos sem caixa ou com caixa ambígua ficam de fora, para não ensinar rótulos duvidosos.
3. O treino v2 parte dos pesos v1 e mistura --n-v1 imagens do treino v1 com as fotos aceitas
   (repetidas --repeticoes vezes na lista, para pesarem mais). A validação continua a do v1
   (Open Images).

Saídas: data/dataset_v2/ (images/, labels/, train.txt e luzsegura_v2.yaml) e
        data/manifests/pseudo_rotulos_v2.csv (uma linha por foto do pool: aceita ou motivo do descarte)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from baixar_sf311 import DISTANCIA_MIN_TESTE_M, coordenadas, distancia_min_m  # script vizinho (scripts/dados)
from luzsegura.config import CLASSE_ACESA, CLASSE_APAGADA, CLASSES
from luzsegura.detector import DetectorLuminarias
from luzsegura.luminancia import ACESA, INDEFINIDO

ID = {c: i for i, c in enumerate(CLASSES)}
FONTE = "sf311_treino"  # pasta em data/raw e prefixo das imagens no dataset_v2
FORTE = 0.15  # caixa que vira pseudo-rótulo
FRACA = 0.05  # abaixo de FORTE, mas ainda pode ser uma luminária que o v1 quase perdeu
CONF_INDEFINIDA = 0.40  # caixa sem estado pelo brilho só vira rótulo (com a classe do modelo) acima disto
IOU_MESMA_CAIXA = 0.5  # o YOLO26 não usa NMS: a mesma luminária pode sair com as duas classes
IOU_COINCIDE = 0.3  # candidata fraca sobre uma caixa forte é a mesma luminária, não uma perdida
MARGEM_IGNORADA = 0.15  # a região ignorada cresce 15% de cada lado da caixa fraca
CINZA = 114  # cor do preenchimento do letterbox do Ultralytics: o modelo já a vê como fundo neutro
CAMPOS = ["fonte", "arquivo", "service_request_id", "rotulo_fraco", "aceita", "motivo", "acesas", "apagadas",
          "regioes_ignoradas"]


def ler(manifesto: Path) -> list[dict]:
    with manifesto.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def sha256(caminho: Path) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    uniao = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / uniao if uniao > 0 else 0.0


def ignorar_regioes(img: np.ndarray, regioes: list[tuple], fortes: list[tuple]) -> np.ndarray:
    """Cópia da foto com as regiões duvidosas cobertas de cinza, sem tocar nas caixas rotuladas."""
    h, w = img.shape[:2]
    saida = img.copy()
    for x1, y1, x2, y2 in regioes:
        mx, my = (x2 - x1) * MARGEM_IGNORADA, (y2 - y1) * MARGEM_IGNORADA
        a, b = max(0, int(x1 - mx)), max(0, int(y1 - my))
        c, d = min(w, int(np.ceil(x2 + mx))), min(h, int(np.ceil(y2 + my)))
        saida[b:d, a:c] = CINZA
    for x1, y1, x2, y2 in fortes:  # a margem não pode apagar parte de uma luminária rotulada
        a, b, c, d = max(0, int(x1)), max(0, int(y1)), min(w, int(np.ceil(x2))), min(h, int(np.ceil(y2)))
        saida[b:d, a:c] = img[b:d, a:c]
    return saida


def rotular(detector: DetectorLuminarias, img: np.ndarray) -> tuple[list[tuple[int, tuple]] | None, str, list, list]:
    """Pseudo-rótulos YOLO de uma foto de chamado de luminária queimada.

    Devolve (caixas, motivo, regiões a ignorar, caixas rotuladas em pixels); caixas é None quando a
    foto é descartada, e então o motivo diz por quê.
    """
    h, w = img.shape[:2]
    deteccoes = sorted(detector.detectar(img), key=lambda l: -l.confianca)
    todas_fortes = [l for l in deteccoes if l.confianca >= FORTE]
    if not todas_fortes:
        return None, "sem_caixa", [], []
    ignoradas = [l.xyxy for l in deteccoes
                 if l.confianca < FORTE and all(_iou(l.xyxy, f.xyxy) < IOU_COINCIDE for f in todas_fortes)]
    fortes = []  # uma caixa por luminária: a mais confiante
    for l in todas_fortes:
        if all(_iou(l.xyxy, f.xyxy) < IOU_MESMA_CAIXA for f in fortes):
            fortes.append(l)
    caixas = []
    for l in fortes:
        if l.medida.estado == INDEFINIDO:
            if l.confianca < CONF_INDEFINIDA:
                return None, "brilho_indefinido", [], []
            classe = l.classe
        else:
            classe = CLASSE_ACESA if l.medida.estado == ACESA else CLASSE_APAGADA
        x1, y1 = max(0.0, l.xyxy[0]), max(0.0, l.xyxy[1])
        x2, y2 = min(float(w), l.xyxy[2]), min(float(h), l.xyxy[3])
        caixas.append((ID[classe], ((x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h)))
    if all(c != ID[CLASSE_APAGADA] for c, _ in caixas):
        return None, "sem_apagada", [], []  # o chamado diz que há luminária apagada, mas nenhuma caixa confirma
    return caixas, "", ignoradas, [l.xyxy for l in fortes]


def conferir_separacao(pool: list[dict], aval: list[dict], raw: Path) -> list[str]:
    """Motivos pelos quais o pool de treino tocaria a avaliação do SF 311 (lista vazia: está separado)."""
    problemas = []
    comuns = {r["service_request_id"] for r in pool} & {r["service_request_id"] for r in aval}
    if comuns:
        problemas.append(f"{len(comuns)} chamado(s) do pool também estão na avaliação (ex.: {min(comuns)})")
    inicio = min(datetime.fromisoformat(r["aberto_em"]) for r in aval if r["grupo"] == "noturno_queimada")
    tardios = sum(datetime.fromisoformat(r["aberto_em"]) >= inicio for r in pool)
    if tardios:
        problemas.append(f"{tardios} chamado(s) do pool abertos a partir de {inicio.isoformat()} (início da avaliação noturna)")
    pontos = np.array([p for r in aval if (p := coordenadas(r["lat"], r["lon"]))], dtype=np.float64).reshape(-1, 2)
    perto = 0
    for r in pool:
        p = coordenadas(r["lat"], r["lon"])
        if p is None or not len(pontos) or distancia_min_m(*p, pontos) <= DISTANCIA_MIN_TESTE_M:
            perto += 1
    if perto:
        problemas.append(f"{perto} chamado(s) do pool sem coordenadas ou a até {DISTANCIA_MIN_TESTE_M:.0f} m "
                         "de um chamado avaliado")
    ausentes = [r["arquivo"] for r in aval if not (raw / "sf311" / r["arquivo"]).is_file()]
    if ausentes:
        problemas.append(f"{len(ausentes)} foto(s) da avaliação ausentes em {raw / 'sf311'} (ex.: {ausentes[0]}): "
                         "sem elas não dá para conferir fotos idênticas")
    else:
        avaliadas = {sha256(raw / "sf311" / r["arquivo"]) for r in aval}
        iguais = [r["arquivo"] for r in pool
                  if (raw / FONTE / r["arquivo"]).is_file() and sha256(raw / FONTE / r["arquivo"]) in avaliadas]
        if iguais:
            problemas.append(f"{len(iguais)} foto(s) do pool idênticas a fotos da avaliação (ex.: {iguais[0]})")
    return problemas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pesos", type=Path, default=Path("models/luzsegura_yolo26n_v1.pt"), help="pesos do modelo v1")
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--dataset-v1", type=Path, default=Path("data/dataset"))
    ap.add_argument("--manifestos", type=Path, default=Path("data/manifests"))
    ap.add_argument("--saida", type=Path, default=Path("data/dataset_v2"))
    ap.add_argument("--n-v1", type=int, default=1500, help="imagens do treino v1 mantidas no v2")
    ap.add_argument("--repeticoes", type=int, default=2, help="vezes que cada foto aceita entra na lista de treino")
    ap.add_argument("--semente", type=int, default=2026)
    args = ap.parse_args()
    torch.set_num_threads(8)

    manifesto_pool = args.manifestos / "sf311_treino.csv"
    if not manifesto_pool.exists():
        print(f"erro: {manifesto_pool} não existe; rode scripts/dados/baixar_sf311.py --treino", file=sys.stderr)
        return 1
    pool = ler(manifesto_pool)
    problemas = conferir_separacao(pool, ler(args.manifestos / "sf311.csv"), args.raw)
    if problemas:
        print("erro: o pool de treino não está separado da avaliação:\n  " + "\n  ".join(problemas), file=sys.stderr)
        return 1
    v1 = sorted((args.dataset_v1 / "images" / "train").glob("*.jpg"))
    if not v1:
        print(f"erro: nenhuma imagem em {args.dataset_v1 / 'images' / 'train'}; rode montar_dataset.py", file=sys.stderr)
        return 1
    if not args.pesos.exists():
        print(f"erro: pesos do v1 não encontrados em {args.pesos}; copie para lá os pesos do treino v1 "
              "(runs/treino/<nome>/weights/best.pt) ou use --pesos", file=sys.stderr)
        return 1
    # Tudo o que pode falhar vem antes de apagar a saída: pesos errados não destroem um dataset_v2 existente.
    detector = DetectorLuminarias(args.pesos, conf=FRACA)

    if args.saida.exists():
        shutil.rmtree(args.saida)
    (args.saida / "images").mkdir(parents=True)
    (args.saida / "labels").mkdir(parents=True)
    manifesto, aceitas = [], []
    for r in pool:
        origem = args.raw / FONTE / r["arquivo"]
        img = cv2.imread(str(origem), cv2.IMREAD_COLOR)
        caixas, motivo, ignoradas, rotuladas = rotular(detector, img) if img is not None else (None, "foto_ilegivel", [], [])
        manifesto.append({"fonte": FONTE, "arquivo": r["arquivo"], "service_request_id": r["service_request_id"],
                          "rotulo_fraco": r["detalhe"], "aceita": int(caixas is not None), "motivo": motivo,
                          "acesas": sum(c == ID[CLASSE_ACESA] for c, _ in caixas or []),
                          "apagadas": sum(c == ID[CLASSE_APAGADA] for c, _ in caixas or []),
                          "regioes_ignoradas": len(ignoradas) if caixas is not None else 0})
        if caixas is None:
            continue
        nome = f"{FONTE}_{Path(r['arquivo']).stem}"
        destino = args.saida / "images" / f"{nome}.jpg"
        if ignoradas:  # cópia própria, com as regiões duvidosas cobertas
            cv2.imwrite(str(destino), ignorar_regioes(img, ignoradas, rotuladas), [cv2.IMWRITE_JPEG_QUALITY, 95])
        else:
            try:
                os.link(origem, destino)
            except OSError:
                shutil.copy2(origem, destino)
        (args.saida / "labels" / f"{nome}.txt").write_text(
            "".join(f"{c} {x:.6f} {y:.6f} {bw:.6f} {bh:.6f}\n" for c, (x, y, bw, bh) in caixas), encoding="utf-8")
        aceitas.append(destino.resolve())
    with (args.manifestos / "pseudo_rotulos_v2.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        w.writeheader()
        w.writerows(manifesto)

    random.Random(args.semente).shuffle(v1)
    v1 = v1[: args.n_v1]
    lista = [p.resolve() for p in v1] + aceitas * args.repeticoes
    random.Random(args.semente).shuffle(lista)
    (args.saida / "train.txt").write_text("\n".join(p.as_posix() for p in lista) + "\n", encoding="utf-8")
    yaml = args.saida / "luzsegura_v2.yaml"
    yaml.write_text(f"# Gerado por scripts/dados/montar_dataset_v2.py\npath: {args.dataset_v1.resolve().as_posix()}\n"
                    f"train: {(args.saida / 'train.txt').resolve().as_posix()}\n"
                    "val: images/val\ntest: images/test\nnames:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)),
                    encoding="utf-8")

    descartes = Counter(m["motivo"] for m in manifesto if not m["aceita"])
    print(f"pseudo-rótulos: {len(aceitas)} de {len(pool)} fotos do pool aceitas "
          f"({sum(m['acesas'] for m in manifesto)} caixas acesas, {sum(m['apagadas'] for m in manifesto)} apagadas) | "
          "descartes: " + (", ".join(f"{k} {v}" for k, v in descartes.most_common()) or "nenhum"))
    print(f"lista de treino v2: {len(lista)} entradas ({len(v1)} do v1 + {len(aceitas)} x {args.repeticoes}) | yaml: {yaml}",
          flush=True)
    if not aceitas:
        print("erro: nenhuma foto do pool foi aceita; o v2 treinaria só com o v1", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
