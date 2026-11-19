"""Treina o detector de luminárias do Luz Segura (YOLO26n, ajuste fino a partir do COCO).

Uso:
  python scripts/treino/treinar.py --dados data/dataset/luzsegura.yaml --epocas 25
  python scripts/treino/treinar.py --dados data/dataset_v2/luzsegura_v2.yaml --modelo models/luzsegura_yolo26n_v1.pt \\
      --nome luzsegura_yolo26n_v2 --destino-pesos models/luzsegura_yolo26n_v2.pt --destino-resultados results/treino_v2 --epocas 12

Ao final grava os melhores pesos em --destino-pesos (padrão models/luzsegura_yolo26n.pt) e uma cópia com o
nome do treino ao lado (ex.: models/luzsegura_yolo26n_v1.pt), para o v2 e a escolha entre v1 e v2 sempre
acharem os pesos de cada versão. O Ultralytics guarda nos pesos os caminhos absolutos do treino (com o nome
do usuário da máquina); nas cópias, eles ficam relativos ao repositório (scripts/limpar_caminhos_pesos.py).
Para --destino-resultados (padrão results/treino/) vão só os gráficos de métricas, o results.csv e o
gráfico de estatística dos rótulos (labels.jpg): os mosaicos de fotos do treino (train_batch*, val_batch*)
mostram pessoas sem anonimização e o args.yaml tem caminhos locais.
Treinado em CPU (AMD Ryzen 7 5700X3D, 8 núcleos): cerca de 10 min por época com 2.835 imagens a 640 px.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # scripts/, onde fica limpar_caminhos_pesos.py
from limpar_caminhos_pesos import gravar_pesos, limpar_caminhos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dados", type=Path, default=Path("data/dataset/luzsegura.yaml"))
    ap.add_argument("--modelo", default="yolo26n.pt",
                    help="pesos iniciais (padrão: YOLO26n pré-treinado no COCO; o v2 parte dos pesos do v1)")
    ap.add_argument("--epocas", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--paciencia", type=int, default=8)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--nome", default="luzsegura_yolo26n_v1",
                    help="nome da pasta do treino (em --projeto) e da cópia dos pesos ao lado de --destino-pesos")
    ap.add_argument("--projeto", type=Path, default=Path("runs/treino"))
    ap.add_argument("--destino-pesos", type=Path, default=Path("models/luzsegura_yolo26n.pt"))
    ap.add_argument("--destino-resultados", type=Path, default=Path("results/treino"))
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    modelo = YOLO(args.modelo)
    inicio = time.time()
    modelo.train(
        data=str(args.dados), epochs=args.epocas, imgsz=args.imgsz, batch=args.lote, device="cpu",
        workers=0, cache="ram", patience=args.paciencia, seed=0, deterministic=True,
        project=str(args.projeto.resolve()), name=args.nome, exist_ok=True,
        # Aumentos: brilho (hsv_v) reduzido para não apagar artificialmente as luminárias acesas.
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.25, degrees=0.0, translate=0.1, scale=0.5,
        fliplr=0.5, mosaic=1.0, mixup=0.0, close_mosaic=5, plots=True,
    )
    print(f"treino concluído em {(time.time() - inicio) / 3600:.2f} h", flush=True)

    pasta = args.projeto / args.nome
    melhor = pasta / "weights" / "best.pt"
    args.destino_pesos.parent.mkdir(parents=True, exist_ok=True)
    pesos = torch.load(melhor, map_location="cpu", weights_only=False)  # checkpoint do próprio treino
    limpar_caminhos(pesos)
    gravar_pesos(pesos, args.destino_pesos)
    # O próximo treino com o destino padrão sobrescreveria estes pesos; a cópia com o nome do treino
    # mantém cada versão (o v2 parte do v1, e a escolha final compara os dois).
    arquivo = args.destino_pesos.parent / f"{args.nome}.pt"
    copia = arquivo.resolve() != args.destino_pesos.resolve()
    if copia:
        gravar_pesos(pesos, arquivo)
    args.destino_resultados.mkdir(parents=True, exist_ok=True)
    for arq in pasta.iterdir():
        # Lista fechada: train_batch*/val_batch*.jpg são mosaicos de fotos do treino sem anonimização (e sem os
        # créditos das fotos) e o args.yaml tem caminhos locais; os gráficos de métricas são todos .png.
        if arq.is_file() and "batch" not in arq.name and (arq.suffix in (".png", ".csv") or arq.name == "labels.jpg"):
            shutil.copy2(arq, args.destino_resultados / arq.name)
    print(f"pesos: {args.destino_pesos}{f' (cópia em {arquivo})' if copia else ''} | "
          f"gráficos: {args.destino_resultados}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
