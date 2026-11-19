"""Mede a velocidade em CPU: PyTorch x ONNX, com e sem o filtro de privacidade.

Usa quadros reais de um vídeo noturno (1280x720): o --video dado ou, sem ele, o primeiro vídeo
do manifesto data/manifests/videos.csv que estiver baixado em data/raw/videos (o nome do vídeo
usado fica no fps.json). Também usa o modelo em ONNX (models/luzsegura_yolo26n.onnx), formato usado
para rodar mais rápido em CPU, reexportado sempre que o .onnx não vier destes pesos .pt
(luzsegura.detector.onnx_atualizado, conferido pelo selo SHA-256 ao lado do .onnx).

As medidas com PyTorch usam 8 threads ("threads"). As com ONNX rodam como no processar_videos.py: ao carregar
o modelo ONNX, o Ultralytics deixa o PyTorch com 2 threads ("threads_pytorch_com_onnx") e o ONNX Runtime usa as
suas; isso vale também para o filtro de privacidade, que é PyTorch.

Saída: results/avaliacao/fps.json
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
from pathlib import Path
from time import perf_counter

import cv2
import torch

from luzsegura.detector import DetectorLuminarias, onnx_atualizado
from luzsegura.luminancia import luma, medir_cena
from luzsegura.privacidade import FiltroPrivacidade


def quadros(video: Path, n: int, inicio_s: float = 0.0, largura: int = 1280) -> list:
    cap = cv2.VideoCapture(str(video))
    if inicio_s > 0:  # pula a cartela de abertura: quadros sem cena mediriam uma velocidade irreal
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(inicio_s * (cap.get(cv2.CAP_PROP_FPS) or 30)))
    saida = []
    while len(saida) < n:
        ok, q = cap.read()
        if not ok:
            break
        if q.shape[1] != largura:
            q = cv2.resize(q, (largura, round(q.shape[0] * largura / q.shape[1])))
        saida.append(q)
    cap.release()
    return saida


def candidatos(pasta: Path, manifesto: Path) -> list[tuple[Path, float]]:
    """(vídeo, segundo em que a cena começa), na ordem do manifesto (os que falharam no download não estão nele)."""
    if manifesto.exists():
        with manifesto.open(encoding="utf-8") as f:
            return [(pasta / r["arquivo"], float(r.get("inicio_s") or 0)) for r in csv.DictReader(f)]
    return sorted((p, 0.0) for p in pasta.glob("*") if p.suffix.lower() in (".mp4", ".webm", ".ogv", ".mkv"))


def medir(funcao, qs: list, aquecimento: int = 5) -> float:
    for q in qs[:aquecimento]:
        funcao(q)
    inicio = perf_counter()
    for q in qs:
        funcao(q)
    return len(qs) / (perf_counter() - inicio)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pesos", type=Path, default=Path("models/luzsegura_yolo26n.pt"))
    ap.add_argument("--video", type=Path, default=None,
                    help="vídeo noturno (padrão: o primeiro de data/manifests/videos.csv baixado em data/raw/videos)")
    ap.add_argument("--videos", type=Path, default=Path("data/raw/videos"))
    ap.add_argument("--manifesto", type=Path, default=Path("data/manifests/videos.csv"))
    ap.add_argument("--inicio", type=float, default=0.0, help="com --video: segundo em que a medição começa")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--saida", type=Path, default=Path("results/avaliacao/fps.json"))
    args = ap.parse_args()
    if not args.pesos.exists():
        print(f"erro: pesos não encontrados: {args.pesos} (rode scripts/treino/treinar.py)", file=sys.stderr)
        return 1
    torch.set_num_threads(8)

    # Quadros antes da exportação: um vídeo ausente falha na hora, sem pagar a exportação para ONNX.
    video, inicio_s, qs = None, 0.0, []
    for v, ini in ([(args.video, args.inicio)] if args.video else candidatos(args.videos, args.manifesto)):
        qs = quadros(v, args.n, ini) if v.exists() else []
        if qs:
            video, inicio_s = v, ini
            break
    if not qs:
        if args.video:
            print(f"erro: nenhum quadro lido de {args.video}; passe --video com um vídeo existente", file=sys.stderr)
        else:
            print(f"erro: nenhum vídeo de {args.manifesto} está baixado em {args.videos}; rode "
                  "scripts/dados/baixar_videos.py ou passe --video com um vídeo existente", file=sys.stderr)
        return 1
    onnx = onnx_atualizado(args.pesos)
    det_pt = DetectorLuminarias(args.pesos)
    privacidade = FiltroPrivacidade()

    def completo(det):
        def f(q):
            det.detectar(q)
            medir_cena(luma(q))
            privacidade.aplicar(q)
        return f

    # PyTorch primeiro: ao preparar um modelo ONNX em CPU, o Ultralytics 8.4.215 reduz o PyTorch a 2 threads
    # (o ONNX Runtime usa as suas). Medido depois disso, o PyTorch rodaria com 2 threads, não com 8.
    torch.set_num_threads(8)
    fps = {"detector_pytorch": round(medir(det_pt.detectar, qs), 1),
           "pipeline_pytorch_com_privacidade": round(medir(completo(det_pt), qs), 1)}
    threads_pytorch = torch.get_num_threads()
    # O ONNX é medido no estado em que roda no processar_videos.py: filtro de privacidade (PyTorch) com as
    # threads que sobram depois de carregar o detector ONNX.
    det_onnx = DetectorLuminarias(onnx)
    fps |= {"detector_onnx": round(medir(det_onnx.detectar, qs), 1),
            "pipeline_onnx_com_privacidade": round(medir(completo(det_onnx), qs), 1)}

    resultados = {
        "cpu": platform.processor() or platform.machine(),
        "threads": threads_pytorch,
        "threads_pytorch_com_onnx": torch.get_num_threads(),
        "video": video.name,
        "inicio_s": inicio_s,
        "quadros": len(qs),
        "resolucao": f"{qs[0].shape[1]}x{qs[0].shape[0]}",
        "imgsz": 640,
        "fps": {k: fps[k] for k in ("detector_pytorch", "detector_onnx",
                                    "pipeline_pytorch_com_privacidade", "pipeline_onnx_com_privacidade")},
    }
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(resultados, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
