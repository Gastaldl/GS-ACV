"""Roda o Luz Segura em todos os quadros dos vídeos noturnos baixados (em CPU) e salva vídeos anotados e resumos.

Sem --pesos, usa os pesos finais em PyTorch (models/luzsegura_yolo26n.pt): os vídeos saem do mesmo
modelo das tabelas. O PyTorch e não o ONNX porque, no pipeline completo em CPU, ele foi o mais rápido:
com um modelo ONNX carregado, o Ultralytics limita o PyTorch a 2 threads, e o filtro de privacidade
(PyTorch) fica mais lento (results/avaliacao/fps.json). --pesos models/luzsegura_yolo26n.onnx usa o ONNX.

Saídas:
  data/outputs/videos/<id>_anotado.mp4   (pessoas e veículos desfocados; fora do git, pelo tamanho)
  results/videos/<id>.json                (resumo do trajeto: luminárias únicas, apagadas, trecho escuro, FPS,
                                           pesos usados e SHA-256 dos pesos .pt de origem)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

from luzsegura.pipeline import criar_analisador
from luzsegura.video import processar_video

PESOS_FINAIS = Path("models/luzsegura_yolo26n.pt")


def relativo(caminho: Path) -> str:
    """Caminho relativo ao repositório: o JSON é versionado e não deve levar a pasta local de quem rodou."""
    if not caminho.is_absolute():
        return caminho.as_posix()
    try:
        return caminho.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return caminho.name


def sha256_origem(pesos: Path) -> str:
    """SHA-256 dos pesos .pt de onde o modelo veio (o mesmo registrado em results/modelo_final.json).

    Para um .onnx vale o selo gravado por onnx_atualizado; sem selo, o hash do próprio arquivo.
    """
    selo = pesos.with_name(pesos.name + ".sha256")
    if pesos.suffix == ".onnx" and selo.exists():
        return selo.read_text(encoding="utf-8").strip()
    return hashlib.sha256(pesos.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pesos", type=Path, default=None,
                    help=f"padrão: {PESOS_FINAIS.as_posix()} (PyTorch); aceita o .onnx exportado por benchmark_fps.py")
    ap.add_argument("--manifesto", type=Path, default=Path("data/manifests/videos.csv"))
    ap.add_argument("--videos", type=Path, default=Path("data/raw/videos"))
    ap.add_argument("--saida-videos", type=Path, default=Path("data/outputs/videos"))
    ap.add_argument("--saida-resumos", type=Path, default=Path("results/videos"))
    ap.add_argument("--max-segundos", type=float, default=0, help="0 = vídeo inteiro")
    ap.add_argument("--ids", nargs="*", help="processar só estes ids")
    args = ap.parse_args()

    pesos = args.pesos
    if pesos is None:
        if not PESOS_FINAIS.exists():
            print(f"ERRO: pesos finais não encontrados em {PESOS_FINAIS.as_posix()}; rode "
                  "scripts/avaliacao/comparar_modelos.py ou use --pesos", file=sys.stderr)
            return 1
        pesos = PESOS_FINAIS
    elif not pesos.exists():
        print(f"ERRO: pesos não encontrados: {pesos}", file=sys.stderr)
        return 1
    modelo = {"pesos": relativo(pesos), "sha256": sha256_origem(pesos)}
    analisador = criar_analisador(pesos=pesos, privacidade=True, threads=8)
    args.saida_resumos.mkdir(parents=True, exist_ok=True)
    for v in csv.DictReader(open(args.manifesto, encoding="utf-8")):
        if args.ids and v["id"] not in args.ids:
            continue
        origem = args.videos / v["arquivo"]
        max_quadros = None
        if args.max_segundos:
            import cv2

            cap = cv2.VideoCapture(str(origem))
            max_quadros = int(args.max_segundos * (cap.get(cv2.CAP_PROP_FPS) or 30))
            cap.release()
        # inicio_s do manifesto pula a cartela de abertura (texto sobre fundo preto, que pareceria trecho escuro).
        resumo = processar_video(str(origem), analisador, saida=args.saida_videos / f"{v['id']}_anotado.mp4",
                                 periodo="noite", max_quadros=max_quadros, inicio_s=float(v.get("inicio_s") or 0))
        resumo["fonte"] = {k: v[k] for k in ("autor", "licenca", "local", "pagina")}
        resumo["origem"] = relativo(origem)
        if resumo.get("video_anotado"):
            resumo["video_anotado"] = relativo(Path(resumo["video_anotado"]))
        resumo.update(modelo)
        (args.saida_resumos / f"{v['id']}.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2),
                                                            encoding="utf-8")
        u = resumo["luminarias_unicas"]
        revisao = sum(1 for a in resumo.get("luminarias_apagadas", []) if a.get("requer_revisao_humana"))
        print(f"{v['id']:20} {resumo['duracao_s']:6.1f}s acesas={u['luminaria_acesa']:3d} "
              f"apagadas={u['luminaria_apagada']:3d} (revisão={revisao}) escuro={resumo['segundos_em_trecho_escuro']:5.1f}s "
              f"fps={resumo['fps_processamento_medio']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
