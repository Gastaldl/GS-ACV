"""Resume o treino a partir do results.csv do Ultralytics (épocas, tempo, melhor época e métricas).

Uso:
  python scripts/treino/resumir_treino.py   # v1: runs/treino/luzsegura_yolo26n_v1 -> results/treino
  python scripts/treino/resumir_treino.py --resultados runs/treino/luzsegura_yolo26n_v2/results.csv \\
      --saida results/treino_v2/resumo_treino.json

--resultados aceita o results.csv ou a pasta do treino; --saida aceita o .json ou a pasta de destino.
Saída padrão: results/treino/resumo_treino.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resultados", type=Path, default=Path("runs/treino/luzsegura_yolo26n_v1/results.csv"),
                    help="results.csv do treino (ou a pasta do treino)")
    ap.add_argument("--saida", type=Path, default=Path("results/treino/resumo_treino.json"),
                    help="arquivo .json de saída (ou a pasta onde gravar resumo_treino.json)")
    args = ap.parse_args()
    if args.resultados.is_dir():
        args.resultados = args.resultados / "results.csv"
    if args.saida.suffix.lower() != ".json":
        args.saida = args.saida / "resumo_treino.json"
    if not args.resultados.exists():
        print(f"erro: {args.resultados} não existe (o treino rodou com este --nome?)", file=sys.stderr)
        return 1
    with args.resultados.open(encoding="utf-8") as f:
        linhas = [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]
    if not linhas:
        print(f"erro: {args.resultados} não tem nenhuma época", file=sys.stderr)
        return 1

    def f(r, k):
        return float(r[k])

    # Mesmo critério do Ultralytics 8.4.215 para escolher best.pt: só mAP50-95 (Metric.fitness, pesos [0, 0, 0, 1]).
    # A aptidão vem arredondada a 5 casas, como no CSV, e o treinador regrava best.pt quando uma época iguala a
    # melhor (save_model: best_fitness == fitness): num empate vale a última época, daí o max() de trás para frente.
    melhor = max(reversed(linhas), key=lambda r: f(r, "metrics/mAP50-95(B)"))
    resumo = {
        "epocas_executadas": len(linhas),
        "horas": round(f(linhas[-1], "time") / 3600, 2),
        "melhor_epoca": int(float(melhor["epoch"])),
        "melhor_map50": round(f(melhor, "metrics/mAP50(B)"), 4),
        "melhor_map50_95": round(f(melhor, "metrics/mAP50-95(B)"), 4),
        "melhor_precisao": round(f(melhor, "metrics/precision(B)"), 4),
        "melhor_revocacao": round(f(melhor, "metrics/recall(B)"), 4),
        "por_epoca": [{"epoca": int(float(r["epoch"])), "mAP50": round(f(r, "metrics/mAP50(B)"), 4),
                       "mAP50_95": round(f(r, "metrics/mAP50-95(B)"), 4)} for r in linhas],
    }
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in resumo.items() if k != "por_epoca"}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
