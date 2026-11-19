"""Avalia o detector no conjunto de teste (imagens nunca vistas no treino).

Métricas do Ultralytics (precisão, revocação, mAP50 e mAP50-95) para o teste inteiro e
por faixa de luminosidade da cena (noite, anoitecer, dia), com matriz de confusão e
curvas precisão-revocação.

Precisão e revocação aparecem de dois jeitos:
  - precisao/revocacao: as do Ultralytics, no limiar de confiança que maximiza o F1 médio neste
    mesmo teste (limiar_f1); são otimistas, porque o limiar é escolhido olhando o próprio teste;
  - precisao_op/revocacao_op: no ponto de operação do sistema (Limiares.conf_minima = 0,30), contadas
    nas mesmas detecções e com o mesmo pareamento com o gabarito (IoU 0,5) usados no mAP.
O mAP só pontua classes com caixas no gabarito do recorte: no "dia" não há luminária acesa no
gabarito, então uma detecção "acesa" de dia (o erro por trás do aviso de acesa durante o dia) não
entra no mAP. Essas detecções são contadas à parte, no ponto de operação
(falsos_positivos_sem_gabarito_op). Cada classe traz também o número de caixas do gabarito.

Saídas: results/avaliacao/deteccao.json e results/avaliacao/deteccao_<grupo>/*.png
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from ultralytics import YOLO
from ultralytics.utils.metrics import smooth

from luzsegura.config import CLASSES, LIMIARES_PADRAO

GRUPOS = {
    "todos": None,
    "noite": ("noite", "noite_reservado"),
    "anoitecer": ("anoitecer", "anoitecer_reservado"),
    "dia": ("dia",),
}


def relativo(caminho: str | Path) -> str:
    """Caminho relativo ao repositório (sem a pasta local de quem rodou); fora dele, só o nome."""
    p = Path(caminho)
    if not p.is_absolute():
        return p.as_posix()
    try:
        return p.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return p.name


def procedencia(pesos: Path) -> dict:
    """Identifica os pesos avaliados: caminho, SHA-256 e dataset de treino gravado no checkpoint."""
    dados = ""
    if pesos.suffix == ".pt":
        ck = torch.load(pesos, map_location="cpu", weights_only=False)
        dados = str((ck.get("train_args") or {}).get("data") or "").replace("\\", "/")
    return {"pesos": relativo(pesos), "sha256_pesos": hashlib.sha256(pesos.read_bytes()).hexdigest(),
            "dados_treino": relativo(dados) if dados else ""}


def guardar_estatisticas(modelo: YOLO) -> dict:
    """Copia, ao fim de cada lote da validação, as listas por imagem (acerto, confiança e classe de cada detecção).

    O Ultralytics 8.4.215 apaga essas listas ao calcular o mAP (get_stats chama DetMetrics.clear_stats). As
    curvas que sobram (p_curve, r_curve) interpolam entre detecções e, no limiar de 0,30, erram até um passo
    de detecção, o que pesa em classes com poucas caixas; com a cópia, a conta no ponto de operação é exata.
    """
    copia: dict = {}
    modelo.add_callback("on_val_batch_end", lambda v: copia.update({k: list(x) for k, x in v.metrics.stats.items()}))
    return copia


def media(valores: list) -> float | None:
    definidos = [v for v in valores if v is not None]
    return round(float(np.mean(definidos)), 4) if definidos else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pesos", type=Path, default=Path("models/luzsegura_yolo26n.pt"))
    ap.add_argument("--dataset", type=Path, default=Path("data/dataset"))
    ap.add_argument("--manifesto", type=Path, default=Path("data/manifests/dataset_composicao.csv"))
    ap.add_argument("--saida", type=Path, default=Path("results/avaliacao"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--sufixo", default="", help="sufixo dos arquivos de saída (ex.: _v1)")
    args = ap.parse_args()
    if not args.pesos.exists():
        print(f"erro: pesos não encontrados: {args.pesos} (rode scripts/treino/treinar.py)", file=sys.stderr)
        return 1
    torch.set_num_threads(8)

    teste = [r for r in csv.DictReader(open(args.manifesto, encoding="utf-8")) if r["split"] == "test"]
    raiz = args.dataset.resolve()
    args.saida.mkdir(parents=True, exist_ok=True)
    op = LIMIARES_PADRAO.conf_minima
    resultados = {**procedencia(args.pesos), "conf_operacao": op}
    for nome, grupos in GRUPOS.items():
        linhas = [r for r in teste if grupos is None or r["grupo"] in grupos]
        if not linhas:
            continue
        lista = raiz / f"test_{nome}.txt"
        lista.write_text("\n".join(str(raiz / "images" / r["arquivo"]) for r in linhas) + "\n", encoding="utf-8")
        yaml = raiz / f"test_{nome}.yaml"
        yaml.write_text(f"path: {raiz.as_posix()}\ntrain: images/train\nval: {lista.as_posix()}\nnames:\n"
                        + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)), encoding="utf-8")
        modelo = YOLO(str(args.pesos))
        est = guardar_estatisticas(modelo)
        # conf padrão da validação (0,001): o mAP precisa da curva inteira; o ponto de operação é filtrado depois.
        m = modelo.val(data=str(yaml), split="val", imgsz=args.imgsz, batch=16, device="cpu",
                       plots=True, project=str((Path("runs") / "avaliacao").resolve()),
                       name=f"deteccao{args.sufixo}_{nome}", exist_ok=True, verbose=False)

        presentes = m.box.ap_class_index.tolist()  # classes com caixa no gabarito, na ordem de p, r e ap
        conf, classe = np.concatenate(est["conf"]), np.concatenate(est["pred_cls"]).astype(int)
        acerto = np.concatenate(est["tp"])[:, 0].astype(bool)  # coluna IoU 0,5, a mesma do mAP50
        gabarito = np.concatenate(est["target_cls"]).astype(int)
        no_op = conf >= op
        por_classe = {}
        for k, idx in enumerate(presentes):
            sel = no_op & (classe == idx)
            n_det, n_vp, n_gab = int(sel.sum()), int(acerto[sel].sum()), int((gabarito == idx).sum())
            por_classe[CLASSES[idx]] = {
                "instancias": n_gab,
                "precisao": round(float(m.box.p[k]), 4), "revocacao": round(float(m.box.r[k]), 4),
                "deteccoes_op": n_det, "acertos_op": n_vp,
                # Sem nenhuma detecção no ponto de operação a precisão fica indefinida (null), não 0 nem 1.
                "precisao_op": round(n_vp / n_det, 4) if n_det else None,
                "revocacao_op": round(n_vp / n_gab, 4) if n_gab else None,
                "mAP50": round(float(m.box.ap50[k]), 4), "mAP50_95": round(float(m.box.ap[k]), 4)}
        instancias = {c: sum(int(r["acesas" if c == "luminaria_acesa" else "apagadas"]) for r in linhas) for c in CLASSES}
        resultados[nome] = {
            "imagens": len(linhas), "instancias": instancias,
            "precisao": round(float(m.box.mp), 4), "revocacao": round(float(m.box.mr), 4),
            # Mesmo índice que o Ultralytics usa para precisao/revocacao: máximo do F1 médio suavizado.
            "limiar_f1": round(float(m.box.px[smooth(m.box.f1_curve.mean(0), 0.1).argmax()]), 3) if presentes else None,
            "precisao_op": media([v["precisao_op"] for v in por_classe.values()]),
            "revocacao_op": media([v["revocacao_op"] for v in por_classe.values()]),
            "mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4),
            "por_classe": por_classe, "tempo_ms_por_imagem": {k: round(v, 1) for k, v in m.speed.items()},
        }

        # Classe sem caixa no gabarito deste recorte (ex.: acesa no "dia"): toda detecção dela no ponto de
        # operação é falso positivo que o mAP não vê. As listas são por imagem, o que dá as imagens afetadas.
        ausentes = [i for i in range(len(CLASSES)) if i not in presentes]
        if ausentes:
            fora = {CLASSES[i]: int((no_op & (classe == i)).sum()) for i in ausentes}
            afetadas = sum(bool(np.isin(c[cf >= op].astype(int), ausentes).any())
                           for cf, c in zip(est["conf"], est["pred_cls"]))
            resultados[nome]["falsos_positivos_sem_gabarito_op"] = {**fora, "imagens_afetadas": afetadas}

        destino = args.saida / f"deteccao{args.sufixo}_{nome}"
        destino.mkdir(parents=True, exist_ok=True)
        for png in Path(m.save_dir).glob("*.png"):
            shutil.copy2(png, destino / png.name)
        r = resultados[nome]
        print(f"{nome:10} imgs={len(linhas):4d} P={m.box.mp:.3f} R={m.box.mr:.3f} (limiar {r['limiar_f1']}) "
              f"P@{op:.2f}={r['precisao_op']} R@{op:.2f}={r['revocacao_op']} mAP50={m.box.map50:.3f} "
              f"mAP50-95={m.box.map:.3f} {por_classe} {r.get('falsos_positivos_sem_gabarito_op', '')}", flush=True)

    (args.saida / f"deteccao{args.sufixo}.json").write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
