"""Avaliação de ponta a ponta por imagem: o sistema acusa a luminária apagada?

Conjuntos (nenhum usado no treino, nem no do v1 nem no autotreinamento do v2; os manifestos já vêm sem
fotos duplicadas, ver scripts/dados/deduplicar_manifestos.py):
  - chennai: 816 fotos feitas ao anoitecer (Índia), 527 de luminárias funcionando e 289 com defeito.
    O rótulo é a pasta do dataset original. Período informado: noite (horário em que deveriam estar acesas).
  - sf311_noite: 398 fotos reais de chamados noturnos de luminária queimada (San Francisco 311).
    Período calculado pela posição do Sol no horário e no local do chamado.
  - sf311_dia: 60 fotos de chamados diurnos de luminária "acesa o tempo todo" (regra de desperdício).

Três formas de decidir que há falha na imagem:
  - modelo:  pelo menos uma detecção da classe luminaria_apagada;
  - brilho:  pelo menos uma caixa detectada cuja regra de brilho diz "apagada";
  - sistema: diagnóstico final luminaria_apagada (modelo conferido pelo brilho, com revisão humana quando discordam).

Cada imagem passa pela mesma redução de tamanho da API e do app (luzsegura.pipeline.reduzir). Em Chennai,
falso_alarme_piso_funcionando é a fração das fotos de luminárias funcionando diagnosticadas como iluminação
insuficiente ou trecho escuro (o brilho do chão é medido na faixa inferior da foto, que pode ser céu ou fachada).
O JSON registra os pesos avaliados (caminho e SHA-256) e o dataset de treino gravado no checkpoint.

Saídas: results/avaliacao/imagens.json e results/avaliacao/imagens_<conjunto>.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import cv2
import torch

from luzsegura.config import CLASSE_APAGADA, LIMIARES_PADRAO
from luzsegura.detector import DetectorLuminarias
from luzsegura.diagnostico import ACESA_DE_DIA, INSUFICIENTE, LUMINARIA_APAGADA, TRECHO_ESCURO, Contexto, diagnosticar
from luzsegura.luminancia import APAGADA, luma, medir_cena
from luzsegura.pipeline import reduzir

SF = ZoneInfo("America/Los_Angeles")


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


def avaliar(detector, caminho: Path, ctx: Contexto) -> dict:
    img = cv2.imread(str(caminho), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"imagem ausente ou ilegível: {caminho} (rode os scripts de download)")
    img = reduzir(img)
    lums = detector.detectar(img)
    cena = medir_cena(luma(img))
    r = diagnosticar(lums, cena, img.shape[1], img.shape[0], ctx)
    validas = [l for l in lums if l.confianca >= LIMIARES_PADRAO.conf_minima]
    return {
        "acesas_modelo": sum(1 for l in validas if l.classe != CLASSE_APAGADA),
        "apagadas_modelo": sum(1 for l in validas if l.classe == CLASSE_APAGADA),
        "apagadas_brilho": sum(1 for l in validas if l.medida.estado == APAGADA),
        "falha_modelo": any(l.classe == CLASSE_APAGADA for l in validas),
        "falha_brilho": any(l.medida.estado == APAGADA for l in validas),
        "falha_sistema": r["diagnostico"] == LUMINARIA_APAGADA,
        "chamado_automatico": r["diagnostico"] == LUMINARIA_APAGADA and not r["requer_revisao_humana"],
        "diagnostico": r["diagnostico"], "revisao": r["requer_revisao_humana"], "periodo": r["periodo"],
        "conf_max_apagada": round(max((l.confianca for l in validas if l.classe == CLASSE_APAGADA), default=0.0), 3),
    }


def binario(linhas: list[dict], chave: str) -> dict:
    tp = sum(1 for l in linhas if l["rotulo"] == 1 and l[chave])
    fn = sum(1 for l in linhas if l["rotulo"] == 1 and not l[chave])
    fp = sum(1 for l in linhas if l["rotulo"] == 0 and l[chave])
    tn = sum(1 for l in linhas if l["rotulo"] == 0 and not l[chave])
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"vp": tp, "fn": fn, "fp": fp, "vn": tn, "precisao": round(p, 4), "revocacao": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0, "acuracia": round((tp + tn) / len(linhas), 4)}


def distribuicao(linhas: list[dict]) -> dict:
    """Quantas imagens receberam cada diagnóstico, do mais para o menos frequente."""
    return dict(Counter(l["diagnostico"] for l in linhas).most_common())


def gravar_csv(caminho: Path, linhas: list[dict]) -> None:
    with caminho.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(linhas[0].keys()))
        w.writeheader()
        w.writerows(linhas)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pesos", type=Path, default=Path("models/luzsegura_yolo26n.pt"))
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--manifestos", type=Path, default=Path("data/manifests"))
    ap.add_argument("--saida", type=Path, default=Path("results/avaliacao"))
    ap.add_argument("--sufixo", default="", help="sufixo dos arquivos de saída (ex.: _v1, _v2)")
    args = ap.parse_args()
    if not args.pesos.exists():
        print(f"erro: pesos não encontrados: {args.pesos} (rode scripts/treino/treinar.py)", file=sys.stderr)
        return 1
    torch.set_num_threads(8)
    detector = DetectorLuminarias(args.pesos)
    args.saida.mkdir(parents=True, exist_ok=True)
    resumo = procedencia(args.pesos)

    # Chennai
    linhas = []
    for r in csv.DictReader(open(args.manifestos / "chennai_imagens.csv", encoding="utf-8")):
        res = avaliar(detector, args.raw / "chennai" / r["arquivo"], Contexto(periodo_informado="noite"))
        linhas.append({"arquivo": r["arquivo"], "rotulo": int(r["estado_pasta"] == "not_working"), **res})
    gravar_csv(args.saida / f"imagens_chennai{args.sufixo}.csv", linhas)
    funcionando = [l for l in linhas if l["rotulo"] == 0]
    resumo["chennai"] = {"imagens": len(linhas), "com_defeito": sum(l["rotulo"] for l in linhas),
                         **{modo: binario(linhas, f"falha_{modo}") for modo in ("modelo", "brilho", "sistema")},
                         "chamado_automatico": binario(linhas, "chamado_automatico"),
                         "diagnosticos": distribuicao(linhas),
                         "diagnosticos_por_rotulo": {"funcionando": distribuicao(funcionando),
                                                     "com_defeito": distribuicao([l for l in linhas if l["rotulo"] == 1])},
                         # Luminária funcionando e mesmo assim chão "escuro": alarme falso do critério de piso.
                         "falso_alarme_piso_funcionando": round(
                             sum(l["diagnostico"] in (INSUFICIENTE, TRECHO_ESCURO) for l in funcionando) / len(funcionando), 4)
                         if funcionando else 0.0}
    print("chennai", json.dumps(resumo["chennai"], ensure_ascii=False), flush=True)

    # SF 311
    sf = list(csv.DictReader(open(args.manifestos / "sf311.csv", encoding="utf-8")))
    for grupo, nome in (("noturno_queimada", "sf311_noite"), ("diurno_acesa", "sf311_dia")):
        linhas = []
        for r in (x for x in sf if x["grupo"] == grupo):
            # aberto_em vem no horário local de San Francisco, sem fuso: o fuso é anexado aqui (a altura
            # do Sol recusa horário sem fuso).
            quando = datetime.fromisoformat(r["aberto_em"]).replace(tzinfo=SF)
            ctx = Contexto(capturado_em=quando, lat=float(r["lat"]), lon=float(r["lon"]))
            res = avaliar(detector, args.raw / "sf311" / r["arquivo"], ctx)
            linhas.append({"arquivo": r["arquivo"], **res})
        gravar_csv(args.saida / f"imagens_{nome}{args.sufixo}.csv", linhas)
        n = len(linhas)
        resumo[nome] = {
            "imagens": n,
            "periodo": {p: sum(1 for l in linhas if l["periodo"] == p) for p in ("noite", "transicao", "dia")},
            "acusou_falha_modelo": round(sum(l["falha_modelo"] for l in linhas) / n, 4),
            "acusou_falha_brilho": round(sum(l["falha_brilho"] for l in linhas) / n, 4),
            "acusou_falha_sistema": round(sum(l["falha_sistema"] for l in linhas) / n, 4),
            "chamado_automatico": round(sum(l["chamado_automatico"] for l in linhas) / n, 4),
            "detectou_alguma_luminaria": round(sum((l["acesas_modelo"] + l["apagadas_modelo"]) > 0 for l in linhas) / n, 4),
            "diagnosticos": distribuicao(linhas),
        }
        if nome == "sf311_dia":
            resumo[nome]["acusou_acesa_de_dia"] = round(sum(l["diagnostico"] == ACESA_DE_DIA for l in linhas) / n, 4)
        print(nome, json.dumps(resumo[nome], ensure_ascii=False), flush=True)

    (args.saida / f"imagens{args.sufixo}.json").write_text(json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
