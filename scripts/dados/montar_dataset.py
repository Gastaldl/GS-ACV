"""Monta o dataset YOLO do Luz Segura a partir do Open Images.

Classes: 0 = luminaria_acesa, 1 = luminaria_apagada.

Rótulos:
  - caixas: do Open Images (classe "Street light"), ~94% desenhadas por anotadores humanos (xclick)
    e ~6% propostas por algoritmo e verificadas por humanos (activemil);
  - estado em fotos noturnas e de anoitecer: regra de luminosidade (luzsegura.luminancia),
    descartando imagens com alguma caixa "indefinido";
  - estado em fotos diurnas: apagada (de dia a luminária fica desligada);
  - parte das fotos diurnas vira noite simulada (luzsegura.simulacao), rotulada como apagada.

Faixas de luminosidade média da cena: noite < 55, anoitecer 55-105, dia >= 105.

Divisões:
  - train/val: subconjunto "train" do Open Images (val = 10% de cada grupo);
  - test: subconjuntos oficiais "validation" e "test" do Open Images, mais imagens noturnas e
    de anoitecer do subconjunto "train" reservadas antes do treino (nunca vistas no treino).

Saídas:
  data/dataset/images/{train,val,test}/*.jpg e data/dataset/labels/{train,val,test}/*.txt
  data/dataset/luzsegura.yaml
  data/manifests/dataset_composicao.csv

Os pesos versionados foram treinados na divisão de data/manifests/dataset_composicao.csv. Se a divisão
montada sair diferente (ex.: uma foto que faltou no download ou não abriu embaralha o sorteio), o script
para sem mexer em nada, porque o teste poderia conter fotos do treino desses pesos; --nova-divisao aceita
a divisão nova, para quem vai treinar de novo.
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from luzsegura.config import CLASSES
from luzsegura.simulacao import simular_noite

ACESA, APAGADA = 0, 1


def faixa(media: float) -> str:
    if media < 55:
        return "noite"
    if media < 105:
        return "anoitecer"
    return "dia"


def copiar(origem: Path, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists():
        destino.unlink()
    try:
        os.link(origem, destino)
    except OSError:
        shutil.copy2(origem, destino)


def linhas_yolo(caixas: list[dict], classe_fixa: int | None) -> list[str]:
    linhas = []
    for c in caixas:
        x1, y1, x2, y2 = (float(c[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, y1 = max(0.0, x1), max(0.0, y1)
        x2, y2 = min(1.0, x2), min(1.0, y2)
        w, h = x2 - x1, y2 - y1
        if w < 0.002 or h < 0.002:
            continue
        if classe_fixa is not None:
            cls = classe_fixa
        else:
            cls = ACESA if c["estado"] == "acesa" else APAGADA
        linhas.append(f"{cls} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {w:.6f} {h:.6f}")
    return linhas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/raw/openimages"))
    ap.add_argument("--interim", type=Path, default=Path("data/interim"))
    ap.add_argument("--saida", type=Path, default=Path("data/dataset"))
    ap.add_argument("--manifestos", type=Path, default=Path("data/manifests"))
    ap.add_argument("--n-noite", type=int, default=1350)
    ap.add_argument("--n-anoitecer", type=int, default=850)
    ap.add_argument("--n-dia", type=int, default=300)
    ap.add_argument("--n-dia-simulado", type=int, default=650)
    ap.add_argument("--n-teste-noite", type=int, default=250)
    ap.add_argument("--n-teste-anoitecer", type=int, default=150)
    ap.add_argument("--fracao-val", type=float, default=0.10)
    ap.add_argument("--semente", type=int, default=42)
    ap.add_argument("--nova-divisao", action="store_true",
                    help="aceita uma divisão diferente da versionada (exige treinar de novo: os pesos versionados "
                         "foram treinados na divisão versionada)")
    args = ap.parse_args()

    imgs = {r["image_id"]: r for r in csv.DictReader(open(args.interim / "luminancia_openimages_imagens.csv", encoding="utf-8"))}
    caixas = defaultdict(list)
    for c in csv.DictReader(open(args.interim / "luminancia_openimages_caixas.csv", encoding="utf-8")):
        caixas[c["image_id"]].append(c)

    rnd = random.Random(args.semente)
    rng = np.random.default_rng(args.semente)
    # Primeiro a divisão inteira é planejada e conferida; só depois as imagens são gravadas, na mesma ordem.
    plano = []  # (image_id, split, grupo, simulada, classe_fixa)
    composicao = []

    def gravar(image_id: str, split: str, grupo: str, simulada: bool, classe_fixa: int | None) -> None:
        r = imgs[image_id]
        origem = args.raw / "images" / r["subset"] / f"{image_id}.jpg"
        nome = f"{image_id}_sim" if simulada else image_id
        destino_img = args.saida / "images" / split / f"{nome}.jpg"
        parametros = ""
        if simulada:
            img = cv2.imread(str(origem), cv2.IMREAD_COLOR)
            noite, p = simular_noite(img, rng)
            destino_img.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(destino_img), noite, [cv2.IMWRITE_JPEG_QUALITY, 92])
            parametros = ";".join(f"{k}={v}" for k, v in p.items())
        else:
            copiar(origem, destino_img)
        linhas = linhas_yolo(caixas[image_id], classe_fixa)
        destino_lbl = args.saida / "labels" / split / f"{nome}.txt"
        destino_lbl.parent.mkdir(parents=True, exist_ok=True)
        destino_lbl.write_text("\n".join(linhas) + ("\n" if linhas else ""), encoding="utf-8")
        cont = Counter(int(l.split()[0]) for l in linhas)
        composicao.append({"arquivo": f"{split}/{nome}.jpg", "image_id": image_id, "split": split, "grupo": grupo,
                           "simulada": int(simulada), "parametros_simulacao": parametros,
                           "cena_media": r["cena_media"], "acesas": cont.get(ACESA, 0), "apagadas": cont.get(APAGADA, 0)})

    def sem_indefinido(i: str) -> bool:
        return all(c["estado"] != "indefinido" for c in caixas[i])

    def tem_apagada(i: str) -> bool:
        return any(c["estado"] == "apagada" for c in caixas[i])

    # --- test: subconjuntos oficiais validation + test ---
    for i, r in sorted(imgs.items()):
        if r["subset"] == "train":
            continue
        f = faixa(float(r["cena_media"]))
        if f == "dia":
            plano.append((i, "test", "dia", False, APAGADA))
        elif sem_indefinido(i):
            plano.append((i, "test", f, False, None))

    # --- train/val: subconjunto train ---
    treino = [i for i, r in imgs.items() if r["subset"] == "train"]
    por_faixa = defaultdict(list)
    for i in sorted(treino):
        por_faixa[faixa(float(imgs[i]["cena_media"]))].append(i)

    def escolher(candidatos: list[str], n: int, priorizar=None) -> list[str]:
        candidatos = sorted(candidatos)
        rnd.shuffle(candidatos)
        if priorizar:
            candidatos.sort(key=lambda i: not priorizar(i))  # estável: prioritários primeiro
        return candidatos[:n]

    grupos = {}
    for nome_faixa, n_treino, n_teste in (("noite", args.n_noite, args.n_teste_noite),
                                          ("anoitecer", args.n_anoitecer, args.n_teste_anoitecer)):
        limpas = [i for i in por_faixa[nome_faixa] if sem_indefinido(i)]
        com_apagada = escolher([i for i in limpas if tem_apagada(i)], len(limpas))
        sem_apagada = escolher([i for i in limpas if not tem_apagada(i)], len(limpas))
        # Teste reservado ANTES do treino: 25% das imagens com luminária apagada + o restante sem.
        n_teste_ap = min(len(com_apagada) // 4, n_teste)
        teste = com_apagada[:n_teste_ap] + sem_apagada[: n_teste - n_teste_ap]
        for i in teste:
            plano.append((i, "test", f"{nome_faixa}_reservado", False, None))
        reservadas = set(teste)
        restantes = [i for i in limpas if i not in reservadas]
        grupos[nome_faixa] = (escolher(restantes, n_treino, tem_apagada), None, False)
    dia = escolher(por_faixa["dia"], args.n_dia + args.n_dia_simulado)
    grupos["dia"] = (dia[: args.n_dia], APAGADA, False)
    grupos["noite_simulada"] = (dia[args.n_dia:], APAGADA, True)

    for grupo, (ids, classe_fixa, simulada) in grupos.items():
        ids = list(ids)
        rnd.shuffle(ids)
        n_val = max(1, round(len(ids) * args.fracao_val))
        for k, i in enumerate(ids):
            plano.append((i, "val" if k < n_val else "train", grupo, simulada, classe_fixa))

    versionada = args.manifestos / "dataset_composicao.csv"
    if versionada.exists():
        with versionada.open(newline="", encoding="utf-8") as f:
            antes = {(r["image_id"], r["simulada"]): r["split"] for r in csv.DictReader(f)}
        agora = {(i, str(int(simulada))): split for i, split, _, simulada, _ in plano}
        diferentes = sorted(k for k in antes.keys() | agora.keys() if antes.get(k) != agora.get(k))
        if diferentes and not args.nova_divisao:
            faltam = sum(k not in agora for k in diferentes)
            print(f"erro: a divisão montada difere da versionada ({versionada}) em {len(diferentes)} "
                  f"{'imagem' if len(diferentes) == 1 else 'imagens'} ({faltam} ficaram de fora da nova; ex.: "
                  f"{', '.join(i for i, _ in diferentes[:3])}). Em geral é foto que faltou no download ou não "
                  "abriu: rode de novo scripts/dados/baixar_openimages.py (ele baixa só o que falta) e "
                  "scripts/dados/analisar_luminancia.py. Os pesos versionados foram treinados na divisão "
                  "versionada; com outra, o teste poderia conter fotos do treino deles. Para treinar de novo com "
                  "a divisão nova, use --nova-divisao (scripts/reproduzir.py já passa essa opção quando treina o "
                  "v1). Nada foi alterado.", file=sys.stderr)
            return 1
        if diferentes:
            print(f"aviso: divisão nova, diferente da versionada em {len(diferentes)} "
                  f"{'imagem' if len(diferentes) == 1 else 'imagens'} (--nova-divisao); os pesos precisam ser "
                  "treinados de novo nela", file=sys.stderr)

    if args.saida.exists():
        shutil.rmtree(args.saida)
    for passo in plano:
        gravar(*passo)

    yaml = args.saida / "luzsegura.yaml"
    yaml.write_text(
        f"# Gerado por scripts/dados/montar_dataset.py\npath: {args.saida.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\nnames:\n"
        + "".join(f"  {k}: {n}\n" for k, n in enumerate(CLASSES)),
        encoding="utf-8",
    )
    args.manifestos.mkdir(parents=True, exist_ok=True)
    with (args.manifestos / "dataset_composicao.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(composicao[0].keys()))
        w.writeheader()
        w.writerows(sorted(composicao, key=lambda x: x["arquivo"]))

    resumo = defaultdict(lambda: Counter())
    for c in composicao:
        chave = (c["split"], c["grupo"])
        resumo[chave]["imagens"] += 1
        resumo[chave]["acesas"] += c["acesas"]
        resumo[chave]["apagadas"] += c["apagadas"]
    print(f"{'split':6} {'grupo':15} {'imagens':>8} {'acesas':>8} {'apagadas':>9}")
    for (s, g), c in sorted(resumo.items()):
        print(f"{s:6} {g:15} {c['imagens']:8d} {c['acesas']:8d} {c['apagadas']:9d}")
    tot = Counter()
    for c in composicao:
        tot[c["split"]] += 1
    print("total:", dict(tot), "| yaml:", yaml)
    return 0


if __name__ == "__main__":
    sys.exit(main())
