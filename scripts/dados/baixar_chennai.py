"""Baixa o "Comprehensive Dataset for Urban Streetlight Analysis" (Chennai, Índia).

Fonte: https://github.com/Team16Project/Street-Light-Dataset
Artigo: S. et al. (2024), arXiv:2407.01117.

São fotos de celular feitas ao anoitecer, apontadas para luminárias públicas,
separadas em pastas "Working" (acesa) e "Not Working" (apagada). O repositório
não declara licença: as imagens são usadas apenas na avaliação acadêmica, com
citação (nunca no treino), e NÃO são redistribuídas no nosso repositório
(data/ fica fora do git; só o manifesto com as URLs é versionado).

Reprodutibilidade: se o manifesto já existe, o script baixa de novo exatamente as
linhas dele; --nova-amostra lista outra vez as pastas do repositório de origem. O
manifesto passa por deduplicar_manifestos.py (fotos repetidas saem).

Saídas:
  data/raw/chennai/<working|not_working>/<arquivo>.jpg   (reduzidas para no máximo 1280 px)
  data/manifests/chennai_imagens.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
import requests

from deduplicar_manifestos import deduplicar_manifesto  # script vizinho (scripts/dados)

REPO = "Team16Project/Street-Light-Dataset"
PASTAS = {"Working": "working", "Not Working": "not_working"}
API = "https://api.github.com/repos/{repo}/contents/{caminho}"
RAW = "https://raw.githubusercontent.com/{repo}/main/{caminho}"


def listar(sessao: requests.Session, pasta: str) -> list[str]:
    url = API.format(repo=REPO, caminho=quote(f"Raw Dataset/{pasta}"))
    r = sessao.get(url, timeout=30)
    r.raise_for_status()
    return sorted(i["name"] for i in r.json() if i["type"] == "file" and i["name"].lower().endswith(".jpg"))


def baixar_reduzir(sessao: requests.Session, url: str, destino: Path, lado_max: int) -> str:
    if destino.exists() and destino.stat().st_size > 0:
        return "existente"
    for _ in range(3):
        try:
            r = sessao.get(url, timeout=60)
        except requests.RequestException:
            continue
        if r.status_code != 200:
            continue
        img = cv2.imdecode(np.frombuffer(r.content, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return "invalida"
        h, w = img.shape[:2]
        escala = lado_max / max(h, w)
        if escala < 1:
            img = cv2.resize(img, (round(w * escala), round(h * escala)), interpolation=cv2.INTER_AREA)
        destino.parent.mkdir(parents=True, exist_ok=True)
        tmp = destino.with_suffix(".part.jpg")
        cv2.imwrite(str(tmp), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        tmp.replace(destino)
        return "ok"
    return "falhou"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saida", type=Path, default=Path("data/raw/chennai"))
    ap.add_argument("--manifestos", type=Path, default=Path("data/manifests"))
    ap.add_argument("--lado-max", type=int, default=1280)
    ap.add_argument("--paralelo", type=int, default=8)
    ap.add_argument("--nova-amostra", action="store_true",
                    help="lista de novo as pastas do repositório em vez de repetir o manifesto existente")
    args = ap.parse_args()

    sessao = requests.Session()
    sessao.headers["User-Agent"] = "luzsegura-dataset/1.0"
    manifesto = args.manifestos / "chennai_imagens.csv"
    if manifesto.exists() and not args.nova_amostra:
        with manifesto.open(encoding="utf-8", newline="") as f:
            itens = [(r["arquivo"], r["estado_pasta"], r["url_origem"]) for r in csv.DictReader(f)]
        print(f"repetindo as {len(itens)} imagens de {manifesto} (--nova-amostra lista as pastas de novo)")
    else:
        itens = []
        for pasta, rotulo in PASTAS.items():
            nomes = listar(sessao, pasta)
            print(f"{pasta}: {len(nomes)} arquivos")
            itens += [(f"{rotulo}/{nome}", rotulo, RAW.format(repo=REPO, caminho=quote(f"Raw Dataset/{pasta}/{nome}")))
                      for nome in nomes]
    linhas = []
    with ThreadPoolExecutor(max_workers=args.paralelo) as ex:
        tarefas = {ex.submit(baixar_reduzir, sessao, url, args.saida / arquivo, args.lado_max): (arquivo, rotulo, url)
                   for arquivo, rotulo, url in itens}
        for i, fut in enumerate(as_completed(tarefas), 1):
            arquivo, rotulo, url = tarefas[fut]
            linhas.append({"arquivo": arquivo, "estado_pasta": rotulo, "status": fut.result(), "url_origem": url})
            if i % 100 == 0:
                print(f"  {i}/{len(tarefas)}", flush=True)

    ok = [l for l in linhas if l["status"] in ("ok", "existente")]
    print(f"baixadas: {len(ok)} de {len(linhas)}", flush=True)
    if not ok:  # nada baixado (ex.: sem rede): o manifesto existente fica intacto
        print(f"erro: nenhuma imagem baixada; {manifesto} não foi alterado", file=sys.stderr)
        return 1
    if len(ok) < len(linhas):
        # Só ficam no manifesto as fotos que existem em disco: a avaliação não pode ler arquivo ausente.
        print(f"aviso: {len(linhas) - len(ok)} imagem(ns) não baixada(s) saem do manifesto", file=sys.stderr)
    args.manifestos.mkdir(parents=True, exist_ok=True)
    with manifesto.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["arquivo", "estado_pasta", "url_origem"])
        w.writeheader()
        for l in sorted(ok, key=lambda x: x["arquivo"]):
            w.writerow({k: l[k] for k in ("arquivo", "estado_pasta", "url_origem")})
    antes, depois = deduplicar_manifesto(manifesto, args.saida, "estado_pasta",
                                         args.manifestos / "duplicatas_removidas.csv")
    if len(depois) < len(antes):
        print(f"fotos repetidas removidas do manifesto: {len(antes) - len(depois)} (ver duplicatas_removidas.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
