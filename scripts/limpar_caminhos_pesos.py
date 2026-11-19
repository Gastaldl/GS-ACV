"""Troca os caminhos absolutos gravados em pesos do Ultralytics (.pt) por caminhos relativos ao repositório.

O Ultralytics guarda no checkpoint os argumentos do treino: em train_args e em model.args ficam data, project
e save_dir com o caminho completo da máquina em que o modelo foi treinado (inclusive o nome do usuário), e em
git.root, a pasta do repositório. Este script regrava cada arquivo no lugar com esses caminhos relativos à
raiz do repositório e com "/", sem tocar nos pesos da rede; scripts/treino/treinar.py faz o mesmo ao gravar
os melhores pesos em models/.

  python scripts/limpar_caminhos_pesos.py models/luzsegura_yolo26n.pt models/luzsegura_yolo26n_v1.pt

Os bytes do arquivo mudam, e com eles o SHA-256: os resultados que registram o hash dos pesos (em results/)
e o selo do .onnx precisam ser gerados de novo.
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path, PureWindowsPath

import torch

RAIZ = Path(__file__).resolve().parents[1]
CAMINHOS = ("data", "project", "save_dir")  # argumentos que são caminhos mesmo quando relativos (podem vir com "\")


def relativo(caminho: str, raiz: Path) -> str:
    """Caminho relativo à raiz, com "/"; um caminho absoluto fora da raiz volta como veio."""
    p = Path(caminho)
    if p.is_absolute():
        try:
            p = p.relative_to(raiz)
        except ValueError:
            return caminho
    return PureWindowsPath(p).as_posix()


def limpar_caminhos(ck: dict, raiz: Path = RAIZ) -> list[str]:
    """Troca no checkpoint, no lugar, os caminhos dos argumentos do treino (train_args e model.args) e a pasta
    do repositório (git.root) por caminhos relativos à raiz. Devolve as trocas feitas ("onde: antes -> depois")."""
    dicionarios = [("train_args", ck.get("train_args")), ("git", ck.get("git"))]
    dicionarios += [(f"{k}.args", getattr(ck.get(k), "args", None)) for k in ("model", "ema")]
    trocas = []
    for nome, d in dicionarios:
        if not isinstance(d, dict):
            continue
        for chave, antes in d.items():
            if isinstance(antes, str) and antes and (chave in CAMINHOS or Path(antes).is_absolute()):
                depois = relativo(antes, raiz)
                if depois != antes:
                    d[chave] = depois
                    trocas.append(f"{nome}[{chave}]: {antes} -> {depois}")
    return trocas


def gravar_pesos(ck: dict, destino: Path) -> None:
    """Grava o checkpoint sem deixar arquivo pela metade. Salvo primeiro na memória (como faz o Ultralytics), o
    conteúdo não depende do nome do arquivo: o mesmo checkpoint gravado com nomes diferentes sai idêntico."""
    buffer = io.BytesIO()
    torch.save(ck, buffer)
    tmp = destino.with_name(destino.name + ".part")
    tmp.write_bytes(buffer.getvalue())
    tmp.replace(destino)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pesos", type=Path, nargs="+", help="arquivos .pt, regravados no lugar")
    ap.add_argument("--raiz", type=Path, default=RAIZ,
                    help="pasta a que os caminhos ficam relativos (padrão: a raiz deste repositório)")
    args = ap.parse_args()
    for arquivo in args.pesos:
        # Pesos do próprio projeto: o checkpoint do Ultralytics guarda objetos Python, daí weights_only=False.
        ck = torch.load(arquivo, map_location="cpu", weights_only=False)
        trocas = limpar_caminhos(ck, args.raiz.resolve())
        if trocas:
            gravar_pesos(ck, arquivo)
        print(f"{arquivo}: {len(trocas)} caminhos trocados" + "".join(f"\n  {t}" for t in trocas), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
