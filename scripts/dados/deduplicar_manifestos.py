"""Remove dos manifestos de avaliação as fotos repetidas (mesmo conteúdo, byte a byte).

Chennai e SF 311 trazem cópias idênticas da mesma foto, algumas com rótulos opostos (ex.:
working/20240302_051154.jpg e not_working/20240302_051154.jpg). Cada cópia contaria duas vezes nas
métricas, e uma cópia com rótulo trocado garante um erro em qualquer modo de decisão.

Regra, pelo SHA-256 de cada arquivo baixado:
  - cópias com rótulos diferentes (estado_pasta no Chennai, grupo no SF 311): todas saem,
    porque não dá para saber qual rótulo vale;
  - cópias com o mesmo rótulo: fica só a primeira em ordem alfabética do caminho.
As linhas removidas são acrescentadas a data/manifests/duplicatas_removidas.csv, com o motivo.
Rodar de novo não muda nada: os manifestos já ficam sem duplicatas. Os downloads
(baixar_chennai.py e baixar_sf311.py) chamam a mesma rotina ao gravar o manifesto.

Uso: python scripts/dados/deduplicar_manifestos.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from collections import Counter
from pathlib import Path

# (manifesto, pasta em data/raw, coluna do rótulo)
CONJUNTOS = (("chennai_imagens.csv", "chennai", "estado_pasta"), ("sf311.csv", "sf311", "grupo"))
CAMPOS_REGISTRO = ["manifesto", "arquivo", "rotulo", "sha256", "motivo"]


def sha256(caminho: Path) -> str:
    return hashlib.sha256(caminho.read_bytes()).hexdigest()


def deduplicar(linhas: list[dict], pasta: Path, chave: str) -> tuple[list[dict], list[dict]]:
    """Separa as linhas mantidas (na ordem original) das removidas (com sha256 e motivo)."""
    faltando = [r["arquivo"] for r in linhas if not (pasta / r["arquivo"]).is_file()]
    if faltando:
        raise SystemExit(f"{len(faltando)} arquivo(s) do manifesto ausente(s) em {pasta} (ex.: {faltando[0]}); "
                         "rode antes o script de download")
    grupos: dict[str, list[int]] = {}
    for i, r in enumerate(linhas):
        grupos.setdefault(sha256(pasta / r["arquivo"]), []).append(i)
    fora: dict[int, tuple[str, str]] = {}
    for h, idx in grupos.items():
        if len(idx) < 2:
            continue
        idx.sort(key=lambda i: linhas[i]["arquivo"])
        if len({linhas[i][chave] for i in idx}) > 1:
            fora.update({i: (h, "rotulos_conflitantes") for i in idx})
        else:
            fora.update({i: (h, f"copia_de:{linhas[idx[0]]['arquivo']}") for i in idx[1:]})
    mantidas = [r for i, r in enumerate(linhas) if i not in fora]
    removidas = [{"arquivo": linhas[i]["arquivo"], "rotulo": linhas[i][chave], "sha256": h, "motivo": m}
                 for i, (h, m) in sorted(fora.items(), key=lambda kv: linhas[kv[0]]["arquivo"])]
    return mantidas, removidas


def deduplicar_manifesto(manifesto: Path, pasta: Path, chave: str, registro: Path) -> tuple[list[dict], list[dict]]:
    """Reescreve o manifesto sem duplicatas e registra as removidas. Devolve (linhas antes, linhas depois)."""
    with manifesto.open(encoding="utf-8", newline="") as f:
        leitor = csv.DictReader(f)
        campos, linhas = list(leitor.fieldnames or []), list(leitor)
    mantidas, removidas = deduplicar(linhas, pasta, chave)
    if not removidas:
        return linhas, linhas  # nada a remover: o arquivo nem é reescrito
    tmp = manifesto.with_name(manifesto.name + ".part")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(mantidas)
    tmp.replace(manifesto)
    # O registro só ganha linhas novas: rodar de novo depois de uma interrupção não as duplica.
    ja = set()
    if registro.exists():
        with registro.open(encoding="utf-8", newline="") as f:
            ja = {(r["manifesto"], r["arquivo"]) for r in csv.DictReader(f)}
    novas = [{"manifesto": manifesto.name, **r} for r in removidas if (manifesto.name, r["arquivo"]) not in ja]
    if novas:
        cabecalho = not registro.exists() or registro.stat().st_size == 0
        with registro.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=CAMPOS_REGISTRO)
            if cabecalho:
                w.writeheader()
            w.writerows(novas)
    return linhas, mantidas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("data/raw"))
    ap.add_argument("--manifestos", type=Path, default=Path("data/manifests"))
    args = ap.parse_args()

    registro = args.manifestos / "duplicatas_removidas.csv"
    processados = 0
    for nome, pasta, chave in CONJUNTOS:
        manifesto = args.manifestos / nome
        if not manifesto.exists():
            print(f"aviso: {manifesto} não existe; pulado", file=sys.stderr)
            continue
        antes, depois = deduplicar_manifesto(manifesto, args.raw / pasta, chave, registro)
        processados += 1
        c0, c1 = Counter(r[chave] for r in antes), Counter(r[chave] for r in depois)
        print(f"{nome}: {len(antes)} -> {len(depois)} imagens ("
              + ", ".join(f"{k} {c0[k]} -> {c1[k]}" for k in sorted(c0)) + ")")
    return 0 if processados else 1


if __name__ == "__main__":
    sys.exit(main())
