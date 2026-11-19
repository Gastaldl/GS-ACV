"""Baixa as anotações do Open Images V7 e guarda só as linhas de treino da classe "Street light" (/m/033rq4).

Os dois arquivos de treino (caixas, 2,1 GB, e metadados das imagens) são lidos em fluxo e filtrados sem
serem gravados inteiros. Saídas em data/raw/openimages (fora do git), lidas por scripts/dados/baixar_openimages.py:
  validation-annotations-bbox.csv, test-annotations-bbox.csv, class-descriptions-boxable.csv,
  validation-images-with-rotation.csv, test-images-with-rotation.csv   (completos, byte a byte como vêm)
  train-streetlight-bbox.csv     (linhas das caixas de treino com /m/033rq4, sem cabeçalho)
  train-streetlight-ids.txt      (ImageID dessas caixas, em ordem e sem repetição)
  train-streetlight-images.csv   (linhas dos metadados de treino dessas imagens, sem cabeçalho)
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import requests

BASE = "https://storage.googleapis.com/openimages"
COMPLETOS = [f"{BASE}/v5/validation-annotations-bbox.csv", f"{BASE}/v5/test-annotations-bbox.csv",
             f"{BASE}/v5/class-descriptions-boxable.csv",
             f"{BASE}/2018_04/validation/validation-images-with-rotation.csv",
             f"{BASE}/2018_04/test/test-images-with-rotation.csv"]
CAIXAS_TREINO = f"{BASE}/v6/oidv6-train-annotations-bbox.csv"
IMAGENS_TREINO = f"{BASE}/2018_04/train/train-images-boxable-with-rotation.csv"
CLASSE_STREET_LIGHT = b"/m/033rq4"


def linhas(sessao: requests.Session, url: str, limite: int = 0) -> Iterator[bytes]:
    """Lê o arquivo em fluxo e devolve as linhas em bytes, cada uma com o b"\\n" do fim (a última vem sem ele
    se o arquivo não terminar em quebra de linha); com limite, para depois de tantas linhas."""
    with sessao.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        resto, n = b"", 0
        for bloco in r.iter_content(1 << 20):
            partes = (resto + bloco).split(b"\n")
            resto = partes.pop()
            for linha in partes:
                yield linha + b"\n"
                n += 1
                if n == limite:
                    return
        if resto:
            yield resto


def gravar(sessao: requests.Session, url: str, destino: Path, manter: Callable[[bytes], bool] | None = None,
           limite: int = 0) -> int:
    """Grava em destino as linhas do arquivo remoto (todas, ou só as que passam em manter) e devolve quantas
    gravou. Se a rede falhar, tenta de novo até 3 vezes, recomeçando o arquivo do zero."""
    tmp = destino.with_name(destino.name + ".part")
    for tentativa in range(4):
        try:
            n = 0
            with tmp.open("wb") as f:
                for linha in linhas(sessao, url, limite):
                    if manter is None:
                        f.write(linha)  # cópia fiel do arquivo
                        n += 1
                    elif manter(linha):
                        f.write(linha if linha.endswith(b"\n") else linha + b"\n")  # linha filtrada sempre com quebra
                        n += 1
            tmp.replace(destino)
            print(f"{destino.name}: {n} linhas", flush=True)
            return n
        except requests.RequestException as e:
            if tentativa == 3:
                raise
            print(f"  {destino.name}: tentativa {tentativa + 1} falhou ({e}); nova tentativa em 10 s", flush=True)
            time.sleep(10)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--saida", type=Path, default=Path("data/raw/openimages"))
    ap.add_argument("--max-linhas", type=int, default=0,
                    help="lê só as primeiras linhas de cada arquivo, para um teste rápido (0 = todas)")
    args = ap.parse_args()
    pasta: Path = args.saida
    pasta.mkdir(parents=True, exist_ok=True)
    sessao = requests.Session()
    sessao.headers["User-Agent"] = "luzsegura-dataset/1.0"
    try:
        for url in COMPLETOS:
            gravar(sessao, url, pasta / url.rsplit("/", 1)[1], limite=args.max_linhas)
        caixas = pasta / "train-streetlight-bbox.csv"
        n_caixas = gravar(sessao, CAIXAS_TREINO, caixas, lambda linha: CLASSE_STREET_LIGHT in linha, args.max_linhas)
        if not n_caixas:
            print("falhou: nenhuma caixa da classe Street light (/m/033rq4) nas anotações de treino", file=sys.stderr)
            return 1
        # ImageID é a primeira coluna; os IDs ficam em ordem e sem repetição.
        ids = sorted({linha.split(b",", 1)[0] for linha in caixas.read_bytes().split(b"\n") if linha})
        (pasta / "train-streetlight-ids.txt").write_bytes(b"".join(i + b"\n" for i in ids))
        procurados = set(ids)
        n_imagens = gravar(sessao, IMAGENS_TREINO, pasta / "train-streetlight-images.csv",
                           lambda linha: linha.split(b",", 1)[0] in procurados, args.max_linhas)
    except requests.RequestException as e:
        print(f"falhou: {e}", file=sys.stderr)
        return 1
    print(f"caixas de treino: {n_caixas} | imagens: {len(ids)}")
    # Sem nenhum metadado dessas imagens, o baixar_openimages.py descartaria todo o treino. Num teste com
    # --max-linhas, porém, as primeiras linhas dos metadados podem não ter nenhuma delas.
    if not n_imagens and not args.max_linhas:
        print("falhou: nenhuma linha dos metadados de treino é de imagem com Street light", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
