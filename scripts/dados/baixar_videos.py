"""Baixa os vídeos noturnos de licença livre usados na demonstração e nos testes qualitativos.

Nenhum vídeo foi gravado pela equipe: todos vêm do Internet Archive (item sampa_night, Avenida
Paulista, São Paulo), em domínio público (CC0), que permite uso e modificação. Os arquivos ficam em
data/raw/videos (fora do git); o manifesto com autoria e licença é versionado em data/manifests/videos.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import requests

UA = "LuzSeguraDataset/1.0 (projeto academico FIAP; download unico de videos de licenca livre)"
IA = "https://archive.org/download/sampa_night/sampa_1280x720_{n}.mp4"
IA_PAGINA = "https://archive.org/details/sampa_night"

VIDEOS = [
    {"id": f"sao_paulo_{n}", "uso": "teste qualitativo no território (São Paulo, Av. Paulista, 2008)",
     "url": IA.format(n=n), "pagina": IA_PAGINA, "autor": "Ace of Spades / publicvideos.org",
     "licenca": "Domínio público (CC0)", "local": "São Paulo, SP, Brasil",
     # Cada clipe abre com ~6,5 s de cartela de domínio público (texto sobre fundo preto): a análise começa depois.
     "inicio_s": "7"}
    for n in ("00023", "00024", "00027", "00031", "00032", "00035", "00036", "00038", "00045", "00056")
]


def baixar(sessao: requests.Session, url: str, destino: Path) -> int:
    if destino.exists() and destino.stat().st_size > 0:
        return destino.stat().st_size
    tmp = destino.with_suffix(destino.suffix + ".part")
    for tentativa in range(5):
        try:
            with sessao.get(url, stream=True, timeout=60) as r:
                if r.status_code == 429:
                    espera = int(r.headers.get("Retry-After") or 30 * (tentativa + 1))
                    print(f"  HTTP 429 (limite de requisições); nova tentativa em {espera} s", flush=True)
                    time.sleep(espera)
                    continue
                r.raise_for_status()
                with tmp.open("wb") as f:
                    for bloco in r.iter_content(1 << 20):
                        f.write(bloco)
            tmp.replace(destino)
            return destino.stat().st_size
        except requests.RequestException as e:
            print(f"  tentativa {tentativa + 1} falhou: {e}", flush=True)
            time.sleep(10)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--saida", type=Path, default=Path("data/raw/videos"))
    ap.add_argument("--manifestos", type=Path, default=Path("data/manifests"))
    args = ap.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)
    sessao = requests.Session()
    sessao.headers["User-Agent"] = UA
    linhas = []
    for v in VIDEOS:
        ext = Path(v["url"].split("?")[0]).suffix
        destino = args.saida / f"{v['id']}{ext}"
        tamanho = baixar(sessao, v["url"], destino)
        print(f"{v['id']:22} {tamanho / 1e6:8.1f} MB", flush=True)
        if tamanho:
            linhas.append({**v, "arquivo": destino.name})
    # Mesma regra dos outros downloads: falha só sem nenhum vídeo. A demo e os scripts de vídeo pulam
    # os trechos ausentes, então um vídeo indisponível não deve parar scripts/reproduzir.py.
    baixados = {l["id"] for l in linhas}
    faltando = [v["id"] for v in VIDEOS if v["id"] not in baixados]
    if faltando:
        print(f"aviso: vídeos não baixados: {', '.join(faltando)} (a demo pula esses trechos)", file=sys.stderr)
    if not linhas:
        return 1  # nada baixado (ex.: sem rede): o manifesto existente, com os créditos, fica intacto
    args.manifestos.mkdir(parents=True, exist_ok=True)
    with (args.manifestos / "videos.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "arquivo", "uso", "autor", "licenca", "local", "pagina", "url", "inicio_s"])
        w.writeheader()
        w.writerows({k: l[k] for k in w.fieldnames} for l in linhas)
    return 0


if __name__ == "__main__":
    sys.exit(main())
