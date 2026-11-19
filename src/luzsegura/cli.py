"""Linha de comando do Luz Segura (visão).

  luzsegura foto CAMINHO [--saida anotada.jpg] [--json resultado.json] [--lat --lon --quando --periodo]
  luzsegura video CAMINHO|--webcam N [--saida anotado.mp4] [--resumo resumo.json] [--mostrar]
  luzsegura api [--host 127.0.0.1 --porta 8000]
  luzsegura app [--host 127.0.0.1 --porta 7860]

As opções --pesos, --imgsz e --sem-privacidade valem para todos os subcomandos e podem vir antes
ou depois deles; a API e o app recusam --sem-privacidade.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np


def _analisador(args):
    from .pipeline import criar_analisador

    return criar_analisador(pesos=args.pesos, privacidade=not args.sem_privacidade, imgsz=args.imgsz)


def _data_hora(texto: str) -> datetime:
    """Tipo de --quando. Sem fuso, o horário seria lido como UTC: 3 h de erro na altura do Sol em Brasília."""
    try:
        quando = datetime.fromisoformat(texto)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"data e hora inválidas ({texto}); use ISO 8601 com fuso, ex.: 2026-11-18T21:30:00-03:00") from None
    if quando.utcoffset() is None:
        raise argparse.ArgumentTypeError(f"falta o fuso em {texto}, ex.: 2026-11-18T21:30:00-03:00")
    return quando


def _gravar_texto(caminho: Path, texto: str) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(texto, encoding="utf-8")


def cmd_foto(args) -> int:
    from .diagnostico import Contexto

    # Lê e grava os bytes pelo Python: no Windows, cv2.imread e cv2.imwrite não lidam com caminhos
    # acentuados ("Avenida São João.jpg").
    try:
        img = cv2.imdecode(np.fromfile(str(args.caminho), np.uint8), cv2.IMREAD_COLOR)
    except (OSError, cv2.error):  # arquivo ausente, ilegível ou vazio
        img = None
    if img is None:
        print(f"Não consegui abrir {args.caminho}", file=sys.stderr)
        return 1
    ctx = Contexto(origem="foto", capturado_em=args.quando, lat=args.lat, lon=args.lon,
                   periodo_informado=None if args.periodo == "auto" else args.periodo)
    resultado, anotada = _analisador(args).analisar(img, ctx)
    texto = json.dumps(resultado, ensure_ascii=False, indent=2)
    if args.json:
        _gravar_texto(args.json, texto)
    if args.saida:
        try:
            ok, buf = cv2.imencode(args.saida.suffix or ".jpg", anotada)
        except cv2.error:  # extensão sem codificador, ex.: .txt
            ok = False
        if not ok:
            print(f"Não consegui salvar {args.saida}: use uma extensão de imagem, como .jpg ou .png", file=sys.stderr)
            return 1
        args.saida.parent.mkdir(parents=True, exist_ok=True)
        buf.tofile(str(args.saida))
    print(texto)
    return 0


def cmd_video(args) -> int:
    from .video import processar_video

    origem = args.webcam if args.webcam is not None else str(args.caminho)
    resumo = processar_video(origem, _analisador(args), saida=Path(args.saida) if args.saida else None,
                             mostrar=args.mostrar, max_quadros=args.max_quadros,
                             periodo=None if args.periodo == "auto" else args.periodo, inicio_s=args.inicio)
    texto = json.dumps(resumo, ensure_ascii=False, indent=2)
    if args.resumo:
        _gravar_texto(args.resumo, texto)
    print(texto)
    return 0


def cmd_api(args) -> int:
    import uvicorn

    from . import api

    if args.sem_privacidade:
        print("--sem-privacidade não vale para a API: o filtro de privacidade é obrigatório.", file=sys.stderr)
        return 2
    # Criado aqui, o analisador respeita --pesos e --imgsz e falha logo se faltarem os pesos;
    # o ciclo de vida da API não cria outro quando já existe um.
    api.app.state.analisador = _analisador(args)
    uvicorn.run(api.app, host=args.host, port=args.porta)
    return 0


def cmd_app(args) -> int:
    from .app import TAMANHO_MAXIMO_ENVIO, criar_interface

    if args.sem_privacidade:
        # A interface promete desfocar as pessoas e os veículos detectados antes de exibir o resultado.
        print("--sem-privacidade não vale para o app: o filtro de privacidade é obrigatório.", file=sys.stderr)
        return 2
    # run_history=False: o Gradio não guarda no navegador o histórico das análises (local e horário).
    criar_interface(_analisador(args)).launch(server_name=args.host, server_port=args.porta,
                                              max_file_size=TAMANHO_MAXIMO_ENVIO, run_history=False)
    return 0


def _opcoes_comuns(parser: argparse.ArgumentParser, com_padrao: bool) -> None:
    """--pesos, --imgsz e --sem-privacidade. Nos subcomandos ficam sem padrão (SUPPRESS): assim só
    sobrescrevem o valor dado antes do subcomando quando aparecem depois dele."""
    sem = argparse.SUPPRESS
    parser.add_argument("--pesos", default=None if com_padrao else sem,
                        help="pesos do detector (.pt ou .onnx; os resultados publicados são do .pt, e o .onnx tem "
                             "entrada fixa de 640×640); padrão models/luzsegura_yolo26n.pt ou LUZSEGURA_PESOS")
    parser.add_argument("--imgsz", type=int, default=640 if com_padrao else sem,
                        help="lado da entrada do detector, em px (padrão 640)")
    parser.add_argument("--sem-privacidade", action="store_true", default=False if com_padrao else sem,
                        help="não desfocar pessoas e veículos (só para testes de foto e vídeo; a API e o app recusam)")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="luzsegura", description="Luz Segura: visão computacional para iluminação pública")
    _opcoes_comuns(ap, com_padrao=True)
    sub = ap.add_subparsers(dest="comando", required=True)

    f = sub.add_parser("foto", help="analisa uma foto")
    f.add_argument("caminho", type=Path, help="foto JPEG, PNG ou WebP")
    f.add_argument("--saida", type=Path, help="grava a imagem anotada (já anonimizada) neste arquivo")
    f.add_argument("--json", type=Path, help="grava o resultado (contrato JSON v1) neste arquivo")
    f.add_argument("--lat", type=float, help="latitude do local da foto (com --lon e --quando, define o período pelo Sol)")
    f.add_argument("--lon", type=float, help="longitude do local da foto")
    f.add_argument("--quando", type=_data_hora, help="data e hora ISO 8601 com fuso, ex.: 2026-11-18T21:30:00-03:00")
    f.add_argument("--periodo", choices=("auto", "noite", "dia"), default="auto",
                   help="auto: pelo Sol (com horário e local) ou pelo brilho da foto (padrão auto)")
    f.set_defaults(func=cmd_foto)

    v = sub.add_parser("video", help="analisa um vídeo ou a imagem ao vivo da webcam")
    v.add_argument("caminho", type=Path, nargs="?", help="arquivo de vídeo (ou use --webcam)")
    v.add_argument("--webcam", type=int, help="índice da câmera, ex.: 0")
    v.add_argument("--saida", type=Path, help="grava o vídeo anotado e anonimizado (MP4)")
    v.add_argument("--resumo", type=Path, help="grava o resumo do trajeto em JSON")
    v.add_argument("--mostrar", action="store_true", help="mostra o vídeo anotado numa janela (q encerra)")
    v.add_argument("--max-quadros", type=int, help="para depois deste número de quadros")
    v.add_argument("--inicio", type=float, default=0.0, help="segundo do vídeo em que a análise começa (ex.: pular uma abertura)")
    v.add_argument("--periodo", choices=("auto", "noite", "dia"), default="noite",
                   help="período do vídeo (padrão noite; auto estima pelo brilho de cada quadro)")
    v.set_defaults(func=cmd_video)

    a = sub.add_parser("api", help="sobe a API HTTP (FastAPI)")
    a.add_argument("--host", default="127.0.0.1", help="endereço de escuta (padrão 127.0.0.1)")
    a.add_argument("--porta", "--port", dest="porta", type=int, default=8000, help="porta (padrão 8000)")
    a.set_defaults(func=cmd_api)

    g = sub.add_parser("app", help="sobe a interface web de demonstração (Gradio)")
    g.add_argument("--host", default="127.0.0.1", help="endereço de escuta (padrão 127.0.0.1)")
    g.add_argument("--porta", type=int, default=7860, help="porta (padrão 7860)")
    g.set_defaults(func=cmd_app)

    for subcomando in (f, v, a, g):
        _opcoes_comuns(subcomando, com_padrao=False)

    args = ap.parse_args(argv)
    if args.comando == "video" and args.caminho is None and args.webcam is None:
        ap.error("informe o caminho do vídeo ou --webcam N")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
