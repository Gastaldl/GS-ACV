"""API HTTP do componente de visão (contrato JSON v1), consumida pelo MVP do Luz Segura.

Rodar:  uvicorn luzsegura.api:app --host 0.0.0.0 --port 8000
Docs interativas: http://localhost:8000/docs

Privacidade: nada é gravado em disco. O corpo da requisição é limitado a cerca de 10 MB antes
de ser lido, então a foto fica inteira na memória (nunca num arquivo temporário), é
decodificada, analisada e descartada; o conteúdo da imagem não aparece nos logs. Se o cliente
pedir a imagem anotada, ela volta com as pessoas e os veículos detectados pelo filtro já
desfocados; à noite ou em ângulos incomuns, o filtro pode deixar passar algum.
"""

from __future__ import annotations

import base64
import io
from contextlib import asynccontextmanager
from typing import Any, Literal

import anyio
import anyio.to_thread
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import AwareDatetime, BaseModel
from starlette.formparsers import MultiPartParser

from . import __version__
from .config import NOME_MODELO, VERSAO_CONTRATO
from .diagnostico import Contexto

TAMANHO_MAXIMO = 10 * 1024 * 1024  # 10 MB
LIMITE_CORPO = TAMANHO_MAXIMO + 64 * 1024  # a foto mais os demais campos e cabeçalhos do multipart
# ~80 MP cobre as fotos de 50 e 64 MP dos celulares e fica abaixo do alerta de bomba de descompressão
# do Pillow (~89 MP). Decodificada em BGR, uma foto no limite ocupa ~240 MB, uma análise por vez.
PIXELS_MAXIMOS = 80_000_000
TIPOS_ACEITOS = {"image/jpeg", "image/png", "image/webp"}
# Quem decide o formato são os bytes; o tipo declarado na parte é só indicativo. O requests não
# manda tipo nenhum, e o curl manda application/octet-stream para extensões que não conhece.
TIPOS_GENERICOS = {"", "application/octet-stream", "image/jpg", "image/pjpeg"}
FORMATOS_PIL = ("JPEG", "PNG", "WEBP")
INVALIDA = "Arquivo não é uma imagem JPEG, PNG ou WebP válida."
GRANDE = "Imagem maior que 10 MB."
RESOLUCAO = f"Resolução acima de {PIXELS_MAXIMOS // 1_000_000} megapixels."

# Com o corpo limitado por LimitarCorpo, a foto cabe inteira na memória: o Starlette deixa de
# despejá-la num arquivo temporário (o padrão dele é a partir de 1 MB).
MultiPartParser.spool_max_size = LIMITE_CORPO


class ContagemModelo(BaseModel):
    luminaria_acesa: int
    luminaria_apagada: int


class LuminosidadeModelo(BaseModel):
    cena_media: float
    piso_media: float
    fracao_escura: float
    fracao_saturada: float


class LuminariaModelo(BaseModel):
    classe: Literal["luminaria_acesa", "luminaria_apagada"]
    confianca: float
    caixa_norm: list[float]
    estado_por_brilho: Literal["acesa", "apagada", "indefinido"]
    concorda_com_modelo: bool | None
    maior_ponto_de_luz_px: int
    contraste_com_entorno: float
    id_rastreio: int | None


class PrivacidadeModelo(BaseModel):
    pessoas_desfocadas: int
    veiculos_desfocados: int
    imagem_armazenada: bool
    filtro_ativo: bool = True


class ResultadoAnalise(BaseModel):
    """Contrato JSON v1 do componente de visão."""

    versao_contrato: str
    modelo: str
    origem: str
    periodo: Literal["noite", "transicao", "dia"]
    periodo_fonte: Literal["informado", "posicao_do_sol", "luminosidade"]
    altura_do_sol_graus: float | None
    diagnostico: Literal["iluminacao_adequada", "luminaria_apagada", "iluminacao_insuficiente", "trecho_escuro",
                         "fora_do_periodo_noturno", "acesa_durante_o_dia"]
    severidade: Literal["nenhuma", "baixa", "media", "alta"]
    requer_revisao_humana: bool
    acao_sugerida: str
    explicacao: str
    contagem: ContagemModelo
    luminosidade: LuminosidadeModelo
    luminarias: list[LuminariaModelo]
    privacidade: PrivacidadeModelo
    imagem: dict[str, int]
    tempo_processamento_ms: float | None
    localizacao: dict[str, float] | None
    capturado_em: str | None
    imagem_anotada_jpeg_base64: str | None = None


def _criar_analisador():
    from .pipeline import criar_analisador

    return criar_analisador(privacidade=True)


@asynccontextmanager
async def ciclo_de_vida(app: FastAPI):
    if getattr(app.state, "analisador", None) is None:  # testes e a linha de comando injetam o analisador
        app.state.analisador = _criar_analisador()
    # Uma análise por vez: memória e CPU ficam limitadas, e as requisições na fila esperam aqui,
    # sem ocupar o pool de threads que atende /v1/saude.
    app.state.limite_modelo = anyio.CapacityLimiter(1)
    yield


class LimitarCorpo:
    """Middleware ASGI: recusa com 413 o corpo acima do limite antes de o FastAPI ler o formulário."""

    def __init__(self, app, limite: int = LIMITE_CORPO):
        self.app, self.limite = app, limite

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declarado = dict(scope["headers"]).get(b"content-length", b"")
        if declarado.isdigit() and int(declarado) > self.limite:
            await JSONResponse({"detail": GRANDE}, status_code=413)(scope, receive, send)
            return
        recebido = 0

        async def receber():
            nonlocal recebido
            mensagem = await receive()
            if mensagem["type"] == "http.request":
                recebido += len(mensagem.get("body", b""))
                if recebido > self.limite:  # corpo sem Content-Length (chunked): conta os bytes que chegam
                    raise HTTPException(status_code=413, detail=GRANDE)
            return mensagem

        await self.app(scope, receber, send)


app = FastAPI(title="Luz Segura · Visão Computacional", version=__version__, lifespan=ciclo_de_vida,
              description="Detecta luminárias públicas acesas e apagadas e trechos escuros em fotos noturnas.")
app.add_middleware(LimitarCorpo)


@app.get("/v1/saude", summary="Saúde")
def saude() -> dict[str, Any]:
    return {"status": "ok", "modelo": NOME_MODELO, "versao_contrato": VERSAO_CONTRATO, "versao": __version__}


def _formato_real(conteudo: bytes) -> str | None:
    """Formato pela assinatura dos primeiros bytes: só JPEG, PNG e WebP chegam aos decodificadores."""
    if conteudo[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if conteudo[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if conteudo[:4] == b"RIFF" and conteudo[8:12] == b"WEBP":
        return "image/webp"
    return None


def _decodificar(conteudo: bytes) -> np.ndarray:
    """Decodifica a foto e a reduz ao lado maior de análise (config.LADO_MAXIMO_ANALISE)."""
    from .pipeline import reduzir  # importado aqui: o contrato (modelos acima) não depende do torch

    # O Pillow lê só o cabeçalho (abre sem decodificar): a resolução é conferida antes de alocar a
    # imagem inteira, o que barra bombas de descompressão (PNG de poucos KB que vira GB na memória).
    try:
        with Image.open(io.BytesIO(conteudo), formats=FORMATOS_PIL) as cabecalho:
            largura, altura = cabecalho.size
    except Image.DecompressionBombError:  # acima de ~179 MP o próprio Pillow recusa
        raise HTTPException(status_code=413, detail=RESOLUCAO) from None
    except Exception:  # cabeçalho inválido ou truncado
        raise HTTPException(status_code=422, detail=INVALIDA) from None
    if largura * altura > PIXELS_MAXIMOS:
        raise HTTPException(status_code=413, detail=RESOLUCAO)
    try:
        img = cv2.imdecode(np.frombuffer(conteudo, np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:  # ex.: o limite de pixels do próprio OpenCV; sem isto, viraria HTTP 500
        img = None
    if img is None:
        raise HTTPException(status_code=422, detail=INVALIDA)
    # Reduzir já aqui libera a imagem em resolução cheia antes da inferência.
    return reduzir(img)


def _processar(analisador, conteudo: bytes, ctx: Contexto, incluir_imagem: bool) -> dict:
    """Trabalho de CPU (decodificar, os dois YOLO, desenhar e codificar), feito fora do laço de eventos."""
    img = _decodificar(conteudo)
    resultado, anotada = analisador.analisar(img, ctx, desenhar_saida=incluir_imagem)
    if incluir_imagem and anotada is not None:
        ok, buf = cv2.imencode(".jpg", anotada, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if ok:
            resultado["imagem_anotada_jpeg_base64"] = base64.b64encode(buf.tobytes()).decode("ascii")
    return resultado


@app.post("/v1/analisar", response_model=ResultadoAnalise, summary="Analisar")
async def analisar(request: Request,
                   arquivo: UploadFile = File(..., description="Foto JPEG, PNG ou WebP (até 10 MB e 80 megapixels)"),
                   lat: float | None = Form(None, ge=-90, le=90),
                   lon: float | None = Form(None, ge=-180, le=180),
                   capturado_em: AwareDatetime | None = Form(
                       None, description="ISO 8601 com fuso (obrigatório), ex.: 2026-11-18T21:30:00-03:00"),
                   periodo: Literal["auto", "noite", "dia"] = Form("auto"),
                   incluir_imagem: bool = Form(False)) -> dict:
    declarado = (arquivo.content_type or "").split(";")[0].strip().lower()
    if declarado not in TIPOS_ACEITOS | TIPOS_GENERICOS:
        raise HTTPException(status_code=415, detail=f"Tipo não suportado: {declarado}")
    conteudo = await arquivo.read(TAMANHO_MAXIMO + 1)
    if len(conteudo) > TAMANHO_MAXIMO:
        raise HTTPException(status_code=413, detail=GRANDE)
    if _formato_real(conteudo) is None:  # TIFF, BMP, GIF e afins nunca chegam ao cv2.imdecode
        raise HTTPException(status_code=422, detail=INVALIDA)
    ctx = Contexto(origem="foto", capturado_em=capturado_em, lat=lat, lon=lon,
                   periodo_informado=None if periodo == "auto" else periodo)
    # Decodificar e rodar os modelos aqui travaria o laço de eventos (nem /v1/saude responderia).
    return await anyio.to_thread.run_sync(_processar, request.app.state.analisador, conteudo, ctx,
                                          incluir_imagem, limiter=request.app.state.limite_modelo)
