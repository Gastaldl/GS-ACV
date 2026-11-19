import base64
import struct
import threading
import zlib

import cv2
import numpy as np
import pytest
import requests
from fastapi.testclient import TestClient
from starlette.datastructures import UploadFile

from luzsegura import api
from luzsegura.config import LADO_MAXIMO_ANALISE
from luzsegura.diagnostico import diagnosticar
from luzsegura.luminancia import luma, medir_cena

URL = "/v1/analisar"


class AnalisadorFalso:
    """Substitui o modelo: devolve o diagnóstico de uma cena sem luminárias."""

    def __init__(self):
        self.ultima_imagem_shape = None

    def analisar(self, img, ctx=None, desenhar_saida=True):
        self.ultima_imagem_shape = img.shape
        r = diagnosticar([], medir_cena(luma(img)), img.shape[1], img.shape[0], ctx, tempo_ms=1.0)
        return r, (img.copy() if desenhar_saida else None)


@pytest.fixture()
def cliente():
    api.app.state.analisador = AnalisadorFalso()
    with TestClient(api.app) as c:
        yield c
    api.app.state.analisador = None


def codificar(ext, h=120, w=160, valor=10):
    ok, buf = cv2.imencode(ext, np.full((h, w, 3), valor, np.uint8))
    assert ok
    return buf.tobytes()


def jpeg(h=120, w=160, valor=10):
    return codificar(".jpg", h, w, valor)


def png_1bit(largura, altura):
    """PNG preto de 1 bit por pixel, válido: poucos KB que viram largura x altura x 3 bytes ao decodificar."""

    def bloco(tipo, dados):
        return struct.pack(">I", len(dados)) + tipo + dados + struct.pack(">I", zlib.crc32(tipo + dados))

    linha = bytes(1 + (largura + 7) // 8)  # byte de filtro + pixels zerados
    z = zlib.compressobj(9)
    idat = b"".join(z.compress(linha) for _ in range(altura)) + z.flush()
    ihdr = struct.pack(">IIBBBBB", largura, altura, 1, 0, 0, 0, 0)  # 1 bit, tons de cinza
    return b"\x89PNG\r\n\x1a\n" + bloco(b"IHDR", ihdr) + bloco(b"IDAT", idat) + bloco(b"IEND", b"")


@pytest.fixture()
def leituras(monkeypatch):
    """Registra cada leitura do upload pelo endpoint: fica vazia quando o corpo é recusado antes do formulário."""
    chamadas = []
    original = UploadFile.read

    async def ler(self, *args, **kwargs):
        chamadas.append(self.filename)
        return await original(self, *args, **kwargs)

    monkeypatch.setattr(UploadFile, "read", ler)
    return chamadas


def nao_analisou():
    return api.app.state.analisador.ultima_imagem_shape is None


def test_saude(cliente):
    r = cliente.get("/v1/saude")
    assert r.status_code == 200
    assert r.json()["versao_contrato"] == "1.0"


def test_analisar_devolve_contrato(cliente):
    r = cliente.post(URL, files={"arquivo": ("noite.jpg", jpeg(), "image/jpeg")},
                     data={"lat": "-23.53", "lon": "-46.79", "capturado_em": "2026-11-18T22:00:00-03:00"})
    assert r.status_code == 200, r.text
    corpo = r.json()
    assert corpo["periodo_fonte"] == "posicao_do_sol"
    assert corpo["diagnostico"] == "trecho_escuro"
    assert corpo["privacidade"]["imagem_armazenada"] is False
    assert corpo["imagem_anotada_jpeg_base64"] is None


def test_imagem_anotada_opcional(cliente):
    r = cliente.post(URL, files={"arquivo": ("n.jpg", jpeg(), "image/jpeg")}, data={"incluir_imagem": "true"})
    assert r.status_code == 200
    dados = base64.b64decode(r.json()["imagem_anotada_jpeg_base64"])
    assert dados[:2] == b"\xff\xd8"


@pytest.mark.parametrize("capturado_em, status", [
    ("2026-11-18T19:30:00", 422),  # sem fuso: seria lido como UTC, com o Sol ainda alto em São Paulo
    ("2026-11-18", 422),
    ("2026-11-18T19:30:00-03:00", 200),
    ("2026-11-18T22:30:00Z", 200),
])
def test_capturado_em_exige_fuso(cliente, capturado_em, status):
    r = cliente.post(URL, files={"arquivo": ("n.jpg", jpeg(), "image/jpeg")},
                     data={"lat": "-23.55", "lon": "-46.63", "capturado_em": capturado_em})
    assert r.status_code == status, r.text
    if status == 422:
        assert nao_analisou()
    else:
        assert r.json()["periodo"] == "noite"


def test_tipo_nao_suportado(cliente):
    r = cliente.post(URL, files={"arquivo": ("x.gif", b"GIF89a", "image/gif")})
    assert r.status_code == 415


def test_arquivo_corrompido(cliente):
    r = cliente.post(URL, files={"arquivo": ("x.jpg", b"nao e imagem", "image/jpeg")})
    assert r.status_code == 422


def test_upload_sem_tipo_na_parte_como_o_requests_envia(cliente):
    preparado = requests.Request("POST", "http://teste" + URL, files={"arquivo": ("foto.jpg", jpeg())}).prepare()
    assert b"Content-Type" not in preparado.body  # a parte do arquivo vai sem tipo
    r = cliente.post(URL, content=preparado.body, headers={"Content-Type": preparado.headers["Content-Type"]})
    assert r.status_code == 200, r.text


@pytest.mark.parametrize("nome, ext, tipo", [
    ("n.jpg", ".jpg", "application/octet-stream"),
    ("n.jpg", ".jpg", "image/jpg"),
    ("n.webp", ".webp", "application/octet-stream"),
    ("n.png", ".png", "image/png"),
])
def test_tipo_declarado_e_so_indicativo(cliente, nome, ext, tipo):
    r = cliente.post(URL, files={"arquivo": (nome, codificar(ext), tipo)})
    assert r.status_code == 200, r.text


@pytest.mark.parametrize("conteudo", [codificar(".tiff"), codificar(".bmp"), codificar(".ppm"), b"GIF89a" + bytes(64)])
def test_bytes_fora_de_jpeg_png_webp_sao_recusados(cliente, conteudo):
    r = cliente.post(URL, files={"arquivo": ("x.jpg", conteudo, "image/jpeg")})
    assert r.status_code == 422
    assert nao_analisou()


def test_imagem_grande_e_reduzida(cliente):
    r = cliente.post(URL, files={"arquivo": ("g.jpg", jpeg(2400, 3200), "image/jpeg")})
    assert r.status_code == 200
    assert max(api.app.state.analisador.ultima_imagem_shape[:2]) == LADO_MAXIMO_ANALISE


def test_imagem_estreita_nao_quebra_a_reducao(cliente):
    r = cliente.post(URL, files={"arquivo": ("f.png", codificar(".png", 1, 4000), "image/png")})
    assert r.status_code == 200, r.text
    assert api.app.state.analisador.ultima_imagem_shape == (1, LADO_MAXIMO_ANALISE, 3)


def test_erro_do_opencv_vira_422(cliente, monkeypatch):
    def falha(*_):
        raise cv2.error("limite de pixels do OpenCV")

    monkeypatch.setattr(api.cv2, "imdecode", falha)
    r = cliente.post(URL, files={"arquivo": ("n.jpg", jpeg(), "image/jpeg")})
    assert r.status_code == 422


@pytest.mark.parametrize("lado", [9_000, 20_000])  # acima do orçamento da API; acima do limite do próprio Pillow
def test_bomba_de_descompressao_recusada_sem_decodificar(cliente, monkeypatch, lado):
    bomba = png_1bit(lado, lado)
    assert len(bomba) < 100_000  # dezenas de KB que virariam centenas de MB ou GB na memória
    monkeypatch.setattr(api.cv2, "imdecode", lambda *_: pytest.fail("decodificou a imagem inteira"))
    r = cliente.post(URL, files={"arquivo": ("b.png", bomba, "image/png")})
    assert r.status_code == 413
    assert nao_analisou()


def test_foto_pouco_acima_de_10_mb(cliente, leituras):
    # Cabe na folga do middleware para o multipart e cai na conferência do próprio endpoint.
    r = cliente.post(URL, files={"arquivo": ("g.jpg", b"\xff\xd8\xff" + bytes(api.TAMANHO_MAXIMO), "image/jpeg")})
    assert r.status_code == 413
    assert leituras == ["g.jpg"] and nao_analisou()


def test_corpo_grande_recusado_antes_do_formulario(cliente, leituras):
    r = cliente.post(URL, files={"arquivo": ("g.jpg", bytes(12 * 2**20), "image/jpeg")})
    assert r.status_code == 413
    assert int(r.request.headers["content-length"]) > api.LIMITE_CORPO
    assert leituras == [] and nao_analisou()


def test_corpo_sem_content_length_tambem_e_limitado(cliente, leituras):
    fronteira = "fronteira"
    inicio = (f"--{fronteira}\r\nContent-Disposition: form-data; name=\"arquivo\"; filename=\"g.jpg\"\r\n"
              "Content-Type: image/jpeg\r\n\r\n").encode()

    def corpo():  # gerador: o cliente manda o corpo em partes (chunked), sem Content-Length
        yield inicio
        for _ in range(12):
            yield bytes(2**20)
        yield f"\r\n--{fronteira}--\r\n".encode()

    r = cliente.post(URL, content=corpo(), headers={"Content-Type": f"multipart/form-data; boundary={fronteira}"})
    assert "content-length" not in r.request.headers
    assert r.status_code == 413
    assert leituras == [] and nao_analisou()


def test_foto_acima_de_1_mb_nao_vai_para_o_disco(cliente, monkeypatch):
    from starlette import formparsers

    despejos = []

    class Espiao(formparsers.SpooledTemporaryFile):
        def rollover(self):
            despejos.append(self)
            super().rollover()

    monkeypatch.setattr(formparsers, "SpooledTemporaryFile", Espiao)
    ruido = np.random.default_rng(0).integers(0, 256, (1000, 1000, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".png", ruido)
    assert ok and len(buf) > 2 * 2**20  # acima do 1 MB a partir do qual o Starlette gravaria em disco
    r = cliente.post(URL, files={"arquivo": ("r.png", buf.tobytes(), "image/png")})
    assert r.status_code == 200, r.text
    assert not despejos


def test_saude_responde_durante_a_analise(cliente):
    iniciou, liberar = threading.Event(), threading.Event()

    class Lento(AnalisadorFalso):
        liberado = None

        def analisar(self, img, ctx=None, desenhar_saida=True):
            iniciou.set()
            # Se a análise rodasse no laço de eventos, /v1/saude só responderia depois desta espera.
            self.liberado = liberar.wait(5)
            return super().analisar(img, ctx, desenhar_saida)

    lento = Lento()
    api.app.state.analisador = lento
    respostas = []
    t = threading.Thread(target=lambda: respostas.append(
        cliente.post(URL, files={"arquivo": ("n.jpg", jpeg(), "image/jpeg")})))
    t.start()
    assert iniciou.wait(5)
    r = cliente.get("/v1/saude")
    liberar.set()
    t.join(10)
    assert r.status_code == 200
    assert lento.liberado is True
    assert respostas[0].status_code == 200


def test_coordenada_invalida(cliente):
    r = cliente.post(URL, files={"arquivo": ("n.jpg", jpeg(), "image/jpeg")}, data={"lat": "123"})
    assert r.status_code == 422
