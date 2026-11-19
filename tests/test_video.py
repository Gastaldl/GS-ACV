"""Resumo do trajeto em vídeo: voto ponderado por luminária rastreada e revisão humana.

O detector é falso (devolve um roteiro de detecções por quadro) e o vídeo é sintético, gravado
com o OpenCV numa pasta temporária: o teste não depende do modelo treinado.
"""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from luzsegura.config import CLASSE_ACESA, CLASSE_APAGADA, LIMIARES_PADRAO
from luzsegura.diagnostico import Luminaria
from luzsegura.luminancia import ACESA, APAGADA, INDEFINIDO, MedidaCaixa
from luzsegura.video import processar_video


def medida(estado):
    acesa = estado == ACESA
    return MedidaCaixa(media=40, pico=250 if acesa else 90, pixels_nucleo=80 if acesa else 0,
                       fracao_nucleo=0.02 if acesa else 0, maior_mancha=80 if acesa else 0,
                       contraste=2.0 if acesa else 0.9, estado=estado)


def quadros(n, classe, conf, estado, tid=1):
    """n quadros seguidos com a mesma luminária rastreada."""
    return [[Luminaria(classe, conf, (20.0, 4.0, 30.0, 24.0), medida(estado), tid)] for _ in range(n)]


class DetectorFalso:
    """Devolve, a cada quadro, a próxima lista de luminárias do roteiro."""

    def __init__(self, roteiro):
        self.roteiro = roteiro
        self.n = 0
        self.reinicios = 0

    def reiniciar_rastreio(self):
        self.reinicios += 1

    def detectar(self, quadro, rastrear=False):
        assert rastrear, "no vídeo o rastreador mantém o ID de cada luminária entre quadros"
        saida = self.roteiro[self.n] if self.n < len(self.roteiro) else []
        self.n += 1
        return saida


@pytest.fixture()
def resumir(tmp_path):
    """Grava um vídeo escuro com um quadro por item do roteiro e devolve o resumo do trajeto."""

    def _resumir(roteiro, periodo="noite"):
        caminho = tmp_path / "trajeto.avi"
        escritor = cv2.VideoWriter(str(caminho), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
        assert escritor.isOpened()
        for _ in roteiro:
            escritor.write(np.full((48, 64, 3), 10, np.uint8))
        escritor.release()
        detector = DetectorFalso(roteiro)
        analisador = SimpleNamespace(detector=detector, lim=LIMIARES_PADRAO, privacidade=None)
        resumo = processar_video(str(caminho), analisador, periodo=periodo)
        assert resumo["quadros_processados"] == len(roteiro)
        assert detector.reinicios == 1
        return resumo

    return _resumir


def test_inicio_pula_a_abertura_e_os_tempos_contam_dali(tmp_path):
    caminho = tmp_path / "abertura.avi"
    escritor = cv2.VideoWriter(str(caminho), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (64, 48))
    for _ in range(20):
        escritor.write(np.full((48, 64, 3), 10, np.uint8))
    escritor.release()
    roteiro = quadros(15, CLASSE_APAGADA, 0.8, APAGADA)
    analisador = SimpleNamespace(detector=DetectorFalso(roteiro), lim=LIMIARES_PADRAO, privacidade=None)
    resumo = processar_video(str(caminho), analisador, periodo="noite", inicio_s=0.5)
    assert resumo["inicio_s"] == 0.5 and resumo["quadros_processados"] == 15
    assert resumo["luminarias_apagadas"][0]["aparece_em_s"] == 0.0


def test_voto_ponderado_pela_confianca(resumir):
    # Longe, 3 quadros "apagada" a 0,35; perto, 3 quadros "acesa" a 0,92: a contagem empata e a confiança decide.
    resumo = resumir(quadros(3, CLASSE_APAGADA, 0.35, APAGADA) + quadros(3, CLASSE_ACESA, 0.92, ACESA))
    assert resumo["luminarias_unicas"] == {CLASSE_ACESA: 1, CLASSE_APAGADA: 0}
    assert resumo["luminarias_apagadas"] == []
    assert resumo["requer_revisao_humana"] is False


def test_empate_exato_fica_acesa(resumir):
    resumo = resumir(quadros(3, CLASSE_APAGADA, 0.6, APAGADA) + quadros(3, CLASSE_ACESA, 0.6, ACESA))
    assert resumo["luminarias_unicas"] == {CLASSE_ACESA: 1, CLASSE_APAGADA: 0}


def test_trilha_curta_e_ignorada(resumir):
    resumo = resumir(quadros(2, CLASSE_APAGADA, 0.9, APAGADA))
    assert resumo["luminarias_unicas"] == {CLASSE_ACESA: 0, CLASSE_APAGADA: 0}


def test_apagada_confirmada_a_noite_dispensa_revisao(resumir):
    resumo = resumir(quadros(6, CLASSE_APAGADA, 0.8, APAGADA))
    assert resumo["luminarias_apagadas"] == [{
        "id_rastreio": 1, "aparece_em_s": 0.0, "confianca_max": 0.8, "quadros": 6, "quadros_apagada": 6,
        "quadros_por_brilho": {"acesa": 0, "apagada": 6}, "requer_revisao_humana": False, "motivos_revisao": [],
    }]
    assert resumo["requer_revisao_humana"] is False


def test_apagada_contradita_pelo_brilho_pede_revisao(resumir):
    resumo = resumir(quadros(6, CLASSE_APAGADA, 0.8, ACESA))
    [item] = resumo["luminarias_apagadas"]
    assert item["quadros_por_brilho"] == {"acesa": 6, "apagada": 0}
    assert item["requer_revisao_humana"] is True
    assert item["motivos_revisao"] == ["brilho_contradiz_modelo"]
    assert resumo["requer_revisao_humana"] is True


@pytest.mark.parametrize("roteiro, periodo, motivo", [
    pytest.param(quadros(6, CLASSE_APAGADA, 0.8, INDEFINIDO), "noite", "brilho_nao_confirma", id="brilho_indefinido"),
    pytest.param(quadros(6, CLASSE_APAGADA, 0.45, APAGADA), "noite", "confianca_baixa", id="confianca_baixa"),
    pytest.param(quadros(4, CLASSE_APAGADA, 0.8, APAGADA) + quadros(2, CLASSE_ACESA, 0.9, APAGADA), "noite",
                 "quadros_divididos", id="quadros_divididos"),
    pytest.param(quadros(6, CLASSE_APAGADA, 0.8, APAGADA), None, "periodo_estimado_pelo_brilho", id="sem_periodo"),
    pytest.param(quadros(6, CLASSE_APAGADA, 0.8, APAGADA), "transicao", "periodo_transicao", id="transicao"),
])
def test_motivos_de_revisao_por_luminaria(resumir, roteiro, periodo, motivo):
    resumo = resumir(roteiro, periodo)
    [item] = resumo["luminarias_apagadas"]
    assert item["motivos_revisao"] == [motivo]
    assert item["requer_revisao_humana"] is True
    assert resumo["requer_revisao_humana"] is True


def test_confianca_max_e_quadros_contam_so_a_classe_apagada(resumir):
    resumo = resumir(quadros(4, CLASSE_APAGADA, 0.8, APAGADA) + quadros(2, CLASSE_ACESA, 0.9, APAGADA))
    [item] = resumo["luminarias_apagadas"]
    assert item["confianca_max"] == 0.8
    assert (item["quadros"], item["quadros_apagada"]) == (6, 4)


def test_revisao_no_resumo_vale_se_alguma_luminaria_pede(resumir):
    confirmada = quadros(6, CLASSE_APAGADA, 0.8, APAGADA, tid=7)
    contradita = quadros(6, CLASSE_APAGADA, 0.8, ACESA, tid=3)
    resumo = resumir([a + b for a, b in zip(confirmada, contradita)])
    assert {i["id_rastreio"]: i["requer_revisao_humana"] for i in resumo["luminarias_apagadas"]} == {7: False, 3: True}
    assert resumo["requer_revisao_humana"] is True
