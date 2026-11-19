import numpy as np

from luzsegura.config import Limiares
from luzsegura.luminancia import ACESA, APAGADA, INDEFINIDO, classificar_estado, medir_caixa, medir_cena


def test_cena_escura_tem_piso_escuro():
    y = np.full((100, 200), 12, np.uint8)
    y[:40] = 60  # céu um pouco mais claro que o chão
    cena = medir_cena(y)
    assert cena.piso < 15
    assert cena.fracao_escura > 0.5
    assert cena.fracao_saturada == 0


def test_luminaria_acesa_tem_ponto_de_luz():
    y = np.full((200, 200), 20, np.uint8)
    y[50:60, 90:100] = 255  # núcleo saturado de 100 px
    m = medir_caixa(y, (80, 40, 110, 120))
    assert m.maior_mancha == 100
    assert m.estado == ACESA


def test_luminaria_apagada_contra_ceu_claro():
    y = np.full((200, 200), 170, np.uint8)  # céu ainda claro ao anoitecer
    y[40:120, 80:110] = 60  # silhueta escura da luminária
    m = medir_caixa(y, (80, 40, 110, 120))
    assert m.maior_mancha == 0
    assert m.contraste < 1
    assert m.estado == APAGADA


def test_brilho_difuso_sem_nucleo_e_indefinido():
    y = np.full((200, 200), 30, np.uint8)
    y[40:120, 80:110] = 150  # mais clara que o entorno, mas sem ponto saturado
    assert medir_caixa(y, (80, 40, 110, 120)).estado == INDEFINIDO


def test_pontos_isolados_nao_contam_como_luminaria_acesa():
    y = np.full((200, 200), 20, np.uint8)
    y[45, 85] = y[70, 95] = y[100, 105] = 255  # pixels soltos (ruído, reflexo)
    m = medir_caixa(y, (80, 40, 110, 120))
    assert m.pixels_nucleo == 3
    assert m.maior_mancha == 1
    assert m.estado != ACESA


def test_caixa_fora_da_imagem_e_recortada():
    y = np.zeros((50, 50), np.uint8)
    m = medir_caixa(y, (-20, -20, 10, 10))
    assert m.estado in (APAGADA, INDEFINIDO)


def test_limiar_configuravel():
    lim = Limiares(nucleo_min_pixels=200)
    assert classificar_estado(100, 2.0, lim) == INDEFINIDO
    assert classificar_estado(250, 2.0, lim) == ACESA
