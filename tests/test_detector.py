"""Remoção de caixas repetidas da mesma luminária (o YOLO26 não faz NMS entre classes)."""

from luzsegura.config import CLASSE_ACESA, CLASSE_APAGADA, LIMIARES_PADRAO
from luzsegura.detector import remover_duplicatas
from luzsegura.diagnostico import Luminaria
from luzsegura.luminancia import ACESA, MedidaCaixa

MEDIDA = MedidaCaixa(media=200, pico=255, pixels_nucleo=80, fracao_nucleo=0.05, maior_mancha=80, contraste=2.0, estado=ACESA)


def lum(classe, conf, xyxy):
    return Luminaria(classe, conf, xyxy, MEDIDA)


def test_mesma_luminaria_com_duas_classes_fica_so_a_mais_confiante():
    caixas = [lum(CLASSE_ACESA, 0.38, (100, 50, 200, 400)), lum(CLASSE_APAGADA, 0.46, (102, 52, 201, 398))]
    saida = remover_duplicatas(caixas, LIMIARES_PADRAO.iou_mesma_luminaria)
    assert [(l.classe, l.confianca) for l in saida] == [(CLASSE_APAGADA, 0.46)]


def test_luminarias_vizinhas_ficam_todas():
    caixas = [lum(CLASSE_ACESA, 0.8, (0, 0, 100, 100)), lum(CLASSE_ACESA, 0.7, (60, 0, 160, 100)),
              lum(CLASSE_APAGADA, 0.6, (300, 0, 400, 100))]
    assert len(remover_duplicatas(caixas, LIMIARES_PADRAO.iou_mesma_luminaria)) == 3
