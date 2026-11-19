"""Configuração central do componente de visão do Luz Segura."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PACOTE = Path(__file__).resolve().parent
RAIZ = PACOTE.parents[1]
DIR_MODELOS = RAIZ / "models"

VERSAO_CONTRATO = "1.0"
NOME_MODELO = "luzsegura-yolo26n"

CLASSE_ACESA = "luminaria_acesa"
CLASSE_APAGADA = "luminaria_apagada"
CLASSES = (CLASSE_ACESA, CLASSE_APAGADA)

# Pesos do detector pré-treinado no COCO usados apenas pelo filtro de privacidade.
PESOS_PRIVACIDADE = "yolo26n.pt"
# Classes COCO desfocadas: pessoa e veículos (placas).
CLASSES_PRIVACIDADE = {0: "pessoa", 1: "bicicleta", 2: "carro", 3: "moto", 5: "onibus", 7: "caminhao"}

# Toda imagem é reduzida a no máximo este lado maior antes da análise (API, app, CLI e scripts):
# limita memória e tempo e faz a mesma foto ter o mesmo resultado em qualquer porta de entrada.
LADO_MAXIMO_ANALISE = 1920


def caminho_pesos_padrao() -> Path:
    """Pesos do detector de luminárias (pode ser trocado pela variável LUZSEGURA_PESOS)."""
    env = os.environ.get("LUZSEGURA_PESOS")
    return Path(env) if env else DIR_MODELOS / "luzsegura_yolo26n.pt"


@dataclass(frozen=True)
class Limiares:
    """Limiares de decisão. Valores calibrados em data/manifests e documentados no relatório."""

    # Detector
    conf_minima: float = 0.30  # abaixo disso a detecção é descartada
    conf_confiavel: float = 0.55  # acima disso a detecção dispensa revisão humana

    # Cena (luma BT.601, 0-255)
    fracao_piso: float = 0.40  # faixa inferior da imagem usada como "piso" (calçada e rua)
    piso_escuro: float = 40.0  # piso abaixo disso é considerado escuro
    piso_muito_escuro: float = 20.0  # piso abaixo disso agrava a severidade
    cena_diurna: float = 130.0  # sem horário: média da cena acima disso é tratada como foto diurna
    # (o modo noturno dos celulares clareia fotos de noite até ~120; por isso o horário é preferido)
    pixel_escuro: int = 30  # pixel abaixo disso conta como escuro

    # Período
    sol_min_acesa_de_dia: float = 8.0  # "acesa durante o dia" sem revisão só com o Sol acima disso (graus)

    # Estado da luminária pela luminosidade da própria caixa
    pixel_nucleo: int = 235  # pixel no mínimo deste valor conta como núcleo luminoso
    nucleo_min_pixels: int = 6  # área mínima (px) da maior mancha saturada para "acesa"
    contraste_max_apagada: float = 1.15  # sem núcleo e no máximo isto de contraste com o entorno: "apagada"
    lado_medida: int = 1024  # lado maior (px) em que a regra de brilho é medida: a resolução em que foi calibrada
    iou_mesma_luminaria: float = 0.7  # caixas com IoU acima disso são a mesma luminária (fica a mais confiante)


LIMIARES_PADRAO = Limiares()
