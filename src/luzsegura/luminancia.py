"""Medidas de luminosidade que explicam e conferem as detecções do modelo.

A câmera não é um luxímetro: a exposição automática do celular altera o brilho
da foto. Por isso as medidas são relativas (luma de 0 a 255). Elas servem para
explicar a decisão ao cidadão e para apontar casos de revisão humana, não para
medir iluminância em lux. A medição física em lux fica com o dispositivo com
sensor LDR do componente Maker Lab.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .config import LIMIARES_PADRAO, Limiares

ACESA = "acesa"
APAGADA = "apagada"
INDEFINIDO = "indefinido"


def luma(img_bgr: np.ndarray) -> np.ndarray:
    """Converte BGR em luma BT.601 (0-255, uint8)."""
    if img_bgr.ndim == 2:
        return img_bgr
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


@dataclass(frozen=True)
class MedidaCena:
    """Luminosidade da cena inteira e da faixa inferior (piso: calçada e rua)."""

    media: float
    piso: float
    fracao_escura: float
    fracao_saturada: float

    def como_dict(self) -> dict:
        return {
            "cena_media": round(self.media, 1),
            "piso_media": round(self.piso, 1),
            "fracao_escura": round(self.fracao_escura, 4),
            "fracao_saturada": round(self.fracao_saturada, 4),
        }


def medir_cena(y: np.ndarray, lim: Limiares = LIMIARES_PADRAO) -> MedidaCena:
    """Mede a cena a partir da luma. O piso é a faixa inferior da imagem (Limiares.fracao_piso).

    Numa foto apontada para cima essa faixa pode ser céu ou fachada, e não calçada: por isso
    os diagnósticos de chão escuro (iluminação insuficiente, trecho escuro) sempre vão para
    revisão humana (luzsegura.diagnostico).
    """
    altura = y.shape[0]
    inicio_piso = min(altura - 1, int(round(altura * (1.0 - lim.fracao_piso))))
    piso = y[inicio_piso:, :]
    return MedidaCena(
        media=float(y.mean()),
        piso=float(piso.mean()),
        fracao_escura=float((y < lim.pixel_escuro).mean()),
        fracao_saturada=float((y >= lim.pixel_nucleo).mean()),
    )


@dataclass(frozen=True)
class MedidaCaixa:
    """Brilho dentro da caixa de uma luminária, comparado com o entorno."""

    media: float
    pico: float  # percentil 99 da luma dentro da caixa
    pixels_nucleo: int  # pixels com luma >= Limiares.pixel_nucleo (núcleo luminoso)
    fracao_nucleo: float
    maior_mancha: int  # área (px) da maior mancha contígua de pixels saturados
    contraste: float  # média da caixa / média do anel em volta (>1: mais clara que o entorno)
    estado: str  # acesa | apagada | indefinido

    def como_dict(self) -> dict:
        d = asdict(self)
        for k in ("media", "pico", "contraste"):
            d[k] = round(d[k], 2)
        d["fracao_nucleo"] = round(d["fracao_nucleo"], 5)
        return d


def _recortar(xyxy, largura: int, altura: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    x1 = int(np.clip(np.floor(x1), 0, largura - 1))
    y1 = int(np.clip(np.floor(y1), 0, altura - 1))
    x2 = int(np.clip(np.ceil(x2), x1 + 1, largura))
    y2 = int(np.clip(np.ceil(y2), y1 + 1, altura))
    return x1, y1, x2, y2


def medir_caixa(y: np.ndarray, xyxy, lim: Limiares = LIMIARES_PADRAO, expansao: float = 0.5) -> MedidaCaixa:
    """Mede o brilho da caixa (em pixels, xyxy) e do anel ao redor dela."""
    altura, largura = y.shape[:2]
    x1, y1, x2, y2 = _recortar(xyxy, largura, altura)
    recorte = y[y1:y2, x1:x2]
    area = recorte.size
    mascara = (recorte >= lim.pixel_nucleo).astype(np.uint8)
    nucleo = int(np.count_nonzero(mascara))
    fracao = nucleo / area if area else 0.0
    pico = float(np.percentile(recorte, 99)) if area else 0.0
    media = float(recorte.mean()) if area else 0.0
    maior_mancha = 0
    if nucleo:
        n, _, stats, _ = cv2.connectedComponentsWithStats(mascara, connectivity=8)
        if n > 1:
            maior_mancha = int(stats[1:, cv2.CC_STAT_AREA].max())

    # Anel: caixa expandida menos a própria caixa.
    dx, dy = (x2 - x1) * expansao, (y2 - y1) * expansao
    ex1, ey1, ex2, ey2 = _recortar((x1 - dx, y1 - dy, x2 + dx, y2 + dy), largura, altura)
    externo = y[ey1:ey2, ex1:ex2].astype(np.float64)
    soma_anel = externo.sum() - recorte.sum()
    n_anel = externo.size - area
    media_anel = soma_anel / n_anel if n_anel > 0 else media
    contraste = (media + 1.0) / (media_anel + 1.0)

    estado = classificar_estado(maior_mancha, contraste, lim)
    return MedidaCaixa(media, pico, nucleo, fracao, maior_mancha, float(contraste), estado)


def classificar_estado(maior_mancha: int, contraste: float, lim: Limiares = LIMIARES_PADRAO) -> str:
    """Regra explicável.

    - acesa: existe uma mancha contígua de pixels saturados (o ponto de luz);
    - apagada: nenhum ponto de luz e a caixa não é mais clara que o entorno;
    - indefinido: casos intermediários (reflexo, brilho difuso), que não confirmam o modelo:
      à noite, a luminária que ele chama de apagada vai para revisão humana.
    """
    if maior_mancha >= lim.nucleo_min_pixels:
        return ACESA
    if maior_mancha == 0 and contraste <= lim.contraste_max_apagada:
        return APAGADA
    return INDEFINIDO
