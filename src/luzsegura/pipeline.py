"""Pipeline de uma imagem: detectar, medir, anonimizar, diagnosticar e desenhar."""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter

import cv2
import numpy as np

from .config import LADO_MAXIMO_ANALISE, LIMIARES_PADRAO, Limiares
from .desenho import desenhar
from .detector import DetectorLuminarias
from .diagnostico import Contexto, diagnosticar
from .luminancia import luma, medir_cena
from .privacidade import FiltroPrivacidade, sem_filtro


def reduzir(img: np.ndarray, lado_max: int = LADO_MAXIMO_ANALISE) -> np.ndarray:
    """Reduz a imagem para o lado maior caber em lado_max (nunca amplia; devolve a mesma imagem se já cabe)."""
    h, w = img.shape[:2]
    if max(h, w) <= lado_max:
        return img
    f = lado_max / max(h, w)
    return cv2.resize(img, (max(1, round(w * f)), max(1, round(h * f))), interpolation=cv2.INTER_AREA)


@dataclass
class Analisador:
    """Junta detector, filtro de privacidade e diagnóstico. A imagem original nunca é gravada."""

    detector: DetectorLuminarias
    privacidade: FiltroPrivacidade | None = None
    lim: Limiares = field(default_factory=lambda: LIMIARES_PADRAO)

    def analisar(self, img_bgr: np.ndarray, ctx: Contexto | None = None,
                 desenhar_saida: bool = True) -> tuple[dict, np.ndarray | None]:
        inicio = perf_counter()
        img_bgr = reduzir(img_bgr)
        luminarias = self.detector.detectar(img_bgr)
        cena = medir_cena(luma(img_bgr), self.lim)
        if self.privacidade is not None:
            segura, info_privacidade = self.privacidade.aplicar(img_bgr)
        else:
            segura, info_privacidade = img_bgr, sem_filtro()
        altura, largura = img_bgr.shape[:2]
        tempo_ms = (perf_counter() - inicio) * 1000
        resultado = diagnosticar(luminarias, cena, largura, altura, ctx, self.lim, tempo_ms, info_privacidade)
        anotada = desenhar(segura, luminarias, resultado, conf_minima=self.lim.conf_minima) if desenhar_saida else None
        return resultado, anotada


def criar_analisador(pesos=None, privacidade: bool = True, imgsz: int = 640, threads: int | None = None) -> Analisador:
    detector = DetectorLuminarias(pesos, imgsz=imgsz, threads=threads)
    filtro = FiltroPrivacidade(imgsz=imgsz) if privacidade else None
    return Analisador(detector, filtro)
