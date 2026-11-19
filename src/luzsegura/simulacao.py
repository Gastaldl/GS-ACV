"""Simulação de noite a partir de fotos diurnas (dados simulados, documentados no relatório).

Luminárias fotografadas de dia estão apagadas. Escurecendo a foto de forma
fisicamente plausível (gama, exposição menor, céu mais escuro no topo,
tonalidade de iluminação artificial e ruído de sensor), obtemos exemplos de
"luminária apagada em cena noturna", a classe mais rara nos dados públicos.
Todas as imagens geradas ficam marcadas como simuladas no manifesto do dataset.
"""

from __future__ import annotations

import cv2
import numpy as np

# Tonalidades em BGR: noite azulada (LED branco frio, luar) ou alaranjada (vapor de sódio).
TONALIDADES = (
    np.array([1.15, 0.97, 0.82], np.float32),
    np.array([0.72, 0.92, 1.15], np.float32),
)


def simular_noite(img_bgr: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Converte uma foto diurna em uma versão noturna simulada.

    Devolve a imagem e os parâmetros sorteados (registrados no manifesto).
    """
    img = img_bgr.astype(np.float32) / 255.0
    gama = float(rng.uniform(2.2, 3.2))
    exposicao = float(rng.uniform(0.35, 0.65))
    topo = float(rng.uniform(0.25, 0.60))
    tonalidade = int(rng.integers(0, len(TONALIDADES)))
    sigma_ruido = float(rng.uniform(0.008, 0.02))
    desfoque = bool(rng.random() < 0.5)

    img = np.power(img, gama) * exposicao
    altura = img.shape[0]
    gradiente = np.linspace(topo, 1.0, altura, dtype=np.float32)[:, None, None]
    img = img * gradiente * TONALIDADES[tonalidade]
    img = np.clip(img, 0.0, 1.0)
    img = np.clip(img + rng.normal(0.0, sigma_ruido, img.shape).astype(np.float32), 0.0, 1.0)
    saida = (img * 255.0 + 0.5).astype(np.uint8)
    if desfoque:
        saida = cv2.GaussianBlur(saida, (3, 3), 0)
    parametros = {"gama": round(gama, 3), "exposicao": round(exposicao, 3), "topo": round(topo, 3),
                  "tonalidade": ("azulada", "alaranjada")[tonalidade], "ruido": round(sigma_ruido, 4),
                  "desfoque": desfoque}
    return saida, parametros
