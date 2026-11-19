"""Desenha as detecções e o painel de diagnóstico (texto com acentos via Pillow)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .config import CLASSE_ACESA
from .diagnostico import Luminaria

# Cores (RGB). Além da cor, o rótulo escrito diferencia as classes (acessível a daltônicos).
COR_ACESA = (255, 190, 0)
COR_APAGADA = (240, 50, 210)
COR_SEVERIDADE = {"nenhuma": (70, 200, 110), "baixa": (240, 220, 60), "media": (255, 150, 0), "alta": (240, 60, 60)}
PERIODO_PT = {"dia": "dia", "noite": "noite", "transicao": "transição"}
FONTE_PERIODO_PT = {"informado": "informado", "posicao_do_sol": "posição do Sol", "luminosidade": "estimado pelo brilho"}
NOME_DIAG = {
    "iluminacao_adequada": "ILUMINAÇÃO ADEQUADA",
    "luminaria_apagada": "LUMINÁRIA APAGADA",
    "iluminacao_insuficiente": "ILUMINAÇÃO INSUFICIENTE",
    "trecho_escuro": "TRECHO ESCURO",
    "fora_do_periodo_noturno": "FOTO DIURNA",
    "acesa_durante_o_dia": "ACESA DURANTE O DIA",
}
FONTES = ("arial.ttf", "Arial.ttf", "segoeui.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf")


@lru_cache(maxsize=16)
def fonte(tamanho: int) -> ImageFont.ImageFont:
    for nome in FONTES:
        try:
            return ImageFont.truetype(nome, tamanho)
        except OSError:
            continue
    try:  # fonte baixada pelo Ultralytics
        from ultralytics.utils import USER_CONFIG_DIR

        return ImageFont.truetype(str(Path(USER_CONFIG_DIR) / "Arial.ttf"), tamanho)
    except Exception:
        return ImageFont.load_default(size=tamanho)


def desenhar(img_bgr: np.ndarray, luminarias: list[Luminaria], resultado: dict, fps: float | None = None,
             titulo: str = "LUZ SEGURA · visão computacional", conf_minima: float = 0.30) -> np.ndarray:
    """Devolve uma cópia BGR com caixas, rótulos e painel do diagnóstico."""
    altura, largura = img_bgr.shape[:2]
    escala = max(0.6, largura / 1280)
    espessura = max(2, round(2.5 * escala))
    saida = img_bgr.copy()
    for l in luminarias:
        if l.confianca < conf_minima:
            continue
        cor = COR_ACESA if l.classe == CLASSE_ACESA else COR_APAGADA
        x1, y1, x2, y2 = (int(round(v)) for v in l.xyxy)
        cv2.rectangle(saida, (x1, y1), (x2, y2), cor[::-1], espessura, cv2.LINE_AA)

    pil = Image.fromarray(cv2.cvtColor(saida, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(pil, "RGBA")
    f_rotulo = fonte(max(12, round(15 * escala)))
    for l in luminarias:
        if l.confianca < conf_minima:
            continue
        cor = COR_ACESA if l.classe == CLASSE_ACESA else COR_APAGADA
        texto = f"{'ACESA' if l.classe == CLASSE_ACESA else 'APAGADA'} {l.confianca:.0%}"
        if l.concorda_brilho is False:
            texto += " ?"
        x1, y1 = int(l.xyxy[0]), int(l.xyxy[1])
        caixa = d.textbbox((0, 0), texto, font=f_rotulo)
        tw, th = caixa[2] - caixa[0], caixa[3] - caixa[1]
        ty = y1 - th - 6 if y1 - th - 6 > 0 else y1 + 2
        d.rectangle((x1, ty, x1 + tw + 8, ty + th + 6), fill=cor + (230,))
        d.text((x1 + 4, ty + 1), texto, font=f_rotulo, fill=(0, 0, 0))

    # Painel (os códigos do contrato JSON viram texto em português com acentos)
    f_titulo = fonte(max(13, round(16 * escala)))
    f_diag = fonte(max(16, round(24 * escala)))
    f_txt = fonte(max(12, round(15 * escala)))
    sev = resultado["severidade"]
    cont = resultado["contagem"]
    linhas = [
        (titulo, f_titulo, (200, 200, 200)),
        (NOME_DIAG.get(resultado["diagnostico"], resultado["diagnostico"].upper())
         + ("" if sev == "nenhuma" else f" · prioridade {sev.replace('media', 'média')}"), f_diag, COR_SEVERIDADE[sev]),
        (f"Acesas: {cont['luminaria_acesa']}   Apagadas: {cont['luminaria_apagada']}   "
         f"Chão: {resultado['luminosidade']['piso_media']:.0f}/255", f_txt, (255, 255, 255)),
        (f"Período: {PERIODO_PT.get(resultado['periodo'], resultado['periodo'])} "
         f"({FONTE_PERIODO_PT.get(resultado['periodo_fonte'], resultado['periodo_fonte'])})", f_txt, (255, 255, 255)),
    ]
    priv = resultado.get("privacidade") or {}
    if priv.get("filtro_ativo"):
        linhas.append((f"Privacidade: {priv['pessoas_desfocadas']} pessoa(s) e {priv['veiculos_desfocados']} "
                       "veículo(s) desfocados", f_txt, (180, 220, 255)))
    if resultado.get("requer_revisao_humana"):
        linhas.append(("Revisão humana recomendada", f_txt, (255, 210, 120)))
    if fps is not None:
        # Velocidade da análise (detecção, brilho, privacidade e diagnóstico), sem o desenho e a gravação.
        linhas.append((f"Velocidade da análise: {fps:.1f} quadros/s (CPU)".replace(".", ","), f_txt, (170, 255, 170)))
    margem = round(10 * escala)
    larguras, alturas = [], []
    for texto, f, _ in linhas:
        b = d.textbbox((0, 0), texto, font=f)
        larguras.append(b[2] - b[0])
        alturas.append(b[3] - b[1])
    painel_w = max(larguras) + 2 * margem
    painel_h = sum(alturas) + margem * (len(linhas) + 1)
    d.rectangle((0, 0, painel_w, painel_h), fill=(0, 0, 0, 170))
    y = margem
    for (texto, f, cor), h in zip(linhas, alturas):
        d.text((margem, y - 2), texto, font=f, fill=cor)
        y += h + margem
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
