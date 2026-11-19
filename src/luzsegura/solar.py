"""Altura do Sol pelo algoritmo da NOAA, para saber se a luminária deveria estar acesa.

Luminária apagada só é falha depois do pôr do sol. Fotos de celular no modo noturno
ficam claras, e o céu ainda está claro no anoitecer: a luminosidade da imagem engana.
Com horário e localização, o sistema calcula a altura do Sol e decide pelo critério
físico. Precisão do algoritmo: melhor que 0,1° entre 1901 e 2099 (sem refração).
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

# Fotocélulas de iluminação pública acendem com o Sol pouco abaixo do horizonte.
LIMITE_DIA = 2.0  # acima disso: dia (luminária deveria estar desligada)
LIMITE_NOITE = -4.0  # abaixo disso: noite (luminária deveria estar acesa)


def altura_sol(momento: datetime, lat: float, lon: float) -> float:
    """Altura do Sol em graus (negativa abaixo do horizonte). Exige data e hora com fuso.

    Horário sem fuso é recusado: tratado como UTC, 20:00 em São Paulo viraria 17:00, ainda dia,
    e a luminária apagada passaria como "fora do período noturno".
    """
    if momento.utcoffset() is None:
        raise ValueError("altura_sol precisa de data e hora com fuso, ex.: 2026-11-18T21:30:00-03:00")
    utc = momento.astimezone(timezone.utc)
    jd = utc.timestamp() / 86400.0 + 2440587.5
    jc = (jd - 2451545.0) / 36525.0

    l0 = (280.46646 + jc * (36000.76983 + jc * 0.0003032)) % 360.0
    m = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    e = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    mr = math.radians(m)
    c = (math.sin(mr) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
         + math.sin(2 * mr) * (0.019993 - 0.000101 * jc) + math.sin(3 * mr) * 0.000289)
    longitude_aparente = l0 + c - 0.00569 - 0.00478 * math.sin(math.radians(125.04 - 1934.136 * jc))
    obliquidade_media = 23 + (26 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813))) / 60) / 60
    obliquidade = obliquidade_media + 0.00256 * math.cos(math.radians(125.04 - 1934.136 * jc))
    declinacao = math.asin(math.sin(math.radians(obliquidade)) * math.sin(math.radians(longitude_aparente)))

    y = math.tan(math.radians(obliquidade / 2)) ** 2
    l0r = math.radians(l0)
    equacao_tempo = 4 * math.degrees(
        y * math.sin(2 * l0r) - 2 * e * math.sin(mr) + 4 * e * y * math.sin(mr) * math.cos(2 * l0r)
        - 0.5 * y * y * math.sin(4 * l0r) - 1.25 * e * e * math.sin(2 * mr)
    )
    minutos_utc = utc.hour * 60 + utc.minute + utc.second / 60
    tempo_solar = (minutos_utc + equacao_tempo + 4 * lon) % 1440
    angulo_horario = tempo_solar / 4 - 180 if tempo_solar / 4 >= 0 else tempo_solar / 4 + 180

    latr = math.radians(lat)
    cos_zenite = (math.sin(latr) * math.sin(declinacao)
                  + math.cos(latr) * math.cos(declinacao) * math.cos(math.radians(angulo_horario)))
    zenite = math.degrees(math.acos(max(-1.0, min(1.0, cos_zenite))))
    return 90.0 - zenite


def periodo_por_sol(altura: float) -> str:
    """dia | transicao | noite, segundo a altura do Sol."""
    if altura >= LIMITE_DIA:
        return "dia"
    if altura >= LIMITE_NOITE:
        return "transicao"
    return "noite"
