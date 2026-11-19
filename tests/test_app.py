"""Entradas opcionais da interface web: coordenadas e horário."""

import gradio as gr
import pytest

from luzsegura.app import _coordenada, _hora


def test_coordenada_vazia_e_nao_informada():
    # Um campo numérico vazio chegaria como 0, e o Sol seria calculado em (0, 0), no oceano.
    assert _coordenada("", "Latitude", 90) is None
    assert _coordenada(None, "Longitude", 180) is None
    assert _coordenada("   ", "Longitude", 180) is None


def test_coordenada_aceita_ponto_ou_virgula():
    assert _coordenada("-23.5505", "Latitude", 90) == pytest.approx(-23.5505)
    assert _coordenada("-46,6333", "Longitude", 180) == pytest.approx(-46.6333)


@pytest.mark.parametrize("texto, limite", [("abc", 90), ("-91", 90), ("181", 180)])
def test_coordenada_invalida_ou_fora_da_faixa_e_recusada(texto, limite):
    with pytest.raises(gr.Error):
        _coordenada(texto, "Coordenada", limite)


def test_hora_exige_fuso():
    assert _hora("") is None
    assert _hora("2026-11-18T21:30:00-03:00").utcoffset() is not None
    with pytest.raises(gr.Error):
        _hora("2026-11-18T21:30:00")
