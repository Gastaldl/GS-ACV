import json
import re
from datetime import datetime, timedelta, timezone

import pytest

from luzsegura.api import ResultadoAnalise
from luzsegura.diagnostico import (ACESA_DE_DIA, ADEQUADA, FORA_DO_PERIODO, INSUFICIENTE, LUMINARIA_APAGADA,
                                   TRECHO_ESCURO, Contexto, Luminaria, diagnosticar)
from luzsegura.luminancia import ACESA, APAGADA, INDEFINIDO, MedidaCaixa, MedidaCena

BRT = timezone(timedelta(hours=-3))
NOITE = Contexto(periodo_informado="noite")
TRANSICAO = Contexto(periodo_informado="transicao")
MEIO_DIA = Contexto(capturado_em=datetime(2026, 11, 18, 12, 0, tzinfo=BRT), lat=-23.55, lon=-46.63)
AMANHECER = Contexto(capturado_em=datetime(2026, 11, 18, 5, 40, tzinfo=BRT), lat=-23.55, lon=-46.63)  # Sol a +5°


def cena(media=30.0, piso=15.0):
    return MedidaCena(media=media, piso=piso, fracao_escura=0.6, fracao_saturada=0.01)


def lum(classe, estado, conf=0.8, contraste=None):
    if contraste is None:
        contraste = 2.0 if estado == ACESA else 0.9
    medida = MedidaCaixa(media=40, pico=250 if estado == ACESA else 90, pixels_nucleo=80 if estado == ACESA else 0,
                         fracao_nucleo=0.02 if estado == ACESA else 0, maior_mancha=80 if estado == ACESA else 0,
                         contraste=contraste, estado=estado)
    return Luminaria(classe, conf, (100, 50, 140, 150), medida)


def avisos(r):
    """Motivos de revisão como aparecem na explicação ("Atenção: ...")."""
    return re.findall(r"Atenção: (.*?)\. (?=Atenção: |Resultado: )", r["explicacao"])


def test_apagada_confirmada_no_chao_escuro_gera_prioridade_alta():
    r = diagnosticar([lum("luminaria_apagada", APAGADA), lum("luminaria_acesa", ACESA)], cena(), 640, 480, NOITE)
    assert r["diagnostico"] == LUMINARIA_APAGADA
    assert r["severidade"] == "alta"
    assert r["requer_revisao_humana"] is False
    assert r["contagem"] == {"luminaria_acesa": 1, "luminaria_apagada": 1}
    assert "confirma" in r["explicacao"]


def test_conflito_entre_modelo_e_brilho_pede_revisao():
    r = diagnosticar([lum("luminaria_apagada", ACESA)], cena(), 640, 480, NOITE)
    assert r["diagnostico"] == LUMINARIA_APAGADA
    assert r["requer_revisao_humana"] is True


def test_confianca_baixa_pede_revisao():
    r = diagnosticar([lum("luminaria_apagada", INDEFINIDO, conf=0.4)], cena(piso=80), 640, 480, NOITE)
    assert r["severidade"] == "media"
    assert r["requer_revisao_humana"] is True


def test_sem_luminaria_e_chao_escuro_e_trecho_escuro():
    r = diagnosticar([], cena(piso=8), 640, 480, NOITE)
    assert r["diagnostico"] == TRECHO_ESCURO
    assert r["severidade"] == "alta"


def test_luminaria_acesa_mas_chao_escuro_e_iluminacao_insuficiente():
    r = diagnosticar([lum("luminaria_acesa", ACESA)], cena(piso=30), 640, 480, NOITE)
    assert r["diagnostico"] == INSUFICIENTE
    assert r["severidade"] == "media"


def test_chao_claro_e_nenhuma_apagada_e_adequada():
    r = diagnosticar([lum("luminaria_acesa", ACESA)], cena(piso=90), 640, 480, NOITE)
    assert r["diagnostico"] == ADEQUADA
    assert r["severidade"] == "nenhuma"


def test_deteccao_abaixo_da_confianca_minima_e_ignorada():
    r = diagnosticar([lum("luminaria_apagada", APAGADA, conf=0.1)], cena(piso=90), 640, 480, NOITE)
    assert r["diagnostico"] == ADEQUADA
    assert r["luminarias"] == []


@pytest.mark.parametrize("luminarias, ctx", [
    pytest.param([], NOITE, id="noite"),
    pytest.param([], TRANSICAO, id="transicao"),
    pytest.param([], Contexto(), id="noite_estimada_pelo_brilho"),
    pytest.param([lum("luminaria_apagada", APAGADA, conf=0.1)], NOITE, id="so_deteccao_abaixo_da_confianca_minima"),
])
def test_chao_claro_sem_luminaria_encontrada_pede_revisao(luminarias, ctx):
    # A foto do relato deveria mostrar a luminária: sem nenhuma, "adequada" vem só do chão e não encerra o relato.
    r = diagnosticar(luminarias, cena(piso=90), 640, 480, ctx)
    assert (r["diagnostico"], r["severidade"]) == (ADEQUADA, "nenhuma")
    assert r["contagem"] == {"luminaria_acesa": 0, "luminaria_apagada": 0}
    assert r["requer_revisao_humana"] is True
    assert avisos(r) == ["nenhuma luminária foi encontrada na foto: o resultado vem só do brilho do chão"]
    assert r["acao_sugerida"] == "Enviar para revisão humana antes de encerrar o relato."
    assert r["explicacao"].endswith(r["acao_sugerida"])


@pytest.mark.parametrize("luminarias, c, ctx, diagnostico", [
    # Limitação conhecida: com uma luminária acesa encontrada e sem contradição do brilho, não há revisão.
    pytest.param([lum("luminaria_acesa", ACESA)], cena(piso=90), NOITE, ADEQUADA, id="acesa_encontrada_a_noite"),
    pytest.param([], cena(media=150, piso=140), MEIO_DIA, FORA_DO_PERIODO, id="sem_luminaria_de_dia"),
])
def test_sem_revisao_com_luminaria_acesa_encontrada_ou_de_dia(luminarias, c, ctx, diagnostico):
    r = diagnosticar(luminarias, c, 640, 480, ctx)
    assert r["diagnostico"] == diagnostico
    assert r["requer_revisao_humana"] is False
    assert avisos(r) == []


def test_luminaria_acesa_com_sol_alto_e_desperdicio():
    ctx = Contexto(capturado_em=datetime(2026, 11, 18, 12, 0, tzinfo=BRT), lat=-23.55, lon=-46.63)
    r = diagnosticar([lum("luminaria_acesa", ACESA)], cena(media=150, piso=140), 640, 480, ctx)
    assert r["periodo"] == "dia"
    assert r["periodo_fonte"] == "posicao_do_sol"
    assert r["diagnostico"] == ACESA_DE_DIA


def test_dia_estimado_pelo_brilho_nao_acusa_desperdicio():
    r = diagnosticar([lum("luminaria_acesa", ACESA)], cena(media=170, piso=150), 640, 480, Contexto())
    assert r["periodo_fonte"] == "luminosidade"
    assert r["diagnostico"] == FORA_DO_PERIODO


def test_luminaria_apagada_de_dia_nao_e_falha():
    r = diagnosticar([lum("luminaria_apagada", APAGADA)], cena(media=170, piso=150), 640, 480, Contexto())
    assert r["diagnostico"] == FORA_DO_PERIODO
    assert r["severidade"] == "nenhuma"


def test_horario_de_transicao_pede_revisao():
    r = diagnosticar([lum("luminaria_apagada", APAGADA)], cena(), 640, 480, Contexto(periodo_informado="transicao"))
    assert r["requer_revisao_humana"] is True


def test_apagada_com_periodo_estimado_pelo_brilho_pede_revisao():
    # Foto diurna com exposição automática (média 118, abaixo de cena_diurna) passa por noite: sem chamado automático.
    r = diagnosticar([lum("luminaria_apagada", APAGADA)], cena(media=118, piso=100), 640, 480, Contexto())
    assert r["periodo_fonte"] == "luminosidade"
    assert r["diagnostico"] == LUMINARIA_APAGADA
    assert r["requer_revisao_humana"] is True
    assert "confirmar que era noite" in r["explicacao"]


def test_acao_sugerida_com_revisao_vem_depois_da_confirmacao():
    r = diagnosticar([lum("luminaria_apagada", INDEFINIDO, conf=0.8)], cena(), 640, 480, NOITE)
    assert r["requer_revisao_humana"] is True
    assert r["acao_sugerida"].startswith("Enviar para revisão humana; se confirmado, abrir chamado")
    assert r["explicacao"].endswith(r["acao_sugerida"])


def test_apagada_com_brilho_indefinido_pede_revisao():
    r = diagnosticar([lum("luminaria_apagada", INDEFINIDO, conf=0.8)], cena(), 640, 480, NOITE)
    assert r["diagnostico"] == LUMINARIA_APAGADA
    assert r["requer_revisao_humana"] is True
    assert "não confirma 1 luminária apagada" in r["explicacao"]


def test_uma_apagada_nao_confirmada_basta_para_revisao():
    r = diagnosticar([lum("luminaria_apagada", APAGADA), lum("luminaria_apagada", INDEFINIDO)], cena(), 640, 480, NOITE)
    assert r["requer_revisao_humana"] is True


@pytest.mark.parametrize("piso", [60, 90])
def test_acesa_sem_ponto_de_luz_pede_revisao(piso):
    r = diagnosticar([lum("luminaria_acesa", APAGADA, conf=0.9)], cena(piso=piso), 640, 480, NOITE)
    assert r["diagnostico"] == ADEQUADA
    assert r["requer_revisao_humana"] is True
    assert avisos(r) == ["o brilho medido não encontrou ponto de luz em 1 luminária classificada como acesa; "
                         "um técnico deve revisar"]


def test_acesa_sem_ponto_de_luz_nao_segura_apagada_confirmada():
    r = diagnosticar([lum("luminaria_apagada", APAGADA), lum("luminaria_acesa", APAGADA)], cena(), 640, 480, NOITE)
    assert r["diagnostico"] == LUMINARIA_APAGADA
    assert r["requer_revisao_humana"] is False


def test_acesa_sem_ponto_de_luz_de_dia_nao_pede_revisao():
    r = diagnosticar([lum("luminaria_acesa", APAGADA)], cena(media=150, piso=140), 640, 480, MEIO_DIA)
    assert r["diagnostico"] == FORA_DO_PERIODO
    assert r["requer_revisao_humana"] is False


@pytest.mark.parametrize("luminarias, ctx, revisao", [
    pytest.param([lum("luminaria_acesa", ACESA, conf=0.31)], MEIO_DIA, True, id="confianca_baixa"),
    pytest.param([lum("luminaria_acesa", ACESA, contraste=0.75)], MEIO_DIA, True, id="sem_contraste_com_o_ceu"),
    pytest.param([lum("luminaria_acesa", ACESA, contraste=0.75), lum("luminaria_acesa", ACESA, conf=0.35)], MEIO_DIA,
                 True, id="nenhuma_confiavel_e_com_contraste"),
    pytest.param([lum("luminaria_acesa", ACESA)], AMANHECER, True, id="sol_baixo"),
    pytest.param([lum("luminaria_acesa", ACESA)], Contexto(periodo_informado="dia"), True, id="periodo_informado"),
    pytest.param([lum("luminaria_acesa", ACESA)], MEIO_DIA, False, id="evidencia_forte"),
    pytest.param([lum("luminaria_acesa", ACESA), lum("luminaria_acesa", ACESA, conf=0.35, contraste=0.75)], MEIO_DIA,
                 False, id="uma_forte_basta"),
])
def test_acesa_de_dia_so_dispensa_revisao_com_evidencia_forte(luminarias, ctx, revisao):
    r = diagnosticar(luminarias, cena(media=150, piso=140), 640, 480, ctx)
    assert r["diagnostico"] == ACESA_DE_DIA
    assert r["requer_revisao_humana"] is revisao
    assert bool(avisos(r)) is revisao


def test_sol_abaixo_do_minimo_para_acesa_de_dia():
    r = diagnosticar([lum("luminaria_acesa", ACESA)], cena(media=150, piso=140), 640, 480, AMANHECER)
    assert r["periodo"] == "dia"
    assert 2 <= r["altura_do_sol_graus"] < 8
    assert r["requer_revisao_humana"] is True


def test_periodo_dia_informado_prevalece_sobre_o_horario_e_pede_revisao():
    # Como na API: periodo=dia junto com horário e local. O informado prevalece e a altura do Sol não é conferida.
    ctx = Contexto(capturado_em=MEIO_DIA.capturado_em, lat=-23.55, lon=-46.63, periodo_informado="dia")
    r = diagnosticar([lum("luminaria_acesa", ACESA)], cena(media=150, piso=140), 640, 480, ctx)
    assert (r["periodo_fonte"], r["altura_do_sol_graus"]) == ("informado", None)
    assert r["requer_revisao_humana"] is True
    assert avisos(r) == ["com o período apenas informado, não dá para conferir se o Sol já estava alto"]


@pytest.mark.parametrize("luminarias, diagnostico, severidade", [
    # Foto apontada para cima: a faixa de baixo é céu e a luminária acesa funciona.
    pytest.param([lum("luminaria_acesa", ACESA)], INSUFICIENTE, "media", id="acesa_e_chao_muito_escuro"),
    pytest.param([], TRECHO_ESCURO, "alta", id="sem_luminaria_e_chao_muito_escuro"),
])
def test_diagnostico_pelo_chao_sempre_pede_revisao(luminarias, diagnostico, severidade):
    r = diagnosticar(luminarias, cena(piso=8), 640, 480, NOITE)
    assert r["diagnostico"] == diagnostico
    assert r["severidade"] == severidade
    assert r["requer_revisao_humana"] is True
    assert "faixa inferior da foto (40%)" in r["explicacao"]


def test_trecho_escuro_na_transicao_pede_revisao():
    r = diagnosticar([], cena(piso=8), 640, 480, TRANSICAO)
    assert r["diagnostico"] == TRECHO_ESCURO
    assert r["requer_revisao_humana"] is True
    assert "Período: transição (informado)." in r["explicacao"]
    assert any("podem ainda estar acendendo" in a for a in avisos(r))


def test_iluminacao_adequada_na_transicao_nao_pede_revisao():
    r = diagnosticar([lum("luminaria_acesa", ACESA)], cena(piso=90), 640, 480, TRANSICAO)
    assert r["diagnostico"] == ADEQUADA
    assert r["requer_revisao_humana"] is False


@pytest.mark.parametrize("luminarias, c, ctx, n", [
    pytest.param([lum("luminaria_apagada", ACESA, conf=0.4), lum("luminaria_apagada", INDEFINIDO, conf=0.35),
                  lum("luminaria_acesa", APAGADA)], cena(media=118, piso=30), Contexto(), 4,
                 id="contradita_indefinida_confianca_baixa_sem_horario"),
    pytest.param([lum("luminaria_acesa", APAGADA)], cena(piso=8), TRANSICAO, 3, id="chao_acesa_sem_luz_transicao"),
])
def test_cada_motivo_aparece_uma_vez(luminarias, c, ctx, n):
    r = diagnosticar(luminarias, c, 640, 480, ctx)
    a = avisos(r)
    assert r["requer_revisao_humana"] is True
    assert len(a) == len(set(a)) == n


@pytest.mark.parametrize("luminarias", [[], [lum("luminaria_apagada", APAGADA), lum("luminaria_acesa", INDEFINIDO)]])
def test_resultado_respeita_o_contrato(luminarias):
    ctx = Contexto(capturado_em=datetime(2026, 11, 18, 22, 0, tzinfo=BRT), lat=-23.53, lon=-46.79)
    r = diagnosticar(luminarias, cena(), 640, 480, ctx, tempo_ms=42.0)
    json.dumps(r, ensure_ascii=False)
    ResultadoAnalise.model_validate(r)
    assert r["localizacao"] == {"lat": -23.53, "lon": -46.79}
    for item in r["luminarias"]:
        assert all(0 <= v <= 1 for v in item["caixa_norm"])
