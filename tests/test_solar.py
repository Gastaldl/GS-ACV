from datetime import datetime, timedelta, timezone

import pytest

from luzsegura.solar import altura_sol, periodo_por_sol

SP = (-23.55, -46.63)
BRT = timezone(timedelta(hours=-3))


def test_meio_dia_em_sao_paulo_no_verao_sol_alto():
    assert altura_sol(datetime(2026, 11, 18, 12, 0, tzinfo=BRT), *SP) > 70


def test_noite_em_sao_paulo_sol_abaixo_do_horizonte():
    altura = altura_sol(datetime(2026, 11, 18, 21, 30, tzinfo=BRT), *SP)
    assert altura < -20
    assert periodo_por_sol(altura) == "noite"


def test_por_do_sol_entre_17h30_e_19h15():
    tarde = altura_sol(datetime(2026, 11, 18, 17, 30, tzinfo=BRT), *SP)
    noite = altura_sol(datetime(2026, 11, 18, 19, 15, tzinfo=BRT), *SP)
    assert tarde > 0
    assert noite < -4


def test_sem_fuso_e_recusado():
    # Tratar como UTC erraria em 3 h no Brasil: 20:00 local viraria dia.
    with pytest.raises(ValueError):
        altura_sol(datetime(2026, 11, 18, 20, 0), *SP)


def test_utc_e_fuso_local_dao_o_mesmo_resultado():
    local = altura_sol(datetime(2026, 11, 18, 21, 30, tzinfo=BRT), *SP)
    utc = altura_sol(datetime(2026, 11, 19, 0, 30, tzinfo=timezone.utc), *SP)
    assert abs(local - utc) < 1e-9


def test_faixas_do_periodo():
    assert periodo_por_sol(10) == "dia"
    assert periodo_por_sol(0) == "transicao"
    assert periodo_por_sol(-10) == "noite"
