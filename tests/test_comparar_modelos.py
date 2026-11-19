"""Regra pré-registrada de escolha entre o v1 e o v2 (scripts/avaliacao/comparar_modelos.py)."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("comparar_modelos", RAIZ / "scripts" / "avaliacao" / "comparar_modelos.py")
cm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cm)


def metricas(f1=0.5, sf=0.3, map50=0.5):
    m = {k: 0.0 for k in cm.METRICAS}
    m.update(chennai_f1_sistema=f1, sf311_noite_acusou_falha_sistema=sf, oi_teste_map50=map50)
    return m


def test_sem_resultados_do_v2_fica_o_v1():
    assert cm.escolher(metricas(), None)[0] == "v1"


def test_v2_melhor_sem_queda_de_map_e_escolhido():
    assert cm.escolher(metricas(), metricas(f1=0.6, map50=0.49))[0] == "v2"


def test_empate_no_criterio_fica_o_v1():
    # média 0,40 nas duas versões: o v2 precisa ser estritamente melhor
    assert cm.escolher(metricas(f1=0.5, sf=0.3), metricas(f1=0.4, sf=0.4))[0] == "v1"


@pytest.mark.parametrize("queda, esperado", [(0.03, "v2"), (0.031, "v1")])
def test_tolerancia_de_queda_no_map50(queda, esperado):
    assert cm.escolher(metricas(map50=0.5), metricas(f1=0.7, map50=round(0.5 - queda, 4)))[0] == esperado


def _montar(raiz: Path, versao: str, pesos: bytes, sha_registrado: str | None = None) -> None:
    (raiz / "models").mkdir(exist_ok=True)
    (raiz / "models" / f"luzsegura_yolo26n_{versao}.pt").write_bytes(pesos)
    sha = sha_registrado or hashlib.sha256(pesos).hexdigest()
    av = raiz / "results" / "avaliacao"
    (av / f"deteccao_{versao}_noite").mkdir(parents=True, exist_ok=True)
    (av / f"deteccao_{versao}_noite" / "confusion_matrix_normalized.png").write_bytes(b"png")
    (av / f"imagens_chennai_{versao}.csv").write_text("arquivo,rotulo\n", encoding="utf-8")
    img = {"sha256_pesos": sha,
           "chennai": {"sistema": {"f1": 0.5, "precisao": 0.6, "revocacao": 0.4}},
           "sf311_noite": {"acusou_falha_sistema": 0.3, "chamado_automatico": 0.1},
           "sf311_dia": {"acusou_acesa_de_dia": 0.2}}
    det = {"sha256_pesos": sha, "todos": {"mAP50": 0.45, "mAP50_95": 0.2}, "noite": {"mAP50": 0.5}}
    (av / f"imagens_{versao}.json").write_text(json.dumps(img), encoding="utf-8")
    (av / f"deteccao_{versao}.json").write_text(json.dumps(det), encoding="utf-8")


def _rodar(raiz: Path, monkeypatch, *extra: str) -> int:
    monkeypatch.chdir(raiz)
    monkeypatch.setattr(sys, "argv", ["comparar_modelos.py", "--resultados", "results", "--modelos", "models",
                                      "--manifestos", "manifestos", "--dataset-v2", "dataset_v2", *extra])
    return cm.main()


def test_so_v1_publica_pesos_resultados_e_registro(tmp_path, monkeypatch):
    _montar(tmp_path, "v1", b"pesos do v1")
    assert _rodar(tmp_path, monkeypatch, "--sem-v2") == 0
    final = json.loads((tmp_path / "results" / "modelo_final.json").read_text(encoding="utf-8"))
    assert set(final) == {"versao", "pesos", "sha256", "regra", "comparacao", "treino_v2"}
    assert final["versao"] == "v1" and final["comparacao"]["v2"] is None and final["treino_v2"] is None
    assert set(final["comparacao"]["v1"]) == set(cm.METRICAS)
    assert (tmp_path / "models" / "luzsegura_yolo26n.pt").read_bytes() == b"pesos do v1"
    av = tmp_path / "results" / "avaliacao"
    assert (av / "imagens.json").exists() and (av / "deteccao.json").exists() and (av / "imagens_chennai.csv").exists()
    assert (av / "deteccao_noite" / "confusion_matrix_normalized.png").exists()


def test_avaliacao_feita_com_outros_pesos_nao_publica_nada(tmp_path, monkeypatch):
    _montar(tmp_path, "v1", b"pesos do v1", sha_registrado="0" * 64)
    assert _rodar(tmp_path, monkeypatch, "--sem-v2") == 1
    assert not (tmp_path / "models" / "luzsegura_yolo26n.pt").exists()
    assert not (tmp_path / "results" / "modelo_final.json").exists()
