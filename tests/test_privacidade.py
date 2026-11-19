import numpy as np

from luzsegura.privacidade import anonimizar_regiao


def test_anonimizacao_altera_so_a_regiao():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (240, 320, 3), dtype=np.uint8)
    original = img.copy()
    anonimizar_regiao(img, (100, 60, 200, 180), margem=0.0)
    dentro = (slice(60, 180), slice(100, 200))
    assert not np.array_equal(img[dentro], original[dentro])
    # Fora da região nada muda.
    mascara = np.ones(img.shape[:2], bool)
    mascara[dentro] = False
    assert np.array_equal(img[mascara], original[mascara])


def test_anonimizacao_remove_detalhe():
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, (200, 200, 3), dtype=np.uint8)
    antes = img[40:160, 40:160].astype(float).std()
    anonimizar_regiao(img, (40, 40, 160, 160), margem=0.0)
    depois = img[40:160, 40:160].astype(float).std()
    assert depois < antes * 0.5


def test_regiao_minuscula_ou_fora_da_imagem_nao_quebra():
    img = np.zeros((50, 50, 3), np.uint8)
    anonimizar_regiao(img, (10, 10, 11, 11))
    anonimizar_regiao(img, (60, 60, 90, 90))
    assert img.sum() == 0
