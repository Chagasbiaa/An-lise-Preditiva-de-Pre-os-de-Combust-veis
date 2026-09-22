"""Testes das funções puras de interpretabilidade SHAP (arrays fabricados,
sem chamar a biblioteca shap de verdade — o pipeline completo com
TreeExplainer foi validado manualmente contra um modelo real nesta
sessão)."""

import numpy as np

from src.models.explain import _contribuicoes_locais, _resumo_global


def test_resumo_global_ordena_por_importancia_media_absoluta():
    colunas = ["a", "b", "c"]
    # 3 linhas (semanas), 3 features. "b" tem a maior média de |shap|.
    matriz_shap = np.array(
        [
            [0.1, -0.9, 0.0],
            [-0.2, 0.8, 0.05],
            [0.15, -0.7, -0.05],
        ]
    )

    resultado = _resumo_global(matriz_shap, colunas)

    assert resultado["feature"].tolist() == ["b", "a", "c"]
    assert resultado.iloc[0]["importancia_media_abs"] == np.mean([0.9, 0.8, 0.7])


def test_contribuicoes_locais_ordena_por_magnitude():
    colunas = ["preco_medio", "usd_brl", "brent_usd_bbl"]
    valores_shap = np.array([[1.68, 0.01, -0.007]])

    resultado = _contribuicoes_locais(valores_shap, colunas)

    assert resultado["feature"].tolist() == ["preco_medio", "usd_brl", "brent_usd_bbl"]
    assert resultado.iloc[0]["contribuicao"] == 1.68


def test_contribuicoes_locais_aceita_vetor_1d():
    colunas = ["a", "b"]
    valores_shap = np.array([0.5, -0.2])

    resultado = _contribuicoes_locais(valores_shap, colunas)

    assert resultado["feature"].tolist() == ["a", "b"]
    assert resultado["contribuicao"].tolist() == [0.5, -0.2]
