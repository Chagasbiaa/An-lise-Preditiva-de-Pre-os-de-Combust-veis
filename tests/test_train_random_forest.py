"""Teste de ponta a ponta do treino/avaliação do Random Forest, com uma
matriz de features sintética (sinal forte e determinístico, sem tocar o
banco de dados real)."""

import joblib
import numpy as np
import pandas as pd
import pytest

from src import config
from src.models import train_random_forest


def _features_sinteticas(n=300, seed=0):
    rng = np.random.default_rng(seed)
    semanas = pd.date_range("2015-01-04", periods=n, freq="W-SUN")

    # Sinal forte e quase determinístico: alvo é ~3x uma feature + ruído
    # minúsculo. Random Forest deve aprender isso com folga. O preditor é
    # sorteado (não cresce com o tempo) para que treino e holdout cubram a
    # mesma faixa de valores — Random Forest não extrapola bem além do
    # intervalo visto no treino (limitação documentada no TCC), então um
    # preditor monotonicamente crescente com o tempo deixaria o holdout
    # (as últimas semanas) sistematicamente fora da faixa de treino.
    preditor_forte = rng.uniform(0, 100, size=n)
    ruido = rng.normal(scale=0.05, size=n)
    alvo = 3 * preditor_forte + ruido

    return pd.DataFrame(
        {
            "semana": semanas,
            "preco_medio": preditor_forte,
            "preco_lag4": preditor_forte,
            "usd_brl": rng.normal(size=n),  # ruído puro, não informativo
            "preco_alvo": alvo,
        }
    )


def test_train_and_evaluate_ponta_a_ponta(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(train_random_forest, "build_features", lambda municipio, produto: _features_sinteticas())

    resultado = train_random_forest.train_and_evaluate("0000000", "Fake", n_splits=5, semanas_holdout=52)

    for chave in ("mae", "rmse", "mape", "r2"):
        assert chave in resultado["metricas_holdout"]
        assert chave in resultado["metricas_cv"]
        assert "media" in resultado["metricas_cv"][chave]
        assert "desvio_padrao" in resultado["metricas_cv"][chave]

    # Sinal forte por construção -> o modelo deve explicar a maior parte
    # da variância mesmo no holdout nunca visto.
    assert resultado["metricas_holdout"]["r2"] > 0.9

    assert resultado["caminho_modelo"].exists()
    assert resultado["caminho_metadados"].exists()

    carregado = joblib.load(resultado["caminho_modelo"])
    assert carregado["colunas_feature"] == ["preco_medio", "preco_lag4", "usd_brl"]
    entrada = pd.DataFrame([[50.0, 50.0, 0.0]], columns=carregado["colunas_feature"])
    previsao = carregado["modelo"].predict(entrada)[0]
    assert previsao == pytest.approx(150.0, abs=5.0)
