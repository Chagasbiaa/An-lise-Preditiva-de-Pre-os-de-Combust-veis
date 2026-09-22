"""Testes da montagem de features (dados sintéticos, sem I/O)."""

import numpy as np
import pandas as pd

import pytest

from src.features.build_features import montar_features, montar_features_previsao


def _serie_sintetica(n=20):
    semanas = pd.date_range("2024-01-07", periods=n, freq="W-SUN")  # já em semanas ANP (domingos)
    precos = pd.Series(np.arange(1, n + 1, dtype=float))  # 1.0, 2.0, 3.0, ...
    precos_df = pd.DataFrame({"semana": semanas, "preco_medio": precos, "outlier": 0})
    macro_df = pd.DataFrame(
        {
            "semana": semanas,
            "usd_brl": np.arange(n, dtype=float) + 5.0,
            "brent_usd_bbl": np.arange(n, dtype=float) + 80.0,
            "ipca_variacao_mensal": 0.5,
        }
    )
    return precos_df, macro_df


def test_lags_e_janela_movel_corretos_sem_vazamento():
    precos, macro = _serie_sintetica(n=20)

    resultado = montar_features(precos, macro)

    # preco_medio == i (1-indexado pela posição na série original). A
    # primeira linha usável é a que já tem lag12 (posição 13, preco=13.0).
    primeira = resultado.iloc[0]
    assert primeira["preco_medio"] == 13.0
    assert primeira["preco_lag4"] == 9.0
    assert primeira["preco_lag8"] == 5.0
    assert primeira["preco_lag12"] == 1.0
    # Janela móvel das 4 semanas ANTERIORES a t (9,10,11,12), sem incluir t.
    assert primeira["preco_media_movel_4"] == np.mean([9.0, 10.0, 11.0, 12.0])
    # Alvo é o preço da semana seguinte (t+1).
    assert primeira["preco_alvo"] == 14.0


def test_nenhuma_feature_usa_dado_posterior_a_t():
    precos, macro = _serie_sintetica(n=20)
    resultado = montar_features(precos, macro)

    for _, linha in resultado.iterrows():
        t = linha["preco_medio"]
        # Todo lag/janela deve ser estritamente menor que o preço em t
        # (a série é estritamente crescente por construção).
        assert linha["preco_lag4"] < t
        assert linha["preco_lag8"] < t
        assert linha["preco_lag12"] < t
        assert linha["preco_media_movel_4"] < t
        # O alvo é sempre o próximo valor da série (t + 1), nunca igual ou
        # anterior a t.
        assert linha["preco_alvo"] == t + 1


def test_linhas_com_alvo_outlier_sao_descartadas():
    precos, macro = _serie_sintetica(n=20)
    # A posição 14 (índice 13) vira alvo da linha da posição 13 (índice 12).
    # Marcando-a como outlier, a linha correspondente deve sumir do resultado.
    precos.loc[13, "outlier"] = 1

    resultado = montar_features(precos, macro)

    assert not (resultado["preco_alvo"] == 14.0).any()


def test_montar_features_previsao_usa_ultima_semana_sem_alvo():
    precos, macro = _serie_sintetica(n=20)

    ultima = montar_features_previsao(precos, macro)

    # Última semana da série: preco_medio == 20.0 (posição 20).
    assert ultima["preco_medio"] == 20.0
    assert ultima["preco_lag4"] == 16.0
    assert ultima["preco_lag8"] == 12.0
    assert ultima["preco_lag12"] == 8.0
    assert ultima["preco_media_movel_4"] == np.mean([16.0, 17.0, 18.0, 19.0])
    assert "preco_alvo" not in ultima.index


def test_montar_features_previsao_falha_com_serie_curta_demais():
    precos, macro = _serie_sintetica(n=5)  # menos que os 12 pontos exigidos pelo lag12

    with pytest.raises(ValueError):
        montar_features_previsao(precos, macro)
