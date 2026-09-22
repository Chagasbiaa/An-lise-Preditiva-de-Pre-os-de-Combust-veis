"""Testes das funções puras da simulação what-if do dashboard (sem
Streamlit, sem I/O)."""

import pandas as pd

from src.app.dashboard import (
    EVENTO_COLUNA_DURACAO,
    EVENTO_COLUNA_INICIO,
    _fora_da_faixa,
    _montar_cenario,
)


def test_fora_da_faixa():
    faixa = (5.0, 6.0)
    assert _fora_da_faixa(4.99, faixa) is True
    assert _fora_da_faixa(6.01, faixa) is True
    assert _fora_da_faixa(5.5, faixa) is False
    # Nos limites (inclusive) não conta como fora.
    assert _fora_da_faixa(5.0, faixa) is False
    assert _fora_da_faixa(6.0, faixa) is False


def _ultima_linha_sintetica() -> pd.Series:
    return pd.Series(
        {
            "semana": pd.Timestamp("2026-08-02"),
            "preco_medio": 6.33,
            "preco_lag4": 6.20,
            "preco_lag8": 6.10,
            "preco_lag12": 6.00,
            "preco_media_movel_4": 6.25,
            "preco_desvio_movel_4": 0.05,
            "usd_brl": 5.11,
            "brent_usd_bbl": 68.0,
            "ipca_variacao_mensal": -0.32,
            EVENTO_COLUNA_INICIO: 0,
            EVENTO_COLUNA_DURACAO: 234,
        }
    )


COLUNAS_FEATURE = [
    "preco_medio", "preco_lag4", "preco_lag8", "preco_lag12",
    "preco_media_movel_4", "preco_desvio_movel_4",
    "usd_brl", "brent_usd_bbl", "ipca_variacao_mensal",
    EVENTO_COLUNA_INICIO, EVENTO_COLUNA_DURACAO,
]


def test_montar_cenario_sobrescreve_so_as_variaveis_macro():
    ultima_linha = _ultima_linha_sintetica()

    cenario = _montar_cenario(
        COLUNAS_FEATURE, ultima_linha,
        usd_brl=8.0, brent_usd_bbl=120.0, ipca_variacao_mensal=1.5,
        novo_choque_geopolitico=False,
    )

    assert cenario["usd_brl"] == 8.0
    assert cenario["brent_usd_bbl"] == 120.0
    assert cenario["ipca_variacao_mensal"] == 1.5
    # O resto (preço atual, lags, janela móvel) não muda.
    assert cenario["preco_medio"] == ultima_linha["preco_medio"]
    assert cenario["preco_lag4"] == ultima_linha["preco_lag4"]
    assert cenario["preco_lag8"] == ultima_linha["preco_lag8"]
    assert cenario["preco_lag12"] == ultima_linha["preco_lag12"]
    assert cenario["preco_media_movel_4"] == ultima_linha["preco_media_movel_4"]
    # Sem marcar o choque geopolítico, os campos de evento ficam como estavam.
    assert cenario[EVENTO_COLUNA_INICIO] == ultima_linha[EVENTO_COLUNA_INICIO]
    assert cenario[EVENTO_COLUNA_DURACAO] == ultima_linha[EVENTO_COLUNA_DURACAO]


def test_montar_cenario_com_novo_choque_geopolitico():
    ultima_linha = _ultima_linha_sintetica()

    cenario = _montar_cenario(
        COLUNAS_FEATURE, ultima_linha,
        usd_brl=5.11, brent_usd_bbl=68.0, ipca_variacao_mensal=-0.32,
        novo_choque_geopolitico=True,
    )

    assert cenario[EVENTO_COLUNA_INICIO] == 1
    assert cenario[EVENTO_COLUNA_DURACAO] == 0


def test_montar_cenario_nao_altera_a_serie_original():
    ultima_linha = _ultima_linha_sintetica()
    original = ultima_linha.copy()

    _montar_cenario(
        COLUNAS_FEATURE, ultima_linha,
        usd_brl=99.0, brent_usd_bbl=999.0, ipca_variacao_mensal=99.0,
        novo_choque_geopolitico=True,
    )

    pd.testing.assert_series_equal(ultima_linha, original)
