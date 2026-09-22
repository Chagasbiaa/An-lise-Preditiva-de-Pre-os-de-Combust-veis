"""Testes da compatibilização temporal das séries macro (dados sintéticos,
sem I/O)."""

import pandas as pd

from src.processing.clean_macro import clean


def test_cambio_e_brent_viram_media_semanal():
    # Semana ANP de 2024-01-01 (segunda) a 2024-01-06 (sábado).
    cambio = pd.DataFrame(
        {
            "data": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "usd_brl": [5.0, 5.2, 5.4],
        }
    )
    brent = pd.DataFrame(
        {
            "data": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "brent_usd_bbl": [80.0, 82.0],
        }
    )
    ipca = pd.DataFrame({"data": pd.to_datetime(["2024-01-01"]), "ipca_variacao_mensal": [0.5]})

    macro = clean(cambio, brent, ipca)

    semana = pd.Timestamp("2023-12-31")  # domingo que inicia a semana de 01/01 a 06/01
    linha = macro[macro["semana"] == semana].iloc[0]
    assert linha["usd_brl"] == (5.0 + 5.2 + 5.4) / 3
    assert linha["brent_usd_bbl"] == (80.0 + 82.0) / 2


def test_ipca_forward_fill_sem_vazamento():
    cambio = pd.DataFrame(
        {"data": pd.date_range("2024-01-01", "2024-02-29", freq="D"), "usd_brl": 5.0}
    )
    brent = pd.DataFrame(
        {"data": pd.date_range("2024-01-01", "2024-02-29", freq="D"), "brent_usd_bbl": 80.0}
    )
    ipca = pd.DataFrame(
        {"data": pd.to_datetime(["2024-01-01", "2024-02-01"]), "ipca_variacao_mensal": [0.5, 0.3]}
    )

    macro = clean(cambio, brent, ipca)
    macro = macro.sort_values("semana")

    # A 1ª semana (antes de 2024-01-01) fica sem IPCA — ainda não havia
    # nenhum valor publicado para preencher (comportamento esperado do
    # forward fill: sem vazar informação futura nem inventar passado).
    antes_fevereiro = macro[(macro["semana"] >= "2024-01-01") & (macro["semana"] < "2024-02-01")]
    depois_fevereiro = macro[macro["semana"] >= "2024-02-04"]  # já dentro de uma semana só de fevereiro

    assert (antes_fevereiro["ipca_variacao_mensal"] == 0.5).all()
    assert (depois_fevereiro["ipca_variacao_mensal"] == 0.3).all()
