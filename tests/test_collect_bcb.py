"""Teste real (sem mock) do coletor de séries do Banco Central — API pública,
não exige credenciais."""

import datetime as dt

import pandas as pd

from src.data.collect_bcb import _fetch_sgs_series


def test_fetch_sgs_series_returns_recent_nonempty_data():
    inicio = (dt.date.today() - dt.timedelta(days=30)).strftime("%d/%m/%Y")

    df = _fetch_sgs_series(codigo=1, data_inicial=inicio)  # série 1 = USD/BRL

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert list(df.columns) == ["data", "valor"]
    assert pd.api.types.is_datetime64_any_dtype(df["data"])
    assert pd.api.types.is_float_dtype(df["valor"])
    assert (df["valor"] > 0).all()
