"""Teste do coletor de Brent com resposta mockada da API da EIA (não depende
de uma API key real)."""

import pandas as pd
import pytest

from src.data import collect_eia_brent


@pytest.fixture(autouse=True)
def _configura_ambiente(tmp_path, monkeypatch):
    monkeypatch.setattr(collect_eia_brent, "EIA_API_KEY", "chave-fake")
    monkeypatch.setattr(collect_eia_brent, "ARQUIVO_SAIDA", tmp_path / "brent.parquet")


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_collect_pagina_e_salva_resultado(monkeypatch):
    pagina_1 = [{"period": "2024-01-01", "value": "78.5"}, {"period": "2024-01-02", "value": "79.1"}]

    def fake_get(url, params=None, timeout=None):
        assert params["facets[series][]"] == "RBRTE"
        return _FakeResponse({"response": {"data": pagina_1}})

    monkeypatch.setattr(collect_eia_brent.requests, "get", fake_get)

    resultado = collect_eia_brent.collect(force=True)

    assert list(resultado.columns) == ["data", "brent_usd_bbl"]
    assert len(resultado) == 2
    assert resultado["brent_usd_bbl"].tolist() == [78.5, 79.1]
    assert collect_eia_brent.ARQUIVO_SAIDA.exists()


def test_collect_falha_sem_api_key(monkeypatch):
    monkeypatch.setattr(collect_eia_brent, "EIA_API_KEY", None)

    with pytest.raises(RuntimeError):
        collect_eia_brent.collect(force=True)
