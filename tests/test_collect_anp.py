"""Teste do coletor da ANP com resposta mockada do BigQuery (não depende de
credenciais reais do Google Cloud)."""

import pandas as pd
import pytest

from src.data import collect_anp


@pytest.fixture(autouse=True)
def _configura_ambiente(tmp_path, monkeypatch):
    monkeypatch.setattr(collect_anp, "BASEDOSDADOS_PROJECT_ID", "projeto-fake")
    monkeypatch.setattr(collect_anp, "ARQUIVO_SAIDA", tmp_path / "anp_precos.parquet")


def test_collect_salva_resultado_da_query(monkeypatch):
    fake_df = pd.DataFrame(
        {
            "data_coleta": pd.to_datetime(["2024-01-01", "2024-01-08"]),
            "ano": pd.array([2024, 2024], dtype="int32"),
            "sigla_uf": ["SP", "SP"],
            "id_municipio": ["3550308", "3550308"],
            "municipio": ["São Paulo", "São Paulo"],
            "produto": ["Gasolina", "Gasolina"],
            "unidade_medida": ["R$/litro", "R$/litro"],
            "preco_venda_medio": [5.79, 5.85],
            "preco_venda_min": [5.5, 5.6],
            "preco_venda_max": [6.1, 6.2],
            "n_postos_pesquisados": pd.array([10, 12], dtype="int32"),
        }
    )

    def fake_read_sql(query, billing_project_id=None):
        assert "microdados" in query
        assert billing_project_id == "projeto-fake"
        return fake_df

    import basedosdados

    monkeypatch.setattr(basedosdados, "read_sql", fake_read_sql)

    resultado = collect_anp.collect(force=True)

    assert resultado.equals(fake_df)
    assert collect_anp.ARQUIVO_SAIDA.exists()
    assert pd.read_parquet(collect_anp.ARQUIVO_SAIDA).equals(fake_df)


def test_collect_falha_sem_project_id(monkeypatch):
    monkeypatch.setattr(collect_anp, "BASEDOSDADOS_PROJECT_ID", None)

    with pytest.raises(RuntimeError):
        collect_anp.collect(force=True)
