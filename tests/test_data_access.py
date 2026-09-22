"""Testes das funções de acesso ao SQLite usadas pelo dashboard, contra um
banco temporário com dados sintéticos (mesmo schema de
src/processing/build_database.py)."""

import pandas as pd
import pytest

from src.app import data_access
from src.processing.build_database import build


@pytest.fixture
def banco_temporario(tmp_path):
    precos = pd.DataFrame(
        [
            {
                "semana": pd.Timestamp("2024-01-07"), "id_municipio": "3550308", "municipio": "São Paulo",
                "sigla_uf": "SP", "produto": "Gasolina", "unidade_medida": "R$/litro",
                "preco_medio": 5.5, "preco_min": 5.3, "preco_max": 5.7,
                "n_postos_pesquisados": 20, "preco_outlier": False,
            },
            {
                "semana": pd.Timestamp("2024-01-14"), "id_municipio": "3550308", "municipio": "São Paulo",
                "sigla_uf": "SP", "produto": "Gasolina", "unidade_medida": "R$/litro",
                "preco_medio": 5.6, "preco_min": 5.4, "preco_max": 5.8,
                "n_postos_pesquisados": 18, "preco_outlier": True,
            },
            {
                "semana": pd.Timestamp("2024-01-14"), "id_municipio": "3550308", "municipio": "São Paulo",
                "sigla_uf": "SP", "produto": "Diesel", "unidade_medida": "R$/litro",
                "preco_medio": 6.0, "preco_min": 5.8, "preco_max": 6.2,
                "n_postos_pesquisados": 15, "preco_outlier": False,
            },
            {
                "semana": pd.Timestamp("2024-01-07"), "id_municipio": "3304557", "municipio": "Rio de Janeiro",
                "sigla_uf": "RJ", "produto": "Gasolina", "unidade_medida": "R$/litro",
                "preco_medio": 5.9, "preco_min": 5.7, "preco_max": 6.1,
                "n_postos_pesquisados": 10, "preco_outlier": False,
            },
        ]
    )
    macro = pd.DataFrame(
        [
            {"semana": pd.Timestamp("2024-01-07"), "usd_brl": 5.0, "brent_usd_bbl": 80.0, "ipca_variacao_mensal": 0.5},
            {"semana": pd.Timestamp("2024-01-14"), "usd_brl": 5.1, "brent_usd_bbl": 81.0, "ipca_variacao_mensal": 0.5},
        ]
    )
    destino = tmp_path / "fuel_prices.db"
    build(precos, macro, destino)
    return destino


def test_listar_ufs(banco_temporario):
    assert data_access.listar_ufs(db_path=banco_temporario) == ["RJ", "SP"]


def test_listar_municipios_filtra_por_uf(banco_temporario):
    df = data_access.listar_municipios("SP", db_path=banco_temporario)
    assert df["nome"].tolist() == ["São Paulo"]

    df_rj = data_access.listar_municipios("RJ", db_path=banco_temporario)
    assert df_rj["nome"].tolist() == ["Rio de Janeiro"]


def test_historico_precos_ordenado_e_tipado(banco_temporario):
    df = data_access.historico_precos("3550308", "Gasolina", db_path=banco_temporario)

    assert len(df) == 2
    assert df["semana"].is_monotonic_increasing
    assert df["outlier"].tolist() == [False, True]
    assert df["outlier"].dtype == bool


def test_historico_precos_vazio_para_combinacao_sem_dado(banco_temporario):
    df = data_access.historico_precos("3550308", "Gnv", db_path=banco_temporario)
    assert df.empty


def test_precos_atuais_por_produto_pega_ultima_semana_de_cada_produto(banco_temporario):
    df = data_access.precos_atuais_por_produto("3550308", db_path=banco_temporario)

    assert set(df["produto"]) == {"Gasolina", "Diesel"}
    gasolina = df[df["produto"] == "Gasolina"].iloc[0]
    assert gasolina["preco_medio"] == 5.6  # a semana mais recente (14/01), não a de 07/01
