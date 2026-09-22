"""Testes da montagem do banco SQLite (dados sintéticos, banco temporário)."""

import sqlite3

import pandas as pd

from src.processing.build_database import build


def _precos_semanais():
    return pd.DataFrame(
        [
            {
                "semana": pd.Timestamp("2024-01-07"),
                "id_municipio": "3550308",
                "municipio": "São Paulo",
                "sigla_uf": "SP",
                "produto": "Gasolina",
                "unidade_medida": "R$/litro",
                "preco_medio": 5.5,
                "preco_min": 5.3,
                "preco_max": 5.7,
                "n_postos_pesquisados": 20,
                "preco_outlier": False,
            },
            {
                "semana": pd.Timestamp("2024-01-14"),
                "id_municipio": "3550308",
                "municipio": "São Paulo",
                "sigla_uf": "SP",
                "produto": "Gasolina",
                "unidade_medida": "R$/litro",
                "preco_medio": 5.6,
                "preco_min": 5.4,
                "preco_max": 5.8,
                "n_postos_pesquisados": 18,
                "preco_outlier": True,
            },
        ]
    )


def _macro_semanal():
    return pd.DataFrame(
        [
            {"semana": pd.Timestamp("2024-01-07"), "usd_brl": 5.0, "brent_usd_bbl": 80.0, "ipca_variacao_mensal": 0.5},
            {"semana": pd.Timestamp("2024-01-14"), "usd_brl": 5.1, "brent_usd_bbl": 81.0, "ipca_variacao_mensal": 0.5},
        ]
    )


def test_build_cria_schema_e_carrega_dados(tmp_path):
    destino = tmp_path / "fuel_prices.db"

    build(_precos_semanais(), _macro_semanal(), destino)

    with sqlite3.connect(destino) as conn:
        tabelas = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"dim_municipio", "precos_semanais", "macro_semanal"} <= tabelas

        indices = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")
        }
        assert {"idx_precos_municipio", "idx_precos_produto"} <= indices

        municipios = pd.read_sql("SELECT * FROM dim_municipio", conn)
        assert len(municipios) == 1
        assert municipios.iloc[0]["nome"] == "São Paulo"

        precos = pd.read_sql("SELECT * FROM precos_semanais ORDER BY semana", conn)
        assert len(precos) == 2
        assert precos.iloc[0]["semana"] == "2024-01-07"
        assert precos["outlier"].tolist() == [0, 1]

        macro = pd.read_sql("SELECT * FROM macro_semanal ORDER BY semana", conn)
        assert len(macro) == 2


def test_build_e_reexecutavel(tmp_path):
    destino = tmp_path / "fuel_prices.db"

    build(_precos_semanais(), _macro_semanal(), destino)
    build(_precos_semanais(), _macro_semanal(), destino)  # não deve duplicar nem falhar

    with sqlite3.connect(destino) as conn:
        precos = pd.read_sql("SELECT * FROM precos_semanais", conn)
        assert len(precos) == 2
