"""Acesso ao SQLite (data/fuel_prices.db) para o dashboard — isola todo o
SQL usado pelas páginas Streamlit, com cache (`st.cache_data`) para evitar
reconsultas a cada interação do usuário."""

from __future__ import annotations

import sqlite3

import pandas as pd
import streamlit as st

from src.config import DB_PATH
from src.processing.clean_anp import UNIDADE_POR_PRODUTO

PRODUTOS = list(UNIDADE_POR_PRODUTO.keys())


@st.cache_data
def listar_ufs(db_path=DB_PATH) -> list[str]:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT DISTINCT sigla_uf FROM dim_municipio ORDER BY sigla_uf", conn)
    return df["sigla_uf"].tolist()


@st.cache_data
def listar_municipios(sigla_uf: str, db_path=DB_PATH) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(
            "SELECT id_municipio, nome FROM dim_municipio WHERE sigla_uf = ? ORDER BY nome",
            conn,
            params=(sigla_uf,),
        )


@st.cache_data
def historico_precos(id_municipio: str, produto: str, db_path=DB_PATH) -> pd.DataFrame:
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            "SELECT semana, preco_medio, preco_min, preco_max, n_postos_pesquisados, outlier "
            "FROM precos_semanais WHERE id_municipio = ? AND produto = ? ORDER BY semana",
            conn,
            params=(id_municipio, produto),
        )
    df["semana"] = pd.to_datetime(df["semana"])
    df["outlier"] = df["outlier"].astype(bool)
    return df


@st.cache_data
def precos_atuais_por_produto(id_municipio: str, db_path=DB_PATH) -> pd.DataFrame:
    """Última semana disponível de cada produto pesquisado na cidade.

    Usa ROW_NUMBER() em vez de uma subquery correlacionada (MAX por linha)
    — a versão correlacionada chegou a levar mais de 60s na tabela de 1,57
    milhão de linhas; esta leva ~20ms, pois só varre uma vez as linhas já
    filtradas por id_municipio (via idx_precos_municipio)."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            """
            WITH ultimas AS (
                SELECT produto, semana, preco_medio,
                       ROW_NUMBER() OVER (PARTITION BY produto ORDER BY semana DESC) AS rn
                FROM precos_semanais
                WHERE id_municipio = ?
            )
            SELECT produto, semana, preco_medio FROM ultimas WHERE rn = 1 ORDER BY produto
            """,
            conn,
            params=(id_municipio,),
        )
    df["semana"] = pd.to_datetime(df["semana"])
    return df
