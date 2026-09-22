"""Engenharia de features para o modelo preditivo (TCC, seções 2.2.2/2.2.3).

Monta a matriz de features para uma série (município, produto) específica:
lags do preço (4, 8 e 12 observações — ver nota sobre cobertura irregular
abaixo), janelas móveis, variáveis macro contemporâneas e eventos
geopolíticos codificados, com alvo = preço da semana seguinte (t+1).

IMPORTANTE — lags são por número de observações, não por semanas de
calendário: a cobertura semanal da série da ANP nesta base é irregular
antes de ~2022 (alguns trechos de 2004-2021 têm intervalos de vários meses
sem nenhuma semana pesquisada para um dado município/produto). Reindexar
para uma grade semanal contínua e propagar NaN nas lacunas descartaria a
maior parte do histórico pré-2022, então `preco_lag4` significa "preço há
4 observações atrás" (quase sempre ~4 semanas corridas a partir de 2022,
podendo cobrir um intervalo maior em trechos esparsos mais antigos).
"""

from __future__ import annotations

import sqlite3

import pandas as pd

from src.config import DB_PATH
from src.features.geopolitical_events import codificar_eventos

JANELA_MOVEL = 4
LAGS = (4, 8, 12)


def _calcular_features(precos: pd.DataFrame, macro: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Calcula lags, janela móvel e eventos geopolíticos para todas as
    linhas da série (inclusive a última, que ainda não tem alvo
    conhecido). Devolve o DataFrame com as colunas calculadas e a lista de
    nomes das colunas de feature, usada tanto para montar o treino quanto
    para montar a previsão da próxima semana."""
    df = precos.merge(macro, on="semana", how="left").sort_values("semana").reset_index(drop=True)

    for lag in LAGS:
        df[f"preco_lag{lag}"] = df["preco_medio"].shift(lag)

    df["preco_media_movel_4"] = df["preco_medio"].shift(1).rolling(JANELA_MOVEL).mean()
    df["preco_desvio_movel_4"] = df["preco_medio"].shift(1).rolling(JANELA_MOVEL).std()

    eventos = codificar_eventos(df["semana"])
    df = pd.concat([df, eventos], axis=1)

    colunas_feature = [
        "preco_medio",
        *[f"preco_lag{lag}" for lag in LAGS],
        "preco_media_movel_4",
        "preco_desvio_movel_4",
        "usd_brl",
        "brent_usd_bbl",
        "ipca_variacao_mensal",
        *eventos.columns,
    ]

    return df, colunas_feature


def montar_features(precos: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    """Função pura (sem I/O): recebe a série semanal de um (município,
    produto) já filtrada (colunas `semana`, `preco_medio`, `outlier`) e a
    tabela macro semanal (colunas `semana`, `usd_brl`, `brent_usd_bbl`,
    `ipca_variacao_mensal`), devolve a matriz de features + alvo pronta
    para treino."""
    df, colunas_feature = _calcular_features(precos, macro)

    df["preco_alvo"] = df["preco_medio"].shift(-1)
    alvo_outlier = df["outlier"].shift(-1)

    resultado = df[["semana", *colunas_feature, "preco_alvo"]].copy()

    # Descarta linhas sem lag/janela suficiente (início da série), sem
    # semana seguinte (última linha) e onde o ALVO é um outlier (não
    # queremos treinar o modelo para prever um valor suspeito) — os
    # valores outliers continuam sendo usados como entrada de lag/janela
    # normalmente, só não viram alvo de previsão.
    resultado = resultado[~alvo_outlier.fillna(0).astype(bool)]
    resultado = resultado.dropna().reset_index(drop=True)

    return resultado


def montar_features_previsao(precos: pd.DataFrame, macro: pd.DataFrame) -> pd.Series:
    """Função pura (sem I/O): monta o vetor de features da ÚLTIMA semana
    disponível na série, para prever a semana seguinte a ela (que ainda
    não aconteceu). Diferente de `montar_features`, não exige um alvo
    conhecido nem descarta a última linha."""
    df, colunas_feature = _calcular_features(precos, macro)

    ultima_linha = df[["semana", *colunas_feature]].iloc[-1]

    if ultima_linha[colunas_feature].isna().any():
        faltando = ultima_linha[colunas_feature][ultima_linha[colunas_feature].isna()].index.tolist()
        raise ValueError(
            f"Série curta demais para calcular todas as features na última semana "
            f"({ultima_linha['semana'].date()}); faltando: {faltando}."
        )

    return ultima_linha


def _carregar_series(id_municipio: str, produto: str, db_path) -> tuple[pd.DataFrame, pd.DataFrame]:
    with sqlite3.connect(db_path) as conn:
        precos = pd.read_sql(
            "SELECT semana, preco_medio, outlier "
            "FROM precos_semanais WHERE id_municipio = ? AND produto = ? "
            "ORDER BY semana",
            conn,
            params=(id_municipio, produto),
        )
        macro = pd.read_sql("SELECT * FROM macro_semanal ORDER BY semana", conn)

    if precos.empty:
        raise ValueError(f"Nenhum dado em precos_semanais para município={id_municipio!r}, produto={produto!r}.")

    precos["semana"] = pd.to_datetime(precos["semana"])
    macro["semana"] = pd.to_datetime(macro["semana"])
    return precos, macro


def build_features(id_municipio: str, produto: str, db_path=DB_PATH) -> pd.DataFrame:
    """Lê a série (município, produto) e a tabela macro do SQLite e monta
    a matriz de treino (ver `montar_features`)."""
    precos, macro = _carregar_series(id_municipio, produto, db_path)
    return montar_features(precos, macro)


def build_features_previsao(id_municipio: str, produto: str, db_path=DB_PATH) -> pd.Series:
    """Lê a série (município, produto) e a tabela macro do SQLite e monta
    o vetor de previsão da próxima semana (ver `montar_features_previsao`)."""
    precos, macro = _carregar_series(id_municipio, produto, db_path)
    return montar_features_previsao(precos, macro)
