"""Limpeza e compatibilização temporal dos preços da ANP (Objetivo B).

Fonte: data/raw/anp_precos.parquet (uma linha por data_coleta exata, já
agregada por posto no BigQuery — ver src/data/collect_anp.py). As visitas
aos postos se espalham de domingo a sábado dentro de cada semana da
pesquisa ANP, então este script reagrega para uma linha por
(semana, município, produto), como pede a metodologia do TCC (seção 2.2.2:
frequência semanal como base de referência).
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ARQUIVO_ENTRADA = DATA_RAW_DIR / "anp_precos.parquet"
ARQUIVO_SAIDA = DATA_PROCESSED_DIR / "anp_precos_semanal.parquet"

# A coluna `unidade_medida` bruta tem nulos (~3.6 mil linhas), o que causa
# duplicatas de (semana, município, produto) na agregação. A unidade é 1:1
# com o produto, então é mais confiável derivá-la por este mapeamento do
# que confiar na coluna original.
UNIDADE_POR_PRODUTO = {
    "Diesel": "R$/litro",
    "Diesel S10": "R$/litro",
    "Diesel S50": "R$/litro",
    "Etanol": "R$/litro",
    "Gasolina": "R$/litro",
    "Gasolina Aditivada": "R$/litro",
    "Glp": "R$/13kg",
    "Gnv": "R$/m3",
}

CHAVE_SEMANAL = ["semana", "id_municipio", "sigla_uf", "municipio", "produto", "unidade_medida"]


def _semana_anp(data_coleta: pd.Series) -> pd.Series:
    """Semana ANP: domingo a sábado, rotulada pelo domingo inicial."""
    return data_coleta.dt.to_period("W-SAT").dt.start_time


def _marcar_outliers(precos: pd.DataFrame) -> pd.Series:
    """Marca outliers de preco_medio por produto pela regra de IQR (1,5x)."""
    q1 = precos.groupby("produto")["preco_medio"].transform(lambda s: s.quantile(0.25))
    q3 = precos.groupby("produto")["preco_medio"].transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    limite_inferior = q1 - 1.5 * iqr
    limite_superior = q3 + 1.5 * iqr
    return (precos["preco_medio"] < limite_inferior) | (precos["preco_medio"] > limite_superior)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Limpa e reagrega os preços da ANP para granularidade semanal. Função
    pura (sem I/O) — recebe e devolve DataFrames, para facilitar testes."""
    antes = len(df)
    df = df.dropna(subset=["id_municipio", "municipio"]).copy()
    if antes - len(df):
        logger.info("Descartada(s) %d linha(s) com município nulo.", antes - len(df))

    df["unidade_medida"] = df["produto"].map(UNIDADE_POR_PRODUTO)
    if df["unidade_medida"].isna().any():
        produtos_sem_mapa = sorted(df.loc[df["unidade_medida"].isna(), "produto"].unique())
        raise ValueError(f"Produto(s) sem unidade mapeada em UNIDADE_POR_PRODUTO: {produtos_sem_mapa}")

    df["semana"] = _semana_anp(df["data_coleta"])

    logger.info("Reagregando %d linhas (data exata) para granularidade semanal...", len(df))
    df["_preco_x_peso"] = df["preco_venda_medio"] * df["n_postos_pesquisados"]
    agregado = (
        df.groupby(CHAVE_SEMANAL)
        .agg(
            _soma_preco_peso=("_preco_x_peso", "sum"),
            n_postos_pesquisados=("n_postos_pesquisados", "sum"),
            preco_min=("preco_venda_min", "min"),
            preco_max=("preco_venda_max", "max"),
        )
        .reset_index()
    )
    agregado["preco_medio"] = agregado["_soma_preco_peso"] / agregado["n_postos_pesquisados"]
    agregado = agregado.drop(columns="_soma_preco_peso")

    n_duplicatas = int(agregado.duplicated(subset=["semana", "id_municipio", "produto"]).sum())
    if n_duplicatas:
        raise AssertionError(f"{n_duplicatas} duplicata(s) de (semana, município, produto) após a agregação.")

    agregado["preco_outlier"] = _marcar_outliers(agregado)
    return agregado


def run(force: bool = False) -> pd.DataFrame:
    """Lê data/raw/anp_precos.parquet, limpa e salva em ARQUIVO_SAIDA."""
    if ARQUIVO_SAIDA.exists() and not force:
        logger.info("%s já existe — pulando. Use --force para refazer.", ARQUIVO_SAIDA)
        return pd.read_parquet(ARQUIVO_SAIDA)

    logger.info("Lendo %s...", ARQUIVO_ENTRADA)
    bruto = pd.read_parquet(ARQUIVO_ENTRADA)
    agregado = clean(bruto)

    agregado.to_parquet(ARQUIVO_SAIDA, index=False)
    logger.info(
        "%d linhas semanais (%s a %s), %d marcada(s) como outlier -> %s",
        len(agregado),
        agregado["semana"].min().date(),
        agregado["semana"].max().date(),
        int(agregado["preco_outlier"].sum()),
        ARQUIVO_SAIDA,
    )
    return agregado


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Refaz mesmo se o arquivo de saída já existir.")
    args = parser.parse_args()
    run(force=args.force)


if __name__ == "__main__":
    main()
