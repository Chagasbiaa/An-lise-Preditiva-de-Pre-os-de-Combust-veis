"""Monta o banco relacional SQLite a partir dos dados processados (Objetivo C).

Schema pensado para os filtros do dashboard (Objetivo D — "filtros por
cidade e tipo de combustível"): `dim_municipio` evita repetir nome/UF em
cada linha de preço, `precos_semanais` é a tabela fato (uma linha por
semana + município + produto) com índices por município e por produto, e
`macro_semanal` fica separada — pequena (uma linha por semana) e sem
repetição por município/produto. O join entre preços e variáveis macro
acontece na etapa de engenharia de features (Objetivo E), não fica
pré-calculado no banco.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
from pathlib import Path

import pandas as pd

from src.config import DB_PATH
from src.processing import clean_anp, clean_macro

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

TABELAS_DDL = """
CREATE TABLE dim_municipio (
    id_municipio TEXT PRIMARY KEY,
    nome TEXT NOT NULL,
    sigla_uf TEXT NOT NULL
);

CREATE TABLE precos_semanais (
    semana TEXT NOT NULL,
    id_municipio TEXT NOT NULL REFERENCES dim_municipio(id_municipio),
    produto TEXT NOT NULL,
    unidade_medida TEXT NOT NULL,
    preco_medio REAL NOT NULL,
    preco_min REAL NOT NULL,
    preco_max REAL NOT NULL,
    n_postos_pesquisados INTEGER NOT NULL,
    outlier INTEGER NOT NULL,
    PRIMARY KEY (semana, id_municipio, produto)
);

CREATE TABLE macro_semanal (
    semana TEXT PRIMARY KEY,
    usd_brl REAL,
    brent_usd_bbl REAL,
    ipca_variacao_mensal REAL
);
"""

INDICES_DDL = """
CREATE INDEX idx_precos_municipio ON precos_semanais(id_municipio);
CREATE INDEX idx_precos_produto ON precos_semanais(produto);
"""


def build(precos_semanais: pd.DataFrame, macro_semanal: pd.DataFrame, destino: Path) -> None:
    """Cria o schema do zero e carrega os dados no arquivo `destino`. Recebe
    os DataFrames já prontos (em vez de ler arquivos) para facilitar testes
    com um banco temporário."""
    dim_municipio = (
        precos_semanais[["id_municipio", "municipio", "sigla_uf"]]
        .drop_duplicates("id_municipio")
        .rename(columns={"municipio": "nome"})
        .sort_values("id_municipio")
    )

    fato_precos = precos_semanais[
        [
            "semana", "id_municipio", "produto", "unidade_medida",
            "preco_medio", "preco_min", "preco_max",
            "n_postos_pesquisados", "preco_outlier",
        ]
    ].rename(columns={"preco_outlier": "outlier"})
    fato_precos = fato_precos.copy()
    fato_precos["outlier"] = fato_precos["outlier"].astype(int)
    fato_precos["semana"] = pd.to_datetime(fato_precos["semana"]).dt.strftime("%Y-%m-%d")

    macro = macro_semanal.copy()
    macro["semana"] = pd.to_datetime(macro["semana"]).dt.strftime("%Y-%m-%d")

    with sqlite3.connect(destino) as conn:
        conn.executescript(
            "DROP TABLE IF EXISTS precos_semanais;"
            "DROP TABLE IF EXISTS dim_municipio;"
            "DROP TABLE IF EXISTS macro_semanal;"
        )
        conn.executescript(TABELAS_DDL)

        dim_municipio.to_sql("dim_municipio", conn, if_exists="append", index=False)
        fato_precos.to_sql("precos_semanais", conn, if_exists="append", index=False)
        macro.to_sql("macro_semanal", conn, if_exists="append", index=False)

        conn.executescript(INDICES_DDL)
        conn.commit()


def run(force: bool = False) -> None:
    if DB_PATH.exists() and not force:
        logger.info("%s já existe — pulando. Use --force para refazer.", DB_PATH)
        return

    if not clean_anp.ARQUIVO_SAIDA.exists() or not clean_macro.ARQUIVO_SAIDA.exists():
        raise FileNotFoundError(
            "Rode antes 'python -m src.processing.clean_anp' e "
            "'python -m src.processing.clean_macro' (Objetivo B) para gerar "
            "os arquivos em data/processed/."
        )

    logger.info("Lendo dados processados de data/processed/...")
    precos_semanais = pd.read_parquet(clean_anp.ARQUIVO_SAIDA)
    macro_semanal = pd.read_parquet(clean_macro.ARQUIVO_SAIDA)

    build(precos_semanais, macro_semanal, DB_PATH)
    logger.info(
        "Banco criado em %s: %d municípios, %d linhas de preços, %d semanas de macro.",
        DB_PATH,
        precos_semanais["id_municipio"].nunique(),
        len(precos_semanais),
        len(macro_semanal),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Refaz mesmo se o banco já existir.")
    args = parser.parse_args()
    run(force=args.force)


if __name__ == "__main__":
    main()
