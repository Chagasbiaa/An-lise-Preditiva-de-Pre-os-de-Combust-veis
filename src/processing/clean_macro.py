"""Compatibilização temporal das séries macroeconômicas (Objetivo B).

Câmbio (USD/BRL) e Brent, diários, são agregados para semanal por média.
O IPCA, mensal, é reamostrado para a grade semanal por reindexação seguida
de preenchimento para frente (forward fill) — o valor mensal mais recente
já divulgado é replicado até que um novo seja publicado, evitando atribuir
a semanas passadas um índice que ainda não existia (TCC, seção 2.2.2).

Mesmo alinhamento de semana (domingo a sábado, rotulada pelo domingo) usado
em src/processing/clean_anp.py, para que o join por `semana` entre preços e
variáveis macro seja direto.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.config import DATA_PROCESSED_DIR, DATA_RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ARQUIVO_CAMBIO = DATA_RAW_DIR / "cambio.parquet"
ARQUIVO_BRENT = DATA_RAW_DIR / "brent.parquet"
ARQUIVO_IPCA = DATA_RAW_DIR / "ipca.parquet"
ARQUIVO_SAIDA = DATA_PROCESSED_DIR / "macro_semanal.parquet"


def _semana_anp(data: pd.Series) -> pd.Series:
    """Semana ANP: domingo a sábado, rotulada pelo domingo inicial."""
    return data.dt.to_period("W-SAT").dt.start_time


def _media_semanal(df: pd.DataFrame, coluna_data: str, coluna_valor: str) -> pd.Series:
    semana = _semana_anp(df[coluna_data])
    return df.groupby(semana)[coluna_valor].mean()


def clean(cambio: pd.DataFrame, brent: pd.DataFrame, ipca: pd.DataFrame) -> pd.DataFrame:
    """Combina as três séries brutas em uma tabela macro semanal única.
    Função pura (sem I/O) — recebe e devolve DataFrames, para facilitar
    testes."""
    cambio_semanal = _media_semanal(cambio, "data", "usd_brl")
    brent_semanal = _media_semanal(brent, "data", "brent_usd_bbl")

    inicio = min(cambio_semanal.index.min(), brent_semanal.index.min())
    fim = max(cambio_semanal.index.max(), brent_semanal.index.max())
    grade_semanal = pd.date_range(start=inicio, end=fim, freq="W-SUN")

    macro = pd.DataFrame(index=grade_semanal)
    macro.index.name = "semana"
    macro["usd_brl"] = cambio_semanal.reindex(grade_semanal)
    macro["brent_usd_bbl"] = brent_semanal.reindex(grade_semanal)

    ipca_ordenado = ipca.sort_values("data").set_index("data")["ipca_variacao_mensal"]
    macro["ipca_variacao_mensal"] = ipca_ordenado.reindex(grade_semanal, method="ffill")

    return macro.reset_index()


def run(force: bool = False) -> pd.DataFrame:
    """Lê os três parquets brutos, combina e salva em ARQUIVO_SAIDA."""
    if ARQUIVO_SAIDA.exists() and not force:
        logger.info("%s já existe — pulando. Use --force para refazer.", ARQUIVO_SAIDA)
        return pd.read_parquet(ARQUIVO_SAIDA)

    logger.info("Lendo câmbio, Brent e IPCA de data/raw/...")
    cambio = pd.read_parquet(ARQUIVO_CAMBIO)
    brent = pd.read_parquet(ARQUIVO_BRENT)
    ipca = pd.read_parquet(ARQUIVO_IPCA)

    macro = clean(cambio, brent, ipca)

    macro.to_parquet(ARQUIVO_SAIDA, index=False)
    n_ipca_nulo = int(macro["ipca_variacao_mensal"].isna().sum())
    logger.info(
        "%d semanas (%s a %s), %d sem IPCA (antes da 1ª publicação) -> %s",
        len(macro), macro["semana"].min().date(), macro["semana"].max().date(),
        n_ipca_nulo, ARQUIVO_SAIDA,
    )
    return macro


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Refaz mesmo se o arquivo de saída já existir.")
    args = parser.parse_args()
    run(force=args.force)


if __name__ == "__main__":
    main()
