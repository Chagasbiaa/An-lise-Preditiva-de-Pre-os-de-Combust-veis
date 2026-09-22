"""Coleta da série histórica de preços de combustíveis da ANP via Base dos Dados.

Fonte (TCC, seção 2.2.1): Base dos Dados, tabela BigQuery
`basedosdados.br_anp_precos_combustiveis.microdados`, particionada por
`ano, id_municipio, sigla_uf`. Cobertura gratuita confirmada: 2004-05-10 até
2026-08-08 (a fatia mais recente é exclusiva do plano pago BD Pro), por isso
o corte padrão deste script (DATA_FINAL_GRATUITA) evita qualquer consulta
paga.

Requer um projeto Google Cloud configurado em BASEDOSDADOS_PROJECT_ID (ver
README.md) — sem isso, a consulta ao BigQuery falha na autenticação.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from src.config import BASEDOSDADOS_PROJECT_ID, DATA_RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

TABELA = "basedosdados.br_anp_precos_combustiveis.microdados"
TABELA_MUNICIPIO = "basedosdados.br_bd_diretorios_brasil.municipio"
DATA_INICIAL = "2004-01-01"
DATA_FINAL_GRATUITA = "2026-08-08"
ARQUIVO_SAIDA = DATA_RAW_DIR / "anp_precos.parquet"

# Colunas confirmadas via --probe-schema em 21/09/2026 (a tabela não tem uma
# coluna de nome de município, só o código `id_municipio` — por isso o JOIN
# abaixo com o diretório de municípios da Base dos Dados, que é o padrão
# nesse tipo de tabela do projeto).
#
# A tabela crua tem ~16,8 milhões de linhas (uma por posto, por semana) no
# intervalo gratuito — grande demais para um download confiável (o download
# direto quebrava por instabilidade de conexão depois de ~4,7 milhões de
# linhas). Como a variável-alvo do modelo é o "preço médio semanal do
# combustível" (TCC, seção 2.2.3), agregamos por semana + município +
# produto direto no BigQuery: cai para ~2 milhões de linhas e já entrega o
# dado no formato que o restante do pipeline (features, modelo) precisa.
QUERY = f"""
SELECT
    m.data_coleta,
    m.ano,
    m.sigla_uf,
    m.id_municipio,
    ANY_VALUE(d.nome) AS municipio,
    m.produto,
    m.unidade_medida,
    AVG(m.preco_venda) AS preco_venda_medio,
    MIN(m.preco_venda) AS preco_venda_min,
    MAX(m.preco_venda) AS preco_venda_max,
    COUNT(*) AS n_postos_pesquisados
FROM `{TABELA}` AS m
LEFT JOIN `{TABELA_MUNICIPIO}` AS d
    ON m.id_municipio = d.id_municipio
WHERE m.data_coleta BETWEEN '{DATA_INICIAL}' AND '{DATA_FINAL_GRATUITA}'
GROUP BY m.data_coleta, m.ano, m.sigla_uf, m.id_municipio, m.produto, m.unidade_medida
"""

SCHEMA_QUERY = f"""
SELECT column_name, data_type
FROM `basedosdados.br_anp_precos_combustiveis.INFORMATION_SCHEMA.COLUMNS`
WHERE table_name = 'microdados'
ORDER BY ordinal_position
"""


def _require_project_id() -> str:
    if not BASEDOSDADOS_PROJECT_ID:
        raise RuntimeError(
            "BASEDOSDADOS_PROJECT_ID não configurado. Preencha o .env "
            "(veja .env.example e o passo a passo no README.md)."
        )
    return BASEDOSDADOS_PROJECT_ID


def _normalizar_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Converte os tipos específicos do BigQuery (db_dtypes) para tipos
    padrão do pandas, para que o parquet resultante seja lido por qualquer
    ambiente sem depender do pacote db-dtypes."""
    df = df.copy()
    df["data_coleta"] = pd.to_datetime(df["data_coleta"].astype(str))
    df["ano"] = df["ano"].astype("int32")
    df["n_postos_pesquisados"] = df["n_postos_pesquisados"].astype("int32")
    return df


def probe_schema() -> pd.DataFrame:
    """Consulta barata (metadados) para conferir os nomes reais das colunas."""
    import basedosdados as bd

    project_id = _require_project_id()
    df = bd.read_sql(SCHEMA_QUERY, billing_project_id=project_id)
    logger.info("Colunas de %s:\n%s", TABELA, df.to_string(index=False))
    return df


def collect(force: bool = False) -> pd.DataFrame:
    if ARQUIVO_SAIDA.exists() and not force:
        logger.info("%s já existe — pulando. Use --force para refazer.", ARQUIVO_SAIDA)
        return pd.read_parquet(ARQUIVO_SAIDA)

    import basedosdados as bd

    project_id = _require_project_id()
    logger.info("Consultando %s (%s a %s)...", TABELA, DATA_INICIAL, DATA_FINAL_GRATUITA)
    df = bd.read_sql(QUERY, billing_project_id=project_id)
    df = _normalizar_dtypes(df)

    df.to_parquet(ARQUIVO_SAIDA, index=False)
    logger.info("%d linhas coletadas -> %s", len(df), ARQUIVO_SAIDA)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Refaz o download mesmo se já existir localmente.")
    parser.add_argument(
        "--probe-schema",
        action="store_true",
        help="Só lista as colunas reais da tabela (consulta de metadados, sem custo de leitura de dados).",
    )
    args = parser.parse_args()

    if args.probe_schema:
        probe_schema()
    else:
        collect(force=args.force)


if __name__ == "__main__":
    main()
