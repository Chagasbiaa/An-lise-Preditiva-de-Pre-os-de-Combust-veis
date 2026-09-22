"""Coleta de séries macroeconômicas públicas do Banco Central do Brasil (BCB).

O TCC cita o Ipeadata como fonte das séries complementares de câmbio e IPCA
(seção 2.2.1). O Ipeadata, para essas duas séries específicas, republica os
dados do Sistema Gerenciador de Séries Temporais (SGS) do próprio Banco
Central. Optou-se por consultar o SGS diretamente: é a fonte primária, tem
API pública em JSON sem necessidade de cadastro/API key, e é mais estável.

Séries utilizadas (código SGS):
    1   - Taxa de câmbio - Livre - Dólar americano (venda) - diário
    433 - IPCA - variação mensal (%)
"""

from __future__ import annotations

import argparse
import datetime as dt
import logging

import pandas as pd
import requests

from src.config import DATA_RAW_DIR

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SGS_URL = "https://api.bcb.gov.br/dados/serie/bcdata.sgs.{codigo}/dados"

# A API do BCB rejeita (406) requisições sem um User-Agent "de navegador" —
# o padrão usado pela biblioteca requests não é aceito.
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; tg2-fuel-prices/1.0)"}

# Séries diárias (como o câmbio) recusam (406) janelas de consulta maiores
# que 10 anos ("O sistema aceita uma janela de consulta de, no máximo, 10
# anos em séries de periodicidade diária"). Buscamos em blocos menores que
# isso e concatenamos — funciona também para séries mensais como o IPCA.
TAMANHO_JANELA = dt.timedelta(days=3650)

SERIES = {
    "cambio": {"codigo": 1, "coluna_valor": "usd_brl", "arquivo": "cambio.parquet"},
    "ipca": {"codigo": 433, "coluna_valor": "ipca_variacao_mensal", "arquivo": "ipca.parquet"},
}


def _fetch_sgs_window(codigo: int, data_inicial: str, data_final: str) -> list[dict]:
    resp = requests.get(
        SGS_URL.format(codigo=codigo),
        params={"formato": "json", "dataInicial": data_inicial, "dataFinal": data_final},
        headers=HEADERS,
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def _fetch_sgs_series(codigo: int, data_inicial: str = "01/01/2004", data_final: str | None = None) -> pd.DataFrame:
    """Busca uma série completa do SGS/BCB entre `data_inicial` e `data_final`
    (dd/mm/aaaa), paginando em janelas de até TAMANHO_JANELA dias."""
    inicio = dt.datetime.strptime(data_inicial, "%d/%m/%Y").date()
    fim = dt.datetime.strptime(data_final, "%d/%m/%Y").date() if data_final else dt.date.today()

    registros: list[dict] = []
    janela_inicio = inicio
    while janela_inicio <= fim:
        janela_fim = min(janela_inicio + TAMANHO_JANELA, fim)
        registros.extend(
            _fetch_sgs_window(codigo, janela_inicio.strftime("%d/%m/%Y"), janela_fim.strftime("%d/%m/%Y"))
        )
        janela_inicio = janela_fim + dt.timedelta(days=1)

    if not registros:
        raise ValueError(f"SGS série {codigo} retornou vazio")

    df = pd.DataFrame(registros)
    df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
    df["valor"] = df["valor"].astype(float)
    return df.drop_duplicates(subset="data").sort_values("data").reset_index(drop=True)


def collect_series(nome: str, force: bool = False) -> pd.DataFrame:
    """Coleta uma das séries definidas em SERIES e salva em data/raw/<arquivo>."""
    spec = SERIES[nome]
    destino = DATA_RAW_DIR / spec["arquivo"]

    if destino.exists() and not force:
        logger.info("%s já existe (%s) — pulando. Use --force para refazer.", nome, destino)
        return pd.read_parquet(destino)

    logger.info("Buscando série BCB/SGS %s (código %s)...", nome, spec["codigo"])
    df = _fetch_sgs_series(spec["codigo"])
    df = df.rename(columns={"valor": spec["coluna_valor"]})

    df.to_parquet(destino, index=False)
    logger.info(
        "%s: %d linhas, %s a %s -> %s",
        nome, len(df), df["data"].min().date(), df["data"].max().date(), destino,
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Refaz o download mesmo se já existir localmente.")
    args = parser.parse_args()

    for nome in SERIES:
        collect_series(nome, force=args.force)


if __name__ == "__main__":
    main()
