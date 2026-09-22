"""Coleta da série diária do preço do petróleo Brent via API da EIA.

Fonte (TCC, seção 2.2.1): U.S. Energy Information Administration (EIA),
rota `petroleum/pri/spt` ("Europe Brent Spot Price FOB, Daily", série
histórica conhecida como RBRTE).

Requer uma API key gratuita em EIA_API_KEY (ver README.md — cadastro grátis,
sem cartão de crédito, em https://www.eia.gov/opendata/register.php).

Observação: a API v2 da EIA não foi testada de ponta a ponta nesta sessão
(sem API key disponível). O nome do facet usado abaixo (`series`, valor
`RBRTE`) é o documentado pela EIA para compatibilidade com IDs de série da
API v1. Se a EIA devolver erro de facet inválido, rode este script com
--probe para listar os facets/frequências reais da rota `petroleum/pri/spt`
(chamada de metadados, sem custo) e ajustar a constante FACET_SERIES abaixo.
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd
import requests

from src.config import DATA_RAW_DIR, EIA_API_KEY

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASE_URL = "https://api.eia.gov/v2/petroleum/pri/spt"
FACET_SERIES = "RBRTE"  # Europe Brent Spot Price FOB, Daily
DATA_INICIAL = "2004-01-01"
ARQUIVO_SAIDA = DATA_RAW_DIR / "brent.parquet"


def _require_api_key() -> str:
    if not EIA_API_KEY:
        raise RuntimeError(
            "EIA_API_KEY não configurada. Preencha o .env (veja .env.example "
            "e o passo a passo no README.md)."
        )
    return EIA_API_KEY


def probe_route() -> dict:
    """Consulta de metadados da rota (facets/frequências disponíveis), sem custo."""
    api_key = _require_api_key()
    resp = requests.get(BASE_URL, params={"api_key": api_key}, timeout=30)
    resp.raise_for_status()
    info = resp.json()
    logger.info("Metadados de %s:\n%s", BASE_URL, info)
    return info


def collect(force: bool = False) -> pd.DataFrame:
    if ARQUIVO_SAIDA.exists() and not force:
        logger.info("%s já existe — pulando. Use --force para refazer.", ARQUIVO_SAIDA)
        return pd.read_parquet(ARQUIVO_SAIDA)

    api_key = _require_api_key()
    logger.info("Buscando Brent (série %s) desde %s...", FACET_SERIES, DATA_INICIAL)

    params = {
        "api_key": api_key,
        "frequency": "daily",
        "data[0]": "value",
        "facets[series][]": FACET_SERIES,
        "start": DATA_INICIAL,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": 5000,
    }

    registros = []
    while True:
        resp = requests.get(f"{BASE_URL}/data/", params=params, timeout=30)
        resp.raise_for_status()
        payload = resp.json()["response"]
        pagina = payload["data"]
        registros.extend(pagina)
        if len(pagina) < params["length"]:
            break
        params["offset"] += params["length"]

    if not registros:
        raise ValueError("EIA API não retornou dados para a série Brent (RBRTE).")

    df = pd.DataFrame(registros)[["period", "value"]].rename(
        columns={"period": "data", "value": "brent_usd_bbl"}
    )
    df["data"] = pd.to_datetime(df["data"])
    df["brent_usd_bbl"] = df["brent_usd_bbl"].astype(float)
    df = df.sort_values("data").reset_index(drop=True)

    df.to_parquet(ARQUIVO_SAIDA, index=False)
    logger.info(
        "%d linhas coletadas, %s a %s -> %s",
        len(df), df["data"].min().date(), df["data"].max().date(), ARQUIVO_SAIDA,
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Refaz o download mesmo se já existir localmente.")
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Só consulta os metadados (facets/frequências) da rota, sem baixar dados.",
    )
    args = parser.parse_args()

    if args.probe:
        probe_route()
    else:
        collect(force=args.force)


if __name__ == "__main__":
    main()
