"""Configuração central do projeto: caminhos e credenciais."""

from pathlib import Path

from dotenv import load_dotenv
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DB_PATH = PROJECT_ROOT / "data" / "fuel_prices.db"
MODELS_DIR = PROJECT_ROOT / "models"

DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

load_dotenv(PROJECT_ROOT / ".env")

BASEDOSDADOS_PROJECT_ID = os.getenv("BASEDOSDADOS_PROJECT_ID")
EIA_API_KEY = os.getenv("EIA_API_KEY")


def caminho_modelo(id_municipio: str, produto: str) -> tuple[Path, Path]:
    """Caminhos (joblib, json) do modelo Random Forest treinado para um
    (município, produto) — convenção única usada pelo treino
    (src/models/train_random_forest.py), pelo dashboard e pela
    explicabilidade (src/models/explain.py)."""
    base = MODELS_DIR / f"rf_{id_municipio}_{produto}"
    return base.with_suffix(".joblib"), base.with_suffix(".json")
