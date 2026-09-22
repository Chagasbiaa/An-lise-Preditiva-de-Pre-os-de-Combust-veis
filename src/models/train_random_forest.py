"""Treino e avaliação do modelo Random Forest (Objetivo E, TCC seção 2.2.3).

Validação cruzada com divisão temporal (TimeSeriesSplit) sobre o
treino+validação, mais um holdout final (as últimas N semanas, nunca
usadas em treino/CV) para simular uso em produção. O modelo "implantado"
salvo em disco é reajustado em 100% dos dados disponíveis.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

from src.config import caminho_modelo
from src.features.build_features import build_features

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

COLUNA_ALVO = "preco_alvo"
COLUNAS_NAO_FEATURE = {"semana", COLUNA_ALVO}


def _calcular_metricas(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mean_absolute_percentage_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _media_desvio(metricas_por_fold: list[dict]) -> dict:
    chaves = metricas_por_fold[0].keys()
    return {
        chave: {
            "media": float(np.mean([m[chave] for m in metricas_por_fold])),
            "desvio_padrao": float(np.std([m[chave] for m in metricas_por_fold])),
        }
        for chave in chaves
    }


def train_and_evaluate(
    id_municipio: str,
    produto: str,
    n_splits: int = 5,
    semanas_holdout: int = 52,
    random_state: int = 42,
) -> dict:
    dados = build_features(id_municipio, produto)
    colunas_feature = [c for c in dados.columns if c not in COLUNAS_NAO_FEATURE]

    if len(dados) <= semanas_holdout + n_splits:
        raise ValueError(
            f"Dados insuficientes para município={id_municipio!r}, produto={produto!r}: "
            f"{len(dados)} linhas para holdout={semanas_holdout} + {n_splits} folds de CV."
        )

    treino_val = dados.iloc[: -semanas_holdout].reset_index(drop=True)
    holdout = dados.iloc[-semanas_holdout:].reset_index(drop=True)

    X_treino_val = treino_val[colunas_feature]
    y_treino_val = treino_val[COLUNA_ALVO]

    logger.info(
        "Município=%s produto=%s: %d linhas (treino+validação) + %d de holdout final.",
        id_municipio, produto, len(treino_val), len(holdout),
    )

    tscv = TimeSeriesSplit(n_splits=n_splits)
    metricas_por_fold = []
    for fold, (idx_treino, idx_teste) in enumerate(tscv.split(X_treino_val), start=1):
        modelo_fold = RandomForestRegressor(random_state=random_state)
        modelo_fold.fit(X_treino_val.iloc[idx_treino], y_treino_val.iloc[idx_treino])
        previsoes = modelo_fold.predict(X_treino_val.iloc[idx_teste])
        metricas_fold = _calcular_metricas(y_treino_val.iloc[idx_teste], previsoes)
        metricas_por_fold.append(metricas_fold)
        logger.info("  fold %d/%d: %s", fold, n_splits, metricas_fold)

    metricas_cv = _media_desvio(metricas_por_fold)

    # Modelo "de produção": treinado só com treino+validação, avaliado no
    # holdout final nunca visto — é o número que vale reportar no TCC.
    modelo_producao = RandomForestRegressor(random_state=random_state)
    modelo_producao.fit(X_treino_val, y_treino_val)
    previsoes_holdout = modelo_producao.predict(holdout[colunas_feature])
    metricas_holdout = _calcular_metricas(holdout[COLUNA_ALVO], previsoes_holdout)
    logger.info("Holdout final (%d semanas nunca vistas): %s", len(holdout), metricas_holdout)

    # Modelo "implantado": reajustado em 100% dos dados disponíveis, para
    # uso pelo dashboard/what-if (Objetivos D/F) — não usado para reportar
    # métricas, só para servir previsões daqui pra frente.
    modelo_implantado = RandomForestRegressor(random_state=random_state)
    modelo_implantado.fit(dados[colunas_feature], dados[COLUNA_ALVO])

    importancias = dict(
        sorted(
            zip(colunas_feature, modelo_implantado.feature_importances_.tolist()),
            key=lambda item: item[1],
            reverse=True,
        )
    )

    metadados = {
        "id_municipio": id_municipio,
        "produto": produto,
        "treinado_em": datetime.now(timezone.utc).isoformat(),
        "n_linhas_total": len(dados),
        "n_linhas_treino_validacao": len(treino_val),
        "n_linhas_holdout": len(holdout),
        "n_splits_cv": n_splits,
        "colunas_feature": colunas_feature,
        "metricas_cv": metricas_cv,
        "metricas_holdout": metricas_holdout,
        "feature_importances": importancias,
    }

    caminho_joblib, caminho_json = caminho_modelo(id_municipio, produto)
    joblib.dump({"modelo": modelo_implantado, "colunas_feature": colunas_feature}, caminho_joblib)
    caminho_json.write_text(json.dumps(metadados, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Modelo salvo em %s (metadados em %s).", caminho_joblib, caminho_json)

    return {
        "modelo": modelo_implantado,
        "caminho_modelo": caminho_joblib,
        "caminho_metadados": caminho_json,
        **metadados,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--municipio", required=True, help="id_municipio (código IBGE) da série a treinar.")
    parser.add_argument("--produto", required=True, help="Produto (ex.: Gasolina, Diesel, Etanol...).")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--semanas-holdout", type=int, default=52)
    args = parser.parse_args()

    resultado = train_and_evaluate(
        args.municipio, args.produto, n_splits=args.n_splits, semanas_holdout=args.semanas_holdout
    )

    print()
    print(f"=== {args.produto} em município {args.municipio} ===")
    print(f"{'métrica':<8} {'CV (média ± dp)':<22} {'holdout final':<14}")
    for chave in resultado["metricas_holdout"]:
        cv = resultado["metricas_cv"][chave]
        print(
            f"{chave:<8} {cv['media']:.4f} ± {cv['desvio_padrao']:.4f}      "
            f"{resultado['metricas_holdout'][chave]:.4f}"
        )
    print()
    print("Top 5 features mais importantes:")
    for nome, importancia in list(resultado["feature_importances"].items())[:5]:
        print(f"  {nome:<40} {importancia:.4f}")


if __name__ == "__main__":
    main()
