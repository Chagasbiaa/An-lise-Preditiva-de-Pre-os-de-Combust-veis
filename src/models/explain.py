"""Interpretabilidade do modelo com SHAP (Objetivo G, TCC seção 2.1.9).

SHAP dá duas visões complementares às `feature_importances_` do Random
Forest (redução média de impureza, já usada no Objetivo E): importância
GLOBAL (`shap_global`, média de |valor SHAP| entre várias semanas) e
explicação LOCAL de uma previsão específica (`shap_local`), que respondem
diretamente à pergunta que motiva o TCC: "qual fator — câmbio ou preço do
petróleo — exerce maior influência sobre o preço dos combustíveis ao
consumidor?"
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import shap

from src.config import caminho_modelo
from src.features.build_features import build_features, build_features_previsao

TAMANHO_AMOSTRA_PADRAO = 300


def _resumo_global(matriz_shap: np.ndarray, colunas: list[str]) -> pd.DataFrame:
    """Função pura: matriz de valores SHAP (linhas x features) já
    calculada -> importância média (|SHAP| médio), ordenada decrescente."""
    importancia = np.abs(matriz_shap).mean(axis=0)
    return (
        pd.DataFrame({"feature": colunas, "importancia_media_abs": importancia})
        .sort_values("importancia_media_abs", ascending=False)
        .reset_index(drop=True)
    )


def _contribuicoes_locais(valores_shap: np.ndarray, colunas: list[str]) -> pd.DataFrame:
    """Função pura: valores SHAP de uma única linha -> contribuição de
    cada feature, ordenada por magnitude (maior impacto primeiro)."""
    valores_shap = np.ravel(valores_shap)
    df = pd.DataFrame({"feature": colunas, "contribuicao": valores_shap})
    return df.reindex(df["contribuicao"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def _carregar_modelo(id_municipio: str, produto: str) -> dict:
    caminho_joblib, _ = caminho_modelo(id_municipio, produto)
    if not caminho_joblib.exists():
        raise FileNotFoundError(
            f"Nenhum modelo treinado para município={id_municipio!r}, produto={produto!r} "
            f"({caminho_joblib}). Treine primeiro (src.models.train_random_forest)."
        )
    return joblib.load(caminho_joblib)


def shap_global(id_municipio: str, produto: str, tamanho_amostra: int = TAMANHO_AMOSTRA_PADRAO) -> pd.DataFrame:
    """Importância global das features via SHAP, calculada sobre uma
    amostra (reprodutível) dos dados de treino — uma amostra é suficiente
    para estabilizar a média e roda em segundos em vez de minutos."""
    modelo_salvo = _carregar_modelo(id_municipio, produto)
    colunas_feature = modelo_salvo["colunas_feature"]

    dados = build_features(id_municipio, produto)
    amostra = dados[colunas_feature].sample(min(tamanho_amostra, len(dados)), random_state=42)

    explainer = shap.TreeExplainer(modelo_salvo["modelo"])
    matriz_shap = explainer.shap_values(amostra)

    return _resumo_global(matriz_shap, colunas_feature)


def shap_local(id_municipio: str, produto: str) -> dict:
    """Explicação SHAP da previsão da PRÓXIMA semana (mesma linha usada
    pela seção de previsão do dashboard) — soma das contribuições +
    valor_base reconstrói exatamente a previsão do modelo (propriedade de
    aditividade do SHAP)."""
    modelo_salvo = _carregar_modelo(id_municipio, produto)
    colunas_feature = modelo_salvo["colunas_feature"]

    ultima_linha = build_features_previsao(id_municipio, produto)
    entrada = ultima_linha[colunas_feature].to_frame().T

    explainer = shap.TreeExplainer(modelo_salvo["modelo"])
    valores_shap = explainer.shap_values(entrada)
    valor_base = float(np.ravel(explainer.expected_value)[0])

    contribuicoes = _contribuicoes_locais(valores_shap, colunas_feature)
    previsao = valor_base + contribuicoes["contribuicao"].sum()

    return {
        "contribuicoes": contribuicoes,
        "valor_base": valor_base,
        "previsao": float(previsao),
        "semana": ultima_linha["semana"],
    }
