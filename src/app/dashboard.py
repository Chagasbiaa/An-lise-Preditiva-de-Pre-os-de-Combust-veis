"""Dashboard interativo de preços de combustíveis (Objetivos D e F do TCC).

Filtros por cidade e tipo de combustível, visualizações do comportamento
histórico dos preços, uma previsão simples para a semana seguinte quando
já existe modelo treinado (Objetivo E) e uma simulação de cenários
what-if (Objetivo F): o usuário ajusta câmbio/Brent/IPCA e um toggle de
"novo choque geopolítico" e vê o impacto estimado em tempo real.
"""

from __future__ import annotations

import sys
from pathlib import Path

# O Streamlit Community Cloud executa este arquivo sem colocar a raiz do
# repositório no sys.path (diferente de rodar localmente com
# `python -m streamlit run ...`), o que quebraria os imports `from src...`
# abaixo. Garantimos a raiz no sys.path manualmente antes deles.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import joblib
import json
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.app.data_access import PRODUTOS, historico_precos, listar_municipios, listar_ufs, precos_atuais_por_produto
from src.config import caminho_modelo
from src.features.build_features import build_features, build_features_previsao
from src.models.explain import shap_global, shap_local
from src.models.train_random_forest import train_and_evaluate

st.set_page_config(page_title="Preços de Combustíveis", layout="wide")


@st.cache_resource
def _carregar_modelo(caminho_joblib_str: str):
    return joblib.load(caminho_joblib_str)


def _grafico_historico(df: pd.DataFrame, produto: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["semana"], y=df["preco_max"], line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df["semana"], y=df["preco_min"], line=dict(width=0), fill="tonexty",
        fillcolor="rgba(31,119,180,0.15)", name="Faixa mín-máx", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df["semana"], y=df["preco_medio"], mode="lines", name="Preço médio",
        line=dict(color="#1f77b4", width=2),
    ))
    outliers = df[df["outlier"]]
    if not outliers.empty:
        fig.add_trace(go.Scatter(
            x=outliers["semana"], y=outliers["preco_medio"], mode="markers", name="Semana marcada como outlier",
            marker=dict(color="crimson", size=8, symbol="x"),
        ))
    fig.update_layout(
        title=f"Histórico semanal — {produto}",
        xaxis_title="Semana", yaxis_title="R$", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _secao_kpis(df: pd.DataFrame) -> None:
    ultima = df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Preço na última semana", f"R$ {ultima['preco_medio']:.3f}", help=f"Semana de {ultima['semana'].date()}")

    if len(df) > 4:
        variacao_4s = ultima["preco_medio"] / df.iloc[-5]["preco_medio"] - 1
        col2.metric("Variação vs. 4 semanas atrás", f"{variacao_4s:+.1%}")
    else:
        col2.metric("Variação vs. 4 semanas atrás", "N/D")

    if len(df) > 52:
        variacao_52s = ultima["preco_medio"] / df.iloc[-53]["preco_medio"] - 1
        col3.metric("Variação vs. 52 semanas atrás", f"{variacao_52s:+.1%}")
    else:
        col3.metric("Variação vs. 52 semanas atrás", "N/D")

    col4.metric("Postos pesquisados (última semana)", int(ultima["n_postos_pesquisados"]))


def _secao_comparativo(id_municipio: str, nome_municipio: str) -> None:
    comparativo = precos_atuais_por_produto(id_municipio)
    if comparativo.empty:
        return
    st.subheader(f"Preço mais recente por combustível em {nome_municipio}")
    exibicao = comparativo.copy()
    exibicao["semana"] = exibicao["semana"].dt.date
    exibicao["preco_medio"] = exibicao["preco_medio"].round(3)
    exibicao = exibicao.rename(
        columns={"produto": "Produto", "semana": "Semana", "preco_medio": "Preço médio (R$)"}
    ).sort_values("Produto")
    st.dataframe(exibicao, hide_index=True, width="stretch")


def _secao_previsao(id_municipio: str, produto: str) -> dict | None:
    """Renderiza a seção de previsão e devolve o estado usado (modelo,
    metadados, última linha de features, previsão base) para a seção de
    what-if reaproveitar sem recarregar/recalcular nada — ou None quando
    não há modelo/dados suficientes (o what-if fica oculto nesse caso)."""
    st.subheader("Previsão para a próxima semana")
    caminho_joblib, caminho_json = caminho_modelo(id_municipio, produto)

    if not caminho_joblib.exists():
        st.info("Nenhum modelo treinado ainda para essa cidade + combustível.")
        if st.button("Treinar modelo agora", key=f"treinar_{id_municipio}_{produto}"):
            with st.spinner("Treinando modelo (validação temporal + holdout)... leva de 20 a 40 segundos."):
                train_and_evaluate(id_municipio, produto)
            st.rerun()
        return None

    modelo_salvo = _carregar_modelo(str(caminho_joblib))
    metadados = json.loads(caminho_json.read_text(encoding="utf-8"))

    try:
        ultima_linha = build_features_previsao(id_municipio, produto)
    except ValueError as erro:
        st.warning(str(erro))
        return None

    colunas_feature = modelo_salvo["colunas_feature"]
    entrada = pd.DataFrame([ultima_linha[colunas_feature]])
    previsao = modelo_salvo["modelo"].predict(entrada)[0]
    # pd.Timedelta(...) dispara um DeprecationWarning nesta combinação de
    # versões (pandas 2.3.3 + numpy 2.5.3); pd.DateOffset produz o mesmo
    # resultado para uma soma de semanas sem esse problema.
    semana_prevista = (pd.Timestamp(ultima_linha["semana"]) + pd.DateOffset(weeks=1)).date()

    col1, col2 = st.columns([1, 2])
    col1.metric(
        f"Previsão para a semana de {semana_prevista}",
        f"R$ {previsao:.3f}",
        help=f"Com base na última semana disponível ({ultima_linha['semana'].date()}).",
    )

    with col2:
        st.caption("Desempenho do modelo no holdout final (semanas nunca usadas em treino)")
        holdout = metadados["metricas_holdout"]
        st.write(
            f"MAE: R$ {holdout['mae']:.3f} · RMSE: R$ {holdout['rmse']:.3f} · "
            f"MAPE: {holdout['mape']:.2%} · R²: {holdout['r2']:.3f}"
        )
        st.caption(f"Treinado em {metadados['treinado_em'][:10]}, {metadados['n_linhas_total']} semanas de dados.")

    with st.expander("Features mais importantes para este modelo"):
        top5 = list(metadados["feature_importances"].items())[:5]
        st.dataframe(
            pd.DataFrame(top5, columns=["Feature", "Importância"]),
            hide_index=True, width="stretch",
        )

    return {
        "modelo_salvo": modelo_salvo,
        "metadados": metadados,
        "ultima_linha": ultima_linha,
        "previsao_base": float(previsao),
        "semana_prevista": semana_prevista,
    }


GRUPO_MACRO = {"usd_brl", "brent_usd_bbl", "ipca_variacao_mensal"}
CORES_GRUPO = {"Preço/histórico": "#1f77b4", "Macro": "#ff7f0e", "Geopolítico": "#d62728"}


def _grupo_feature(nome: str) -> str:
    if nome.startswith("evento_"):
        return "Geopolítico"
    if nome in GRUPO_MACRO:
        return "Macro"
    return "Preço/histórico"


@st.cache_data(show_spinner=False)
def _shap_global_cacheado(id_municipio: str, produto: str) -> pd.DataFrame:
    return shap_global(id_municipio, produto)


@st.cache_data(show_spinner=False)
def _shap_local_cacheado(id_municipio: str, produto: str) -> dict:
    return shap_local(id_municipio, produto)


def _secao_interpretabilidade(id_municipio: str, produto: str, estado_previsao: dict) -> None:
    st.subheader("Interpretabilidade (SHAP)")
    st.caption(
        "A importância nativa do Random Forest (acima) já sugere que o preço defasado "
        "domina a previsão. O SHAP confirma isso por um método independente e explica "
        "exatamente esta previsão — a mesma pergunta que motiva o TCC: câmbio ou Brent, "
        "qual pesa mais?"
    )

    with st.spinner("Calculando importância SHAP (alguns segundos na primeira vez)..."):
        resumo = _shap_global_cacheado(id_municipio, produto).copy()
        local = _shap_local_cacheado(id_municipio, produto)

    resumo["grupo"] = resumo["feature"].map(_grupo_feature)

    fig_global = go.Figure()
    for grupo, cor in CORES_GRUPO.items():
        subset = resumo[resumo["grupo"] == grupo]
        if subset.empty:
            continue
        fig_global.add_trace(go.Bar(
            x=subset["importancia_media_abs"], y=subset["feature"], orientation="h",
            name=grupo, marker_color=cor,
        ))
    fig_global.update_layout(
        title="Importância global (média de |valor SHAP|, amostra do histórico)",
        xaxis_title="Impacto médio no preço previsto (R$)",
        yaxis=dict(autorange="reversed"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_global, width="stretch")

    cambio_imp = resumo.loc[resumo["feature"] == "usd_brl", "importancia_media_abs"]
    brent_imp = resumo.loc[resumo["feature"] == "brent_usd_bbl", "importancia_media_abs"]
    if not cambio_imp.empty and not brent_imp.empty:
        vencedor = "o câmbio" if cambio_imp.iloc[0] > brent_imp.iloc[0] else "o Brent"
        st.caption(
            f"Câmbio: impacto médio de R$ {cambio_imp.iloc[0]:.4f} · Brent: R$ {brent_imp.iloc[0]:.4f} "
            f"— entre os dois, {vencedor} pesa mais para {produto} nesta cidade."
        )

    contribuicoes = local["contribuicoes"]
    fig_local = go.Figure(go.Waterfall(
        orientation="h",
        measure=["absolute"] + ["relative"] * len(contribuicoes) + ["total"],
        y=["Valor base (média geral)"] + contribuicoes["feature"].tolist() + ["Previsão final"],
        x=[local["valor_base"]] + contribuicoes["contribuicao"].tolist() + [local["previsao"]],
        connector=dict(line=dict(color="rgba(150,150,150,0.4)")),
    ))
    fig_local.update_layout(
        title=f"Por que R$ {local['previsao']:.3f} para a semana de {estado_previsao['semana_prevista']}?",
        xaxis_title="R$",
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_local, width="stretch")


EVENTO_COLUNA_INICIO = "evento_invasao_ucrania_inicio"
EVENTO_COLUNA_DURACAO = "evento_invasao_ucrania_semanas_desde_inicio"


def _fora_da_faixa(valor: float, faixa: tuple[float, float]) -> bool:
    """True quando `valor` está fora do intervalo [min, max] visto no
    treino — usado só para avisar (não para bloquear, ver docstring de
    `_secao_whatif`)."""
    minimo, maximo = faixa
    return not (minimo <= valor <= maximo)


def _montar_cenario(
    colunas_feature: list[str],
    ultima_linha: pd.Series,
    usd_brl: float,
    brent_usd_bbl: float,
    ipca_variacao_mensal: float,
    novo_choque_geopolitico: bool,
) -> pd.Series:
    """Parte da última linha de features real (mantém preço atual, lags e
    janela móvel como estão — só o histórico observado os determina) e
    sobrescreve as variáveis macro pelos valores simulados. Se
    `novo_choque_geopolitico`, força a feature de evento a "começando
    agora" (reaproveita a coluna da invasão da Ucrânia como proxy — ver
    contexto no plano do Objetivo F)."""
    cenario = ultima_linha[colunas_feature].copy()
    cenario["usd_brl"] = usd_brl
    cenario["brent_usd_bbl"] = brent_usd_bbl
    cenario["ipca_variacao_mensal"] = ipca_variacao_mensal
    if novo_choque_geopolitico:
        cenario[EVENTO_COLUNA_INICIO] = 1
        cenario[EVENTO_COLUNA_DURACAO] = 0
    return cenario


@st.cache_data
def _faixa_treino(id_municipio: str, produto: str) -> dict[str, tuple[float, float]]:
    """Faixa (mín, máx) observada no treino para cada variável ajustável
    no what-if — só para o aviso de extrapolação (Random Forest não
    extrapola bem além do que viu, TCC seção 2.2.3), nunca para limitar o
    que o usuário pode digitar."""
    dados = build_features(id_municipio, produto)
    return {
        "usd_brl": (float(dados["usd_brl"].min()), float(dados["usd_brl"].max())),
        "brent_usd_bbl": (float(dados["brent_usd_bbl"].min()), float(dados["brent_usd_bbl"].max())),
        "ipca_variacao_mensal": (
            float(dados["ipca_variacao_mensal"].min()),
            float(dados["ipca_variacao_mensal"].max()),
        ),
    }


def _secao_whatif(id_municipio: str, produto: str, estado_previsao: dict) -> None:
    st.subheader("Simulação de cenários (what-if)")
    st.caption(
        "Ajuste as variáveis abaixo para simular um cenário hipotético — o preço "
        "atual e o histórico recente (lags/janela móvel) continuam os mesmos da "
        "previsão base acima, só as condições macro/geopolíticas mudam."
    )

    ultima_linha = estado_previsao["ultima_linha"]
    faixa = _faixa_treino(id_municipio, produto)

    col1, col2, col3 = st.columns(3)
    usd_simulado = col1.number_input(
        "Câmbio USD/BRL", value=float(ultima_linha["usd_brl"]), step=0.05, format="%.3f"
    )
    brent_simulado = col2.number_input(
        "Brent (US$/barril)", value=float(ultima_linha["brent_usd_bbl"]), step=1.0, format="%.2f"
    )
    ipca_simulado = col3.number_input(
        "IPCA variação mensal (%)", value=float(ultima_linha["ipca_variacao_mensal"]), step=0.05, format="%.2f"
    )

    novo_choque = st.checkbox(
        "Simular novo choque geopolítico nesta semana",
        help=(
            "Reaproveita a feature da invasão da Ucrânia (2022) — é a única variável "
            "geopolítica que o modelo conhece hoje, já que os choques de "
            "1973/1979/1990 discutidos no TCC são anteriores ao início da série "
            "histórica da ANP (2004) e não têm como virar feature."
        ),
    )

    avisos = []
    for rotulo, valor, chave in (
        ("Câmbio", usd_simulado, "usd_brl"),
        ("Brent", brent_simulado, "brent_usd_bbl"),
        ("IPCA", ipca_simulado, "ipca_variacao_mensal"),
    ):
        if _fora_da_faixa(valor, faixa[chave]):
            minimo, maximo = faixa[chave]
            avisos.append(f"{rotulo}: fora da faixa observada no treino ({minimo:.2f}–{maximo:.2f}).")

    if avisos:
        st.warning(
            "Random Forest não extrapola bem além do intervalo visto no treino "
            "(limitação documentada na seção 2.2.3 do TCC) — a previsão abaixo "
            "tende a saturar perto do teto/piso histórico em vez de projetar uma "
            "tendência além dele:\n\n" + "\n".join(f"- {a}" for a in avisos)
        )

    colunas_feature = estado_previsao["modelo_salvo"]["colunas_feature"]
    cenario = _montar_cenario(
        colunas_feature, ultima_linha, usd_simulado, brent_simulado, ipca_simulado, novo_choque
    )
    entrada_cenario = pd.DataFrame([cenario])
    previsao_cenario = float(estado_previsao["modelo_salvo"]["modelo"].predict(entrada_cenario)[0])

    previsao_base = estado_previsao["previsao_base"]
    delta = previsao_cenario - previsao_base

    col1, col2, col3 = st.columns(3)
    col1.metric(f"Previsão base ({estado_previsao['semana_prevista']})", f"R$ {previsao_base:.3f}")
    col2.metric("Previsão no cenário simulado", f"R$ {previsao_cenario:.3f}", delta=f"{delta:+.3f}")
    col3.metric("Impacto do cenário", f"{delta / previsao_base:+.2%}")


def main() -> None:
    st.title("Análise Preditiva de Preços de Combustíveis")
    st.caption("Dados históricos da ANP + câmbio USD/BRL, Brent e IPCA — TCC Fatec Santana de Parnaíba")

    with st.sidebar:
        st.header("Filtros")
        ufs = listar_ufs()
        uf = st.selectbox("Estado (UF)", ufs, index=ufs.index("SP") if "SP" in ufs else 0)

        municipios = listar_municipios(uf)
        if municipios.empty:
            st.warning("Nenhum município com dado nesse estado.")
            st.stop()
        nome_municipio = st.selectbox("Município", municipios["nome"])
        id_municipio = municipios.loc[municipios["nome"] == nome_municipio, "id_municipio"].iloc[0]

        produto = st.selectbox("Combustível", PRODUTOS)

    df = historico_precos(id_municipio, produto)

    if df.empty:
        st.warning(f"Sem dados históricos de {produto} para {nome_municipio}. Tente outra combinação de filtros.")
        return

    st.header(f"{nome_municipio}/{uf} — {produto}")
    _secao_kpis(df)
    st.plotly_chart(_grafico_historico(df, produto), width="stretch")
    _secao_comparativo(id_municipio, nome_municipio)
    estado_previsao = _secao_previsao(id_municipio, produto)
    if estado_previsao is not None:
        _secao_interpretabilidade(id_municipio, produto, estado_previsao)
        _secao_whatif(id_municipio, produto, estado_previsao)


if __name__ == "__main__":
    main()
