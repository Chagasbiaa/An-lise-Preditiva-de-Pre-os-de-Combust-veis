"""Testes da limpeza/agregação semanal dos preços da ANP (dados sintéticos,
sem I/O)."""

import pandas as pd

from src.processing.clean_anp import clean


def _linha(**overrides):
    base = {
        "data_coleta": pd.Timestamp("2024-01-01"),
        "ano": 2024,
        "sigla_uf": "SP",
        "id_municipio": "3550308",
        "municipio": "São Paulo",
        "produto": "Gasolina",
        "unidade_medida": "R$/litro",
        "preco_venda_medio": 5.0,
        "preco_venda_min": 4.9,
        "preco_venda_max": 5.1,
        "n_postos_pesquisados": 10,
    }
    base.update(overrides)
    return base


def test_agrega_por_semana_com_media_ponderada():
    # Duas datas exatas dentro da mesma semana ANP (domingo a sábado),
    # mesma cidade/produto, pesos diferentes.
    df = pd.DataFrame(
        [
            _linha(
                data_coleta=pd.Timestamp("2024-01-01"),
                preco_venda_medio=5.0, preco_venda_min=4.9, preco_venda_max=5.1,
                n_postos_pesquisados=10,
            ),
            _linha(
                data_coleta=pd.Timestamp("2024-01-03"),
                preco_venda_medio=6.0, preco_venda_min=5.9, preco_venda_max=6.1,
                n_postos_pesquisados=5,
            ),
        ]
    )

    resultado = clean(df)

    assert len(resultado) == 1
    linha = resultado.iloc[0]
    assert linha["preco_medio"] == (5.0 * 10 + 6.0 * 5) / 15
    assert linha["preco_min"] == 4.9
    assert linha["preco_max"] == 6.1
    assert linha["n_postos_pesquisados"] == 15


def test_sem_duplicatas_de_semana_municipio_produto():
    df = pd.DataFrame(
        [
            _linha(data_coleta=pd.Timestamp("2024-01-01")),
            _linha(data_coleta=pd.Timestamp("2024-01-03")),
            _linha(data_coleta=pd.Timestamp("2024-01-08"), municipio="São Paulo"),  # semana seguinte
        ]
    )

    resultado = clean(df)

    assert not resultado.duplicated(subset=["semana", "id_municipio", "produto"]).any()
    assert len(resultado) == 2  # duas semanas distintas


def test_descarta_linha_com_municipio_nulo():
    df = pd.DataFrame(
        [
            _linha(),
            _linha(id_municipio=None, municipio=None, data_coleta=pd.Timestamp("2024-02-01")),
        ]
    )

    resultado = clean(df)

    assert len(resultado) == 1
    assert resultado["id_municipio"].notna().all()


def test_unidade_medida_derivada_do_produto_mesmo_com_nulo_na_origem():
    df = pd.DataFrame([_linha(unidade_medida=None, produto="Glp")])

    resultado = clean(df)

    assert resultado.iloc[0]["unidade_medida"] == "R$/13kg"


def test_marca_outlier_por_produto_via_iqr():
    linhas = []
    for i, preco in enumerate([3.0, 3.1, 3.2, 3.0, 3.1]):
        linhas.append(
            _linha(
                produto="Etanol",
                id_municipio=str(1000 + i),
                municipio=f"Cidade {i}",
                data_coleta=pd.Timestamp("2024-01-01"),
                preco_venda_medio=preco,
                preco_venda_min=preco - 0.05,
                preco_venda_max=preco + 0.05,
            )
        )
    # Um valor claramente fora do padrão do mesmo produto.
    linhas.append(
        _linha(
            produto="Etanol",
            id_municipio="9999",
            municipio="Cidade Outlier",
            data_coleta=pd.Timestamp("2024-01-01"),
            preco_venda_medio=100.0,
            preco_venda_min=99.0,
            preco_venda_max=101.0,
        )
    )
    df = pd.DataFrame(linhas)

    resultado = clean(df)

    outliers = resultado[resultado["preco_outlier"]]
    assert len(outliers) == 1
    assert outliers.iloc[0]["id_municipio"] == "9999"
