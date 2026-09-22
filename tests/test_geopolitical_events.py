"""Testes da codificação de eventos geopolíticos (dados sintéticos)."""

import pandas as pd

from src.features.geopolitical_events import codificar_eventos


def test_dummy_e_contador_da_invasao_da_ucrania():
    semanas = pd.to_datetime(
        [
            "2022-02-06",  # bem antes
            "2022-02-20",  # semana ANP que contém 24/02/2022 (domingo a sábado)
            "2022-02-27",  # semana seguinte
            "2022-03-06",  # duas semanas depois
        ]
    )

    resultado = codificar_eventos(semanas)

    assert resultado["evento_invasao_ucrania_inicio"].tolist() == [0, 1, 0, 0]
    assert resultado["evento_invasao_ucrania_semanas_desde_inicio"].tolist() == [0, 0, 1, 2]
