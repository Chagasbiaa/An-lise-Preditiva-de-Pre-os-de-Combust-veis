"""Codificação de eventos geopolíticos como features (TCC, seção 2.2.3).

Só entram aqui eventos com efeito potencialmente capturável nos dados —
ou seja, ocorridos dentro do período coberto pela série da ANP (2004 até
hoje). Os choques históricos analisados no capítulo 3 do TCC (Yom Kippur
1973, Revolução Iraniana/Guerra Irã-Iraque 1979-1988, Guerra do Golfo
1990-1991) são anteriores a 2004 e servem só como contexto/literatura —
não podem virar feature porque não há preço de combustível brasileiro
medido naquela época na base usada aqui.

Cada evento gera duas colunas (os dois mecanismos descritos no TCC):
- `evento_<slug>_inicio`: 1 na semana de início do evento, 0 nas demais.
- `evento_<slug>_semanas_desde_inicio`: 0 antes do início; depois, conta
  quantas semanas se passaram desde o início (captura a propagação
  gradual do impacto, para eventos de efeito persistente).
"""

from __future__ import annotations

import pandas as pd

EVENTOS = [
    {"slug": "invasao_ucrania", "nome": "Invasão da Ucrânia pela Rússia", "data_inicio": "2022-02-24"},
]


def codificar_eventos(semana: pd.Series) -> pd.DataFrame:
    """Recebe a coluna `semana` (datetime) de uma série de preços e devolve
    um DataFrame com duas colunas por evento em EVENTOS, alinhado ao mesmo
    índice de `semana`."""
    semana = pd.Series(pd.to_datetime(semana)).reset_index(drop=True)
    colunas = {}

    for evento in EVENTOS:
        inicio = pd.Timestamp(evento["data_inicio"])
        semana_do_inicio = inicio.to_period("W-SAT").start_time

        dummy_inicio = (semana == semana_do_inicio).astype(int)

        dias_desde_inicio = (semana - semana_do_inicio).dt.days
        semanas_desde_inicio = (dias_desde_inicio // 7).clip(lower=0)

        colunas[f"evento_{evento['slug']}_inicio"] = dummy_inicio
        colunas[f"evento_{evento['slug']}_semanas_desde_inicio"] = semanas_desde_inicio

    return pd.DataFrame(colunas, index=semana.index)
