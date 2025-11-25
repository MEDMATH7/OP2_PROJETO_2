from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


@dataclass
class Componente:
    indice: int               # 1..5 (1 = mais volátil, 5 = menos volátil)
    codigo_prof: int          # 5, 6, 7, 9, 10 
    nome: str
    Tb: float | None = None              # ponto de ebulição 
    MM: float | None = None              # kg/kmol
    alpha_ref: float | None = None       # volatilidade relativa
    dens_liq: float | None = None        # kg/m3
    dens_vap: float | None = None        # kg/m3
    viscosidade: float | None = None     # cP
    tensao_superficial: float | None = None  


def carregar_componentes(caminho_csv: str | Path) -> List[Componente]:
    """
    Lê o arquivo componentes.csv e retorna uma lista de Componente
    ordenada do mais volátil para o menos volátil
    """
    caminho_csv = Path(caminho_csv)
    df = pd.read_csv(caminho_csv)

    comps: List[Componente] = []
    for _, row in df.iterrows():
        comps.append(
            Componente(
                indice=int(row["indice"]),
                codigo_prof=int(row["codigo_prof"]),
                nome=str(row["nome"]),
                Tb=_none_if_nan(row.get("Tb")),
                MM=_none_if_nan(row.get("MM")),
                alpha_ref=_none_if_nan(row.get("alpha_ref")),
                dens_liq=_none_if_nan(row.get("dens_liq")),
                dens_vap=_none_if_nan(row.get("dens_vap")),
                viscosidade=_none_if_nan(row.get("viscosidade")),
                tensao_superficial=_none_if_nan(row.get("tensao_superficial")),
            )
        )

    # Ordena C5 -> C10
    comps.sort(key=lambda c: c.indice)

    return comps


def _none_if_nan(valor):
    import math

    if valor is None:
        return None
    if isinstance(valor, str) and not valor.strip():
        return None
    try:
        v = float(valor)
    except (TypeError, ValueError):
        return None
    if math.isnan(v):
        return None
    return v
