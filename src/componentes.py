from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


@dataclass
class Componente:
    
    nome: str 
    indice: str  
    Tb: float #ponto de ebulicao (K ou celcius)
    MM: float # kg/mol
    alpha_ref: float # volatilidade relatica em relacao ao menos volatil ou HK
    dens_liq: float #kg/m3
    dens_vap: float #kg/m3
    viscosidade: float #cP
    tensao_superficial: float # a escolher unidade
    

def carregar_componentes(caminho_csv:str) ->List[Componente]:
    
    """
    Le arquivo de componentes.csv e retorna lista de componente ordenada do mais volatil ao menos
    """
    
    caminho_csv = Path(caminho_csv)
    df = pd.read_csv(caminho_csv)
    
    comps:List[Componente] = []
    for _, row in df.iterrows():
        comps.append(
            Componente(
                indice=int(row["indice"]),
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
        
        #ordenando do mais ao menos volatil pelo alpha de referencia
    comps.sort(key=lambda c: (-(c.alpha_ref or 0.0), c.indice))
    
    return comps 
    
def _none_if_nan(valor):
    try:
        import math

        if valor is None:
            return None
        if isinstance(valor, str) and not valor.strip():
            return None
        if isinstance(valor, (int, float)) and math.isnan(valor):
            return None
        return float(valor)
    except Exception:
        return None
        
