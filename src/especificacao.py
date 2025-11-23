# src/especificacao.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class SeparationSpec:
    xD: List[float]      # comp de topo (mol/mol) dos 5 componentes
    xB: List[float]      # comp de fundo (mol/mol)
    LK_index: int        # indice do componente leve-chave
    HK_index: int        # indice do componente pesado-chave


def definir_especificacao_inicial(n_comp: int = 5) -> SeparationSpec:

    xD = [0.0] * n_comp
    xB = [0.0] * n_comp
    LK_index = 1  
    HK_index = 3  

    return SeparationSpec(xD=xD, xB=xB, LK_index=LK_index, HK_index=HK_index)
    
