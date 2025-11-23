
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class SeparationSpec:
    """
      0: n-C5  (n-pentano)  - mais volátil
      1: n-C6  (n-hexano)
      2: n-C7  (n-heptano)  - LK (leve-chave)
      3: n-C9  (n-nonano)   - HK (pesado-chave)
      4: n-C10 (n-decano)   - mais pesado
    """

    # Dados da alimentação
    F: float               # vazão total de alimentação (kmol/h)
    zF: List[float]        # composição da alimentação (mol/mol)
    F_i: List[float]       # vazões molares de cada componente em F (kmol/h)

    # Vazões de topo e fundo
    D: float               # vazão total de destilado (kmol/h)
    B: float               # vazão total de fundo (kmol/h)
    D_i: List[float]       # vazões molares de cada componente em D (kmol/h)
    B_i: List[float]       # vazões molares de cada componente em B (kmol/h)

    # Composições de topo e fundo
    xD: List[float]        # composição de topo (mol/mol)
    xB: List[float]        # composição de fundo (mol/mol)

    # Recuperações
    R_D: List[float]       # fração de cada componente recuperada no destilado

    # Componentes chave (índices 0..4 nas listas acima)
    LK_index: int          # índice do componente leve-chave (aqui: n-C7 -> 2)
    HK_index: int          # índice do componente pesado-chave (aqui: n-C9 -> 3)


def definir_especificacao_por_recuperacoes(F: float, z: List[float]) -> SeparationSpec:
    """
    Define a especificação de separação usando recuperações no destilado (R_D),
    a partir da alimentação F, z.

    Ordem esperada de z: [n-C5, n-C6, n-C7, n-C9, n-C10].

    Os valores de R_D abaixo são uma PROPOSTA baseada:
      - no critério do projeto (remover o máximo possível de n-C10 do topo)
        
      - nas definições de componentes chave LK/HK
        

    """

    n = len(z)
    if n != 5:
        raise ValueError("Esta função espera exatamente 5 componentes (n-C5, n-C6, n-C7, n-C9, n-C10).")

    # Recuperações propostas no destilado
    # 0: n-C5  -> quase tudo no topo
    # 1: n-C6  -> quase tudo no topo
    # 2: n-C7  (LK) -> presente em D e B
    # 3: n-C9  (HK) -> pouco no topo
    # 4: n-C10 -> traço no topo
    R_D = [0.999, 0.995, 0.60, 0.05, 0.001]

    if len(R_D) != n:
        raise ValueError("Tamanho de R_D incompatível com z.")

    # Vazões molares em F
    F_i = [F * zi for zi in z]

    # Vazões em D e B
    D_i = [Fi * Ri for Fi, Ri in zip(F_i, R_D)]
    B_i = [Fi - Di for Fi, Di in zip(F_i, D_i)]

    D = sum(D_i)
    B = sum(B_i)

    # Composições
    xD = [Di / D for Di in D_i]
    xB = [Bi / B for Bi in B_i]

    # Chaves: LK = n-C7 (índice 2), HK = n-C9 (índice 3)
    LK_index = 2
    HK_index = 3

    return SeparationSpec(
        F=F,
        zF=list(z),
        F_i=F_i,
        D=D,
        B=B,
        D_i=D_i,
        B_i=B_i,
        xD=xD,
        xB=xB,
        R_D=R_D,
        LK_index=LK_index,
        HK_index=HK_index,
    )
