from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np

from componentes import Componente  


R_GAS = 8.314  # J/mol/K


def massa_molar_media(x: Sequence[float], comps: List[Componente]) -> float:
    """
    Calcula a massa molar média (kg/kmol) usando as composições x
    e as massas molares dos componentes.

    """
    x_arr = np.asarray(x, dtype=float)
    mm = []
    for c in comps:
        if c.MM is not None and not np.isnan(c.MM):
            MM = float(c.MM)
        else:
            
            nome = c.nome.lower()
            if "pentano" in nome:
                MM = 72.15
            elif "hexano" in nome:
                MM = 86.18
            elif "heptano" in nome:
                MM = 100.21
            elif "nonano" in nome:
                MM = 128.26
            elif "decano" in nome:
                MM = 142.29
            else:
                raise ValueError(f"Sem MM para componente {c.nome}")
        mm.append(MM)

    mm_arr = np.array(mm, dtype=float)
    return float(np.sum(x_arr * mm_arr))


@dataclass
class SecaoPratos:
    nome: str
    L_kmol_h: float
    V_kmol_h: float
    MW_vap: float      # kg/kmol
    rho_vap: float     # kg/m3
    rho_liq: float     # kg/m3
    u_flood: float     # m/s
    u_op: float        # m/s
    A_ativa: float     # m2
    A_total: float     # m2
    D: float           # m


@dataclass
class TrayColumnSizing:
    N_pratos: int
    H_ativa: float     # m
    H_total: float     # m
    diametro: float    # m 
    sec_topo: SecaoPratos
    sec_fundo: SecaoPratos


def dimensionar_secao(
    nome: str,
    L_kmol_h: float,
    V_kmol_h: float,
    x_vap: Sequence[float],
    comps: List[Componente],
    P_atm: float,
    T_K: float,
    rho_L_assumida: float,
    frac_flood: float = 0.75,
    frac_area_ativa: float = 0.80,
    C_capacidade: float = 0.15,
) -> SecaoPratos:
    """
    Dimensiona uma seção do topo ou fundo da coluna para pratos válvulados
    """
 
    MW_vap = massa_molar_media(x_vap, comps)  # kg/kmol
    P_Pa = P_atm * 101325.0
    rho_vap = P_Pa * (MW_vap / 1000.0) / (R_GAS * T_K)  # gás ideal
    rho_liq = rho_L_assumida

 
    u_flood = C_capacidade * np.sqrt((rho_liq - rho_vap) / rho_vap)
    u_op = frac_flood * u_flood


    m_dot_vap_h = V_kmol_h * MW_vap        # kg/h
    m_dot_vap_s = m_dot_vap_h / 3600.0     # kg/s
    V_vol = m_dot_vap_s / rho_vap          # m3/s


    A_ativa = V_vol / u_op
    A_total = A_ativa / frac_area_ativa
    D = np.sqrt(4.0 * A_total / np.pi)

    return SecaoPratos(
        nome=nome,
        L_kmol_h=L_kmol_h,
        V_kmol_h=V_kmol_h,
        MW_vap=MW_vap,
        rho_vap=rho_vap,
        rho_liq=rho_liq,
        u_flood=u_flood,
        u_op=u_op,
        A_ativa=A_ativa,
        A_total=A_total,
        D=D,
    )


def altura_coluna(N_pratos: int, espacamento: float = 0.5, altura_extra: float = 4.0) -> tuple[float, float]:
    """
    calcula a altura ativa e a altura total da coluna de pratos.

    espacamento: distância vertical entre pratos m
    altura_extra: folgas adicionais m
    """
    H_ativa = (N_pratos - 1) * espacamento
    H_total = H_ativa + altura_extra
    return H_ativa, H_total


def dimensionar_coluna_pratos(
    N_pratos: int,
    F: float,
    q_liq: float,
    spec,
    RR: float,
    comps: List[Componente],
    P_atm: float = 2.0,
) -> TrayColumnSizing:
    """
    dimensionar a coluna de pratos válvulados (diâmetro e altura) a partir:

      - N_pratos real
      - F, q_liq,D, xD, xB, zF
      - RR refluxo de operação

    """
    D_total = spec.D
    F_total = F

    L_R = RR * D_total
    V_R = (RR + 1.0) * D_total

    L_S = L_R + q_liq * F_total
    V_S = V_R + (1.0 - q_liq) * F_total

    sec_topo = dimensionar_secao(
        "topo",
        L_kmol_h=L_R,
        V_kmol_h=V_R,
        x_vap=spec.xD,
        comps=comps,
        P_atm=P_atm,
        T_K=370.0,        # supondo 370 K no topo
        rho_L_assumida=650.0,  # supondo
    )

    sec_fundo = dimensionar_secao(
        "fundo",
        L_kmol_h=L_S,
        V_kmol_h=V_S,
        x_vap=spec.xB,
        comps=comps,
        P_atm=P_atm,
        T_K=430.0,        # supondo 430 K no fundo
        rho_L_assumida=700.0,  # supondo
    )


    D_coluna = max(sec_topo.D, sec_fundo.D)


    H_ativa, H_total = altura_coluna(N_pratos)

    return TrayColumnSizing(
        N_pratos=N_pratos,
        H_ativa=H_ativa,
        H_total=H_total,
        diametro=D_coluna,
        sec_topo=sec_topo,
        sec_fundo=sec_fundo,
    )
