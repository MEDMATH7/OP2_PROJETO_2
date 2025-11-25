
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from componentes import Componente
from dimensionamento_pratos import massa_molar_media


@dataclass
class PackedColumnSizing:
    D_m: float             # diâmetro da coluna (m)
    H_recheio_m: float     # altura de recheio (m)
    H_total_m: float       # altura total sugerida (m)
    MM_top: float          # massa molar média no topo (kg/kmol)
    rho_V: float           # densidade do vapor (kg/m3)
    rho_L: float           # densidade do líquido (kg/m3)
    FLV: float             # F_LV
    Y: float              
    uV_flood_m_s: float    # velocidade de inundação (m/s)
    uV_op_m_s: float       # velocidade de operação (m/s)
    A_m2: float            # área de seção transversal (m2)


def dimensionar_coluna_recheada_intalox(
    spec,
    RR: float,
    comps: List[Componente],
    N_teorico: float,
    P_atm: float = 2.0,
    T_top_K: float = 370.0,
    rho_L: float = 630.0,
    mu_L_cP: float = 0.5,
    phi: float = 0.70,
) -> PackedColumnSizing:
    """
    dimensionar a coluna recheada com Intalox Saddles 1" usando:

      - Método de Leva para velocidade de inundação
      - Operação a phi * uV_flood (phi ~ 0,7)
      - HETP = 1,5 * Dp[in]  (recheio aleatório, líquidos pouco viscosos)

    Usa exclusivamente:
      - spec.D, spec.xD  (vazão e composição de topo)
      - RR   (razão de refluxo)
      - N_teorico    (estágios teóricos do FUG)
      - comps         (para MM)

    """

    # Dados de topo
    D_total = spec.D  # kmol/h de destilado
    xD = spec.xD      # composição de topo


    MM_top = massa_molar_media(xD, comps)

    # densidade do vapor no topo (gás ideal)

    P_kPa = P_atm * 101.325
    Rbar = 8.314
    rho_V = P_kPa * MM_top / (Rbar * T_top_K)  # kg/m3

    #Vazões L0 e V1 em kmol/h
    L0_kmol_h = RR * D_total
    V1_kmol_h = (RR + 1.0) * D_total  # L0 + D

    # Converter p kg/h
    L_mass_kg_h = L0_kmol_h * MM_top
    V_mass_kg_h = V1_kmol_h * MM_top

    # F_LV 
    FLV = (L_mass_kg_h / V_mass_kg_h) * math.sqrt(rho_V / rho_L)

    # Método de Leva (Intalox Saddles 1 polegada)
    lnF = math.log(FLV)
    Y = math.exp(
        -3.7121
        - 1.0371 * lnF
        - 0.1501 * lnF**2
        - 0.007544 * lnF**3
    )

    rho_H2O = 995.6  # kg/m3

    F1 = (
        -0.8787
        + 2.6776 * (rho_H2O / rho_L)
        - 0.6313 * (rho_H2O / rho_L) ** 2
    )
    F2 = 0.96 * (mu_L_cP ** 0.19)
    Fp = 92.0 


    uV_flood_ft_s = math.sqrt(
        32.2 * Y * (rho_H2O / rho_V) / (Fp * F1 * F2)
    )
    # converte p m/s
    uV_flood_m_s = uV_flood_ft_s * 0.3048


    uV_op_m_s = phi * uV_flood_m_s

    # --- 6) Área e diâmetro ---
    V_mass_kg_s = V_mass_kg_h / 3600.0  # kg/s
    G = rho_V * uV_op_m_s               # kg/m2.s
    A_m2 = V_mass_kg_s / G              # m2

    D_m = math.sqrt(4.0 * A_m2 / math.pi)

    #altura pelo HETP

    HETP_ft = 1.5
    HETP_m = HETP_ft * 0.3048

    H_recheio_m = N_teorico * HETP_m

    # Folgas em cima/baixo... chute de 2 metros
    H_total_m = H_recheio_m + 2.0

    return PackedColumnSizing(
        D_m=D_m,
        H_recheio_m=H_recheio_m,
        H_total_m=H_total_m,
        MM_top=MM_top,
        rho_V=rho_V,
        rho_L=rho_L,
        FLV=FLV,
        Y=Y,
        uV_flood_m_s=uV_flood_m_s,
        uV_op_m_s=uV_op_m_s,
        A_m2=A_m2,
    )
