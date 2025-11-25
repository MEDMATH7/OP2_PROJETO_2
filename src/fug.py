from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


def fenske_Nmin(
    xD: Sequence[float],
    xB: Sequence[float],
    i_LK: int,
    i_HK: int,
    alpha: Sequence[float],
) -> float:

    xD = np.asarray(xD, dtype=float)
    xB = np.asarray(xB, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    alpha_LKHK = alpha[i_LK] / alpha[i_HK]

    num = (xD[i_LK] / xD[i_HK]) * (xB[i_HK] / xB[i_LK])
    Nmin = np.log(num) / np.log(alpha_LKHK)
    return Nmin


def _underwood_balance(theta: float, alpha: np.ndarray, zF: np.ndarray, q_liq: float) -> float:
    """
    Funçãoda equação de Underwood:

    """
    return np.sum(alpha * zF / (alpha - theta)) - (1.0 - q_liq)


def solve_underwood_theta(
    alpha: Sequence[float],
    zF: Sequence[float],
    q_liq: float,
    i_HK: int,
    i_LK: int,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """
    Resolve pela equação de Underwood usando bissecção.
    """
    alpha = np.asarray(alpha, dtype=float)
    zF = np.asarray(zF, dtype=float)

    alpha_HK = alpha[i_HK]
    alpha_LK = alpha[i_LK]

    lo = alpha_HK + 1e-6
    hi = alpha_LK - 1e-6

    f_lo = _underwood_balance(lo, alpha, zF, q_liq)
    f_hi = _underwood_balance(hi, alpha, zF, q_liq)
    if f_lo * f_hi > 0:
        raise ValueError(
            "Underwood: não achei intervalo com mudança de sinal. "
            "Verifique α, zF, q_liq e os índices de LK/HK."
        )

    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = _underwood_balance(mid, alpha, zF, q_liq)
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid

    raise RuntimeError("Underwood: bissecção não convergiu dentro do número máximo de iterações.")


def underwood_RRmin(
    alpha: Sequence[float],
    xD: Sequence[float],
    zF: Sequence[float],
    q_liq: float,
    i_LK: int,
    i_HK: int,
) -> Tuple[float, float]:
    """
    Calcula a razão de refluxo mínima R_R,min por Underwood
    e devolve o theta encontrado.
    """
    alpha = np.asarray(alpha, dtype=float)
    xD = np.asarray(xD, dtype=float)
    zF = np.asarray(zF, dtype=float)

    theta = solve_underwood_theta(alpha, zF, q_liq, i_HK=i_HK, i_LK=i_LK)
    RR_min = np.sum(alpha * xD / (alpha - theta)) - 1.0
    return RR_min, theta


def gilliland_N(Nmin: float, RR: float, RR_min: float) -> float:
    """
    Correlação de Gilliland:

      X = (R - R_min) / (R + 1)
      Y = 0.75 * (1 - X**0.5668)
      Y = (N - Nmin) / (N + 1)

    Retorna N 
    """
    Nmin = float(Nmin)
    RR = float(RR)
    RR_min = float(RR_min)

    X = (RR - RR_min) / (RR + 1.0)
    X = max(1e-6, min(0.999999, X))  

    Y = 0.75 * (1.0 - X**0.5668)

    N = (Nmin + Y) / (1.0 - Y)
    return N



def fenske_NR_NS_min(
    xD,
    xB,
    zF,
    i_LK: int,
    i_HK: int,
    alpha,
):
    """
    Calcula os estágios mínimos acima (NR_min) e abaixo (NS_min) da carga
    usando a forma segmentada da Fenske para LK/HK.

    """
    xD = np.asarray(xD, dtype=float)
    xB = np.asarray(xB, dtype=float)
    zF = np.asarray(zF, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    alpha_LKHK = alpha[i_LK] / alpha[i_HK]

    ratio_D = xD[i_LK] / xD[i_HK]
    ratio_F = zF[i_LK] / zF[i_HK]
    ratio_B = xB[i_LK] / xB[i_HK]

    NR_min = np.log((ratio_D / ratio_F)) / np.log(alpha_LKHK)
    NS_min = np.log((ratio_F / ratio_B)) / np.log(alpha_LKHK)

    return float(NR_min), float(NS_min)


def calcular_prato_otimo(
    Nmin: float,
    Nteo: float,
    N_real: float,
    NR_min: float,
    NS_min: float,
):

    f = Nteo / Nmin

    NR_op = f * NR_min
    NS_op = f * NS_min

    # prato de alimentação em estágios teóricos (contando do topo)
    j_feed_teo = NR_op + 1.0

    fator_real = N_real / Nteo
    NR_real = NR_op * fator_real
    j_feed_real = NR_real + 1.0

    return NR_op, NS_op, j_feed_teo, j_feed_real
