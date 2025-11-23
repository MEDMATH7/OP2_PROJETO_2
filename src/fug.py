# src/fug.py
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
    """
    Calcula N_min pelo método de Fenske (Classe I, α constantes).

    xD, xB: composições de topo e fundo (listas ou arrays)
    i_LK, i_HK: índices (0..n-1) da light key e heavy key
    alpha: volatilidades relativas em relação a um componente de referência
           (no nosso caso: [C5, C6, C7, C9, C10] = [3.0, 2.3, 1.8, 1.3, 1.0])
    """
    xD = np.asarray(xD, dtype=float)
    xB = np.asarray(xB, dtype=float)
    alpha = np.asarray(alpha, dtype=float)

    alpha_LKHK = alpha[i_LK] / alpha[i_HK]

    num = (xD[i_LK] / xD[i_HK]) * (xB[i_HK] / xB[i_LK])
    Nmin = np.log(num) / np.log(alpha_LKHK)
    return Nmin


def _underwood_balance(theta: float, alpha: np.ndarray, zF: np.ndarray, q_liq: float) -> float:
    """
    Função f(θ) da equação de Underwood:
        sum( α_i z_F,i / (α_i - θ) ) = 1 - q_liq
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
    Resolve θ pela equação de Underwood usando bissecção.
    θ deve estar entre α_HK e α_LK.
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
    e devolve também o θ encontrado.
    """
    alpha = np.asarray(alpha, dtype=float)
    xD = np.asarray(xD, dtype=float)
    zF = np.asarray(zF, dtype=float)

    theta = solve_underwood_theta(alpha, zF, q_liq, i_HK=i_HK, i_LK=i_LK)
    RR_min = np.sum(alpha * xD / (alpha - theta)) - 1.0
    return RR_min, theta


def gilliland_N(Nmin: float, RR: float, RR_min: float) -> float:
    """
    Correlação de Gilliland (forma de Eduljee):

      X = (R - R_min) / (R + 1)
      Y = 0.75 * (1 - X**0.5668)
      Y = (N - Nmin) / (N + 1)

    Retorna N (número de estágios teóricos para RR escolhido).
    """
    Nmin = float(Nmin)
    RR = float(RR)
    RR_min = float(RR_min)

    X = (RR - RR_min) / (RR + 1.0)
    X = max(1e-6, min(0.999999, X))  # evita problemas numéricos

    Y = 0.75 * (1.0 - X**0.5668)

    N = (Nmin + Y) / (1.0 - Y)
    return N
