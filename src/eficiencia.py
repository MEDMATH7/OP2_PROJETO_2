
from __future__ import annotations

from typing import Sequence

import numpy as np


def estimar_mu_feed_cP(zF: Sequence[float]) -> float:

    zF_arr = np.asarray(zF, dtype=float)
    if zF_arr.shape[0] != 5:
        raise ValueError("zF deve ter 5 componentes (n-C5, n-C6, n-C7, n-C9, n-C10).")

    # viscosidades em cP a 25 graus (ordem de grandeza, literatura)
    mu_cP = np.array([0.224, 0.295, 0.389, 0.665, 0.850], dtype=float)

    mu_mix = float(np.dot(zF_arr, mu_cP))
    return mu_mix


def oconnell_eta(alpha_rel: float, mu_F_cP: float) -> float:

    alpha_mu = alpha_rel * mu_F_cP
    if alpha_mu <= 0:
        raise ValueError("alpha_rel * mu_F_cP deve ser > 0.")

    eta_G = 0.492 * (alpha_mu ** (-0.245))
    return float(eta_G)


def calcular_N_real(N_teorico: float, eta_G: float) -> float:

    if eta_G <= 0:
        raise ValueError("eta_G deve ser > 0.")
    return float(N_teorico / eta_G)
