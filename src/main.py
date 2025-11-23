# src/main.py
from __future__ import annotations

from pathlib import Path

import numpy as np

from componentes import carregar_componentes
from especificacao import definir_especificacao_por_recuperacoes
from fug import fenske_Nmin, underwood_RRmin, gilliland_N
from eficiencia import estimar_mu_feed_cP, oconnell_eta, calcular_N_real


def definir_alimentacao():
    """
    Alimentação segundo o enunciado do projeto:

      F = 1000 kmol/h
      z = [0.05, 0.10, 0.25, 0.30, 0.30]  (n-C5 -> n-C10)
      20% vaporizada a 2 atm

    IMPORTANTE:
      Na equação de Underwood, o parâmetro q é a FRAÇÃO LÍQUIDA.
      Se a fração vaporizada é 0.20, então q_liq = 0.80.
    """
    F = 1000.0  # kmol/h
    z = [0.05, 0.10, 0.25, 0.30, 0.30]  # [n-C5, n-C6, n-C7, n-C9, n-C10]
    fracao_vapor = 0.20
    q_liq = 1.0 - fracao_vapor  # 0.80
    P = 2.0  # atm
    return F, z, q_liq, P, fracao_vapor


def check_composicao(z, tol: float = 1e-6):
    soma = sum(z)
    if abs(soma - 1.0) > tol:
        raise ValueError(f"Soma das frações molares z = {soma:.6f} (não é 1.0)")
    return soma


def main():
    base_dir = Path(__file__).resolve().parent.parent

    # 1) Componentes
    caminho_componentes = base_dir / "data" / "componentes.csv"
    comps = carregar_componentes(caminho_componentes)

    print("=== COMPONENTES (mais volátil -> menos volátil) ===")
    for c in comps:
        print(f"Comp {c.indice} (prof {c.codigo_prof}): {c.nome}")

    # 2) Alimentação
    F, z, q_liq, P, fracao_vapor = definir_alimentacao()
    soma_z = check_composicao(z)

    print("\n=== ALIMENTAÇÃO ===")
    print(f"F = {F} kmol/h")
    print(f"z = {z}  (soma = {soma_z:.4f})")
    print("Ordem z: [n-C5, n-C6, n-C7, n-C9, n-C10]")
    print(f"fração vaporizada = {fracao_vapor:.2f}")
    print(f"q (fração líquida para Underwood) = {q_liq:.2f}")
    print(f"P = {P} atm")

    # 3) Especificação de separação (Etapa 3)
    spec = definir_especificacao_por_recuperacoes(F, z)

    print("\n=== ESPECIFICAÇÃO DE SEPARAÇÃO (proposta) ===")
    print(f"D = {spec.D:.2f} kmol/h, B = {spec.B:.2f} kmol/h")
    print(f"LK_index = {spec.LK_index} (componente físico: {comps[spec.LK_index].nome})")
    print(f"HK_index = {spec.HK_index} (componente físico: {comps[spec.HK_index].nome})")

    header = (
        "\nComp  Nome       F_i     R_D      D_i      B_i      xD       xB\n"
        "----- ---------- ------- -------- ------- -------- -------- --------"
    )
    print(header)
    for i, c in enumerate(comps):
        Fi = spec.F_i[i]
        RD = spec.R_D[i]
        Di = spec.D_i[i]
        Bi = spec.B_i[i]
        xDi = spec.xD[i]
        xBi = spec.xB[i]
        print(
            f"{i+1:>3}   {c.nome:<10} {Fi:7.2f} {RD:8.4f} {Di:7.2f} {Bi:8.2f} {xDi:8.4f} {xBi:8.4f}"
        )

    # 4) MÉTODO FUG (Etapa 4)
    print("\n=== MÉTODO FUG (Fenske–Underwood–Gilliland) ===")

    # 4.1 Definir volatilidades relativas α para cada componente
    # Ordem: [n-C5, n-C6, n-C7, n-C9, n-C10]
    # Chutes plausíveis (sem HYSYS):
    alpha = np.array([3.0, 2.3, 1.8, 1.3, 1.0])

    i_LK = spec.LK_index  # n-C7
    i_HK = spec.HK_index  # n-C9

    # 4.2 Fenske -> N_min
    Nmin = fenske_Nmin(spec.xD, spec.xB, i_LK, i_HK, alpha)
    print(f"N_min (Fenske) = {Nmin:.2f} estágios teóricos")

    # 4.3 Underwood -> R_R,min
    RR_min, theta = underwood_RRmin(alpha, spec.xD, spec.zF, q_liq, i_LK, i_HK)
    print(f"R_R,min (Underwood) = {RR_min:.3f}")
    print(f"θ (Underwood) = {theta:.4f}")

    # 4.4 Gilliland -> N teórico para refluxo de operação
    RR = 1.30 * RR_min  # escolha inicial: 1,3 * RR_min
    Nteo = gilliland_N(Nmin, RR, RR_min)
    print(f"R_R (operação) = {RR:.3f}")
    print(f"N teórico (Gilliland) = {Nteo:.2f} estágios")
    # 5) EFICIÊNCIA GLOBAL (O'CONNELL) E NÚMERO DE PRATOS REAIS
    print("\n=== ETAPA 5 – EFICIÊNCIA GLOBAL (O'Connell) E N_real ===")

    # 5.1 Estimar viscosidade da alimentação
    mu_F_cP = estimar_mu_feed_cP(spec.zF)
    print(f"Viscosidade média estimada da alimentação: μ_F ≈ {mu_F_cP:.3f} cP")

    # 5.2 Volatilidade relativa "chave" (mesma usada em Fenske)
    alpha_rel = alpha[i_LK] / alpha[i_HK]
    print(f"Volatilidade relativa usada na O'Connell: α_rel (LK/HK) ≈ {alpha_rel:.3f}")

    # 5.3 Eficiência global pela correlação de O'Connell
    eta_G = oconnell_eta(alpha_rel, mu_F_cP)
    print(f"Eficiência global estimada (O'Connell): η_G ≈ {eta_G:.3f} (fração)")
    print(f"Eficiência global ≈ {eta_G*100:.1f} %")

    # 5.4 Número de pratos reais
    N_real = calcular_N_real(Nteo, eta_G)
    print(f"Número de estágios reais (antes de arredondar): N_real ≈ {N_real:.2f}")

    # arredonda pra cima para obter número inteiro de pratos
    import math
    N_pratos = math.ceil(N_real)
    print(f"Número de pratos de projeto (arredondado para cima): N_pratos = {N_pratos}")



if __name__ == "__main__":
    main()
