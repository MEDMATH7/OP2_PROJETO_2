from __future__ import annotations

from pathlib import Path

import numpy as np

from componentes import carregar_componentes
from especificacao import definir_especificacao_por_recuperacoes
from fug import (
    fenske_Nmin,
    underwood_RRmin,
    gilliland_N,
    fenske_NR_NS_min,
    calcular_prato_otimo,
)

from eficiencia import estimar_mu_feed_cP, oconnell_eta, calcular_N_real
from dimensionamento_pratos import dimensionar_coluna_pratos
from dimensionamento_recheios import dimensionar_coluna_recheada_intalox



def definir_alimentacao():
    """
      F = 1000 kmol/h
      z = [0.05, 0.10, 0.25, 0.30, 0.30]  (n-C5 -> n-C10)
      0.20 vaporizada a 2 atm
    """
    F = 1000.0  # kmol/h
    z = [0.05, 0.10, 0.25, 0.30, 0.30]  # [n-C5, n-C6, n-C7, n-C9, n-C10]
    fracao_vapor = 0.20
    q_liq = 1.0 - fracao_vapor  
    P = 2.0  
    return F, z, q_liq, P, fracao_vapor


def check_composicao(z, tol: float = 1e-6):
    soma = sum(z)
    if abs(soma - 1.0) > tol:
        raise ValueError(f"Soma das frações molares z = {soma:.6f} (não é 1.0)")
    return soma


def main():
    base_dir = Path(__file__).resolve().parent.parent


    caminho_componentes = base_dir / "data" / "componentes.csv"
    comps = carregar_componentes(caminho_componentes)

    print("=== COMPONENTES (mais volátil -> menos volátil) ===")
    for c in comps:
        print(f"Comp {c.indice} (prof {c.codigo_prof}): {c.nome}")


    F, z, q_liq, P, fracao_vapor = definir_alimentacao()
    soma_z = check_composicao(z)

    print("\n=== ALIMENTAÇÃO ===")
    print(f"F = {F} kmol/h")
    print(f"z = {z}  (soma = {soma_z:.4f})")
    print("Ordem z: [n-C5, n-C6, n-C7, n-C9, n-C10]")
    print(f"fração vaporizada = {fracao_vapor:.2f}")
    print(f"q (fração líquida para Underwood) = {q_liq:.2f}")
    print(f"P = {P} atm")


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

    print("\n=== MÉTODO FUG (Fenske–Underwood–Gilliland) ===")

    # Chutes plausíveis (sem HYSYS):
    alpha = np.array([3.0, 2.3, 1.8, 1.3, 1.0])

    i_LK = spec.LK_index  # n-C7
    i_HK = spec.HK_index  # n-C9


    Nmin = fenske_Nmin(spec.xD, spec.xB, i_LK, i_HK, alpha)
    print(f"N_min (Fenske) = {Nmin:.2f} estágios teóricos")


    RR_min, theta = underwood_RRmin(alpha, spec.xD, spec.zF, q_liq, i_LK, i_HK)
    print(f"R_R,min (Underwood) = {RR_min:.3f}")
    print(f"θ (Underwood) = {theta:.4f}")


    RR = 1.30 * RR_min  # escolha inicial: 1,3 * RR_min
    Nteo = gilliland_N(Nmin, RR, RR_min)
    print(f"R_R (operação) = {RR:.3f}")
    print(f"N teórico (Gilliland) = {Nteo:.2f} estágios")

    print("\n=== ETAPA 5 – EFICIÊNCIA GLOBAL (O'Connell) E N_real ===")


    mu_F_cP = estimar_mu_feed_cP(spec.zF)
    print(f"Viscosidade média estimada da alimentação: μ_F ≈ {mu_F_cP:.3f} cP")


    alpha_rel = alpha[i_LK] / alpha[i_HK]
    print(f"Volatilidade relativa usada na O'Connell: α_rel (LK/HK) ≈ {alpha_rel:.3f}")


    eta_G = oconnell_eta(alpha_rel, mu_F_cP)
    print(f"Eficiência global estimada (O'Connell): η_G ≈ {eta_G:.3f} (fração)")
    print(f"Eficiência global ≈ {eta_G*100:.1f} %")


    N_real = calcular_N_real(Nteo, eta_G)
    print(f"Número de estágios reais (antes de arredondar): N_real ≈ {N_real:.2f}")


    import math
    N_pratos = math.ceil(N_real)
    print(f"Número de pratos de projeto (arredondado para cima): N_pratos = {N_pratos}")
    


    NR_min, NS_min = fenske_NR_NS_min(
        spec.xD,
        spec.xB,
        spec.zF,
        i_LK,
        i_HK,
        alpha,
    )

    NR_op, NS_op, j_feed_teo, j_feed_real = calcular_prato_otimo(
        Nmin=Nmin,
        Nteo=Nteo,
        N_real=N_real,
        NR_min=NR_min,
        NS_min=NS_min,
    )

    print("\n=== PRATO ÓTIMO DE ALIMENTAÇÃO ===")
    print(f"NR_min ≈ {NR_min:.2f} (estágios mínimos acima da carga)")
    print(f"NS_min ≈ {NS_min:.2f} (estágios mínimos abaixo da carga)")
    print(f"NR_op  ≈ {NR_op:.2f} (estágios operacionais teóricos acima da carga)")
    print(f"NS_op  ≈ {NS_op:.2f} (estágios operacionais teóricos abaixo da carga)")
    print(f"Prato de alimentação (teórico, do topo): j_feed,teo ≈ {j_feed_teo:.1f}")

    j_feed_real_int = math.ceil(j_feed_real)
    print(
        f"Prato de alimentação correspondente na coluna real ≈ prato {j_feed_real_int} "
        f"de {N_pratos} (contando do topo)"
    )


    

    print("\n=== ETAPA 6 – DIMENSIONAMENTO DA COLUNA DE PRATOS VÁLVULADOS ===")

    sizing = dimensionar_coluna_pratos(
        N_pratos=N_pratos,
        F=F,
        q_liq=q_liq,
        spec=spec,
        RR=RR,
        comps=comps,
        P_atm=P,
    )

    print(f"Diâmetro da coluna (adotado) ≈ {sizing.diametro:.2f} m")
    print(f"Altura ativa (entre pratos) ≈ {sizing.H_ativa:.1f} m")
    print(f"Altura total estimada ≈ {sizing.H_total:.1f} m")

    print("\n--- Seção de Topo ---")
    st = sizing.sec_topo
    print(f"V_topo = {st.V_kmol_h:.1f} kmol/h")
    print(f"MW_vap_topo ≈ {st.MW_vap:.1f} kg/kmol")
    print(f"rho_vap_topo ≈ {st.rho_vap:.2f} kg/m3")
    print(f"u_flood_topo ≈ {st.u_flood:.2f} m/s")
    print(f"u_op_topo ≈ {st.u_op:.2f} m/s")
    print(f"D_topo (se isolado) ≈ {st.D:.2f} m")

    print("\n--- Seção de Fundo ---")
    sf = sizing.sec_fundo
    print(f"V_fundo = {sf.V_kmol_h:.1f} kmol/h")
    print(f"MW_vap_fundo ≈ {sf.MW_vap:.1f} kg/kmol")
    print(f"rho_vap_fundo ≈ {sf.rho_vap:.2f} kg/m3")
    print(f"u_flood_fundo ≈ {sf.u_flood:.2f} m/s")
    print(f"u_op_fundo ≈ {sf.u_op:.2f} m/s")
    print(f"D_fundo (se isolado) ≈ {sf.D:.2f} m")

    print("\n=== ETAPA 7 – COLUNA RECHEADA (INTALOX SADDLES 1 in) ===")

    packed = dimensionar_coluna_recheada_intalox(
        spec=spec,
        RR=RR,
        comps=comps,
        N_teorico=Nteo,
        P_atm=P,
        T_top_K=370.0,   # chute de T de topo (pode ajustar depois)
        rho_L=630.0,     # chute de densidade líquida de topo
        mu_L_cP=0.5,     # chute de viscosidade do líquido
        phi=0.70,        # 70% da inundação
    )

    print(f"MM média no topo ≈ {packed.MM_top:.2f} kg/kmol")
    print(f"ρ_V(top) ≈ {packed.rho_V:.2f} kg/m³, ρ_L(chute) = {packed.rho_L:.1f} kg/m³")
    print(f"F_LV ≈ {packed.FLV:.3f}, Y (Leva) ≈ {packed.Y:.3f}")
    print(f"u_V,flood ≈ {packed.uV_flood_m_s:.3f} m/s, u_V,op ≈ {packed.uV_op_m_s:.3f} m/s")
    print(f"Área ≈ {packed.A_m2:.2f} m²  →  D_recheada ≈ {packed.D_m:.2f} m")
    print(f"Altura de recheio (HETP) ≈ {packed.H_recheio_m:.2f} m")
    print(f"Altura total sugerida ≈ {packed.H_total_m:.2f} m")


    print("\n=== ETAPA 8 – ANÁLISE DE DIFERENTES RAZÕES DE REFLUXO ===")
    print("f_R   R_R    N_teo   N_real   N_pratos   D_coluna(m)   H_total(m)")

    fatores_R = [1.10, 1.30, 1.50, 2.00]

    for fR in fatores_R:
        RR_test = fR * RR_min
        Nteo_test = gilliland_N(Nmin, RR_test, RR_min)
        N_real_test = calcular_N_real(Nteo_test, eta_G)
        N_pratos_test = math.ceil(N_real_test)

        sizing_test = dimensionar_coluna_pratos(
            N_pratos=N_pratos_test,
            F=F,
            q_liq=q_liq,
            spec=spec,
            RR=RR_test,
            comps=comps,
            P_atm=P,
        )

        print(
            f"{fR:4.2f}  {RR_test:6.3f}  {Nteo_test:7.2f}  {N_real_test:7.2f}  "
            f"{N_pratos_test:8d}    {sizing_test.diametro:8.2f}   {sizing_test.H_total:8.2f}"
        )



if __name__ == "__main__":
    main()
