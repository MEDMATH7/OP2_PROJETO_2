from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox

from componentes import carregar_componentes, Componente
from especificacao import definir_especificacao_por_recuperacoes
from fug import (
    fenske_Nmin,
    underwood_RRmin,
    gilliland_N,
    fenske_NR_NS_min,
    calcular_prato_otimo,
)
from eficiencia import estimar_mu_feed_cP, oconnell_eta, calcular_N_real
from dimensionamento_pratos import (
    SecaoPratos,
    TrayColumnSizing,
    dimensionar_secao,
    altura_coluna,
)
from dimensionamento_recheios import dimensionar_coluna_recheada_intalox

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure



# =========================
#   PARÂMETROS PADRÃO
# =========================

DEFAULT_F = 1000.0
DEFAULT_Z = [0.05, 0.10, 0.25, 0.30, 0.30]  # [n-C5, n-C6, n-C7, n-C9, n-C10]
DEFAULT_FRAC_VAPOR = 0.20
DEFAULT_P_ATM = 2.0

# fator de refluxo (R = f_R * Rmin)
DEFAULT_FATOR_R = 1.30
DEFAULT_FATORES_R_SWEEP = "1.10, 1.30, 1.50, 2.00"

# volatilidades relativas (mesmas do projeto)
DEFAULT_ALPHA = np.array([3.0, 2.3, 1.8, 1.3, 1.0])

# "dados de HYSYS" / chutes para pratos
DEFAULT_T_TOP_TRAY = 370.0    # K
DEFAULT_T_BOT_TRAY = 430.0    # K
DEFAULT_RHO_L_TOP = 650.0     # kg/m3
DEFAULT_RHO_L_BOT = 700.0     # kg/m3
DEFAULT_FRAC_FLOOD_TRAY = 0.75
DEFAULT_FRAC_AREA_ATIVA = 0.80
DEFAULT_C_CAPACIDADE = 0.15

# dados para coluna recheada
DEFAULT_T_TOP_PACKED = 370.0   # K
DEFAULT_RHO_L_PACKED = 630.0   # kg/m3
DEFAULT_MU_L_PACKED = 0.5      # cP
DEFAULT_PHI_PACKED = 0.70      # fração de inundação


@dataclass
class ResultadosProjeto:
    # dados básicos
    F: float
    z: List[float]
    fracao_vapor: float
    q_liq: float
    P: float
    comps: List[Componente]

    # especificação
    spec: object  # SeparationSpec

    # FUG / O'Connell
    alpha: np.ndarray
    Nmin: float
    RR_min: float
    theta: float
    RR_oper: float
    Nteo: float
    mu_F_cP: float
    alpha_rel: float
    eta_G: float
    N_real: float
    N_pratos: int
    NR_min: float
    NS_min: float
    NR_op: float
    NS_op: float
    j_feed_teo: float
    j_feed_real: float
    j_feed_real_int: int

    # coluna de pratos
    tray_sizing: TrayColumnSizing

    # coluna recheada
    packed_sizing: object  # PackedColumnSizing

    # sweep de refluxo
    reflux_table: List[dict]


# =========================
#   FUNÇÕES DE CÁLCULO
# =========================

def normalizar_composicao(z: Sequence[float]) -> List[float]:
    z_arr = np.array(z, dtype=float)
    soma = float(z_arr.sum())
    if soma <= 0:
        raise ValueError("Soma das frações molares z é zero ou negativa.")
    return list((z_arr / soma).tolist())


def rodar_calculos(
    F: float,
    z: List[float],
    fracao_vapor: float,
    P: float,
    alpha: np.ndarray,
    fator_R_oper: float,
    fatores_R_sweep: List[float],
    # parâmetros "HYSYS" / chutes:
    T_top_tray: float,
    T_bot_tray: float,
    rho_L_top: float,
    rho_L_bot: float,
    frac_flood_tray: float,
    frac_area_ativa: float,
    C_capacidade: float,
    T_top_packed: float,
    rho_L_packed: float,
    mu_L_packed: float,
    phi_packed: float,
) -> ResultadosProjeto:

    base_dir = Path(__file__).resolve().parent.parent
    caminho_componentes = base_dir / "data" / "componentes.csv"
    comps = carregar_componentes(caminho_componentes)

    # garante que z soma 1
    z_norm = normalizar_composicao(z)
    q_liq = 1.0 - fracao_vapor

    # === Especificação ===
    spec = definir_especificacao_por_recuperacoes(F, z_norm)

    i_LK = spec.LK_index
    i_HK = spec.HK_index

    # === FUG ===
    Nmin = fenske_Nmin(spec.xD, spec.xB, i_LK, i_HK, alpha)
    RR_min, theta = underwood_RRmin(alpha, spec.xD, spec.zF, q_liq, i_LK, i_HK)

    RR_oper = fator_R_oper * RR_min
    Nteo = gilliland_N(Nmin, RR_oper, RR_min)

    # === O'Connell ===
    mu_F_cP = estimar_mu_feed_cP(spec.zF)
    alpha_rel = alpha[i_LK] / alpha[i_HK]
    eta_G = oconnell_eta(alpha_rel, mu_F_cP)
    N_real = calcular_N_real(Nteo, eta_G)
    N_pratos = math.ceil(N_real)

    # === Prato ótimo ===
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

    j_feed_real_int = math.ceil(j_feed_real)

    # === Coluna de pratos (usando dimensionar_secao + altura_coluna com T/rho custom) ===
    D_total = spec.D
    F_total = F

    L_R = RR_oper * D_total
    V_R = (RR_oper + 1.0) * D_total

    L_S = L_R + q_liq * F_total
    V_S = V_R + (1.0 - q_liq) * F_total

    sec_topo = dimensionar_secao(
        "topo",
        L_kmol_h=L_R,
        V_kmol_h=V_R,
        x_vap=spec.xD,
        comps=comps,
        P_atm=P,
        T_K=T_top_tray,
        rho_L_assumida=rho_L_top,
        frac_flood=frac_flood_tray,
        frac_area_ativa=frac_area_ativa,
        C_capacidade=C_capacidade,
    )

    sec_fundo = dimensionar_secao(
        "fundo",
        L_kmol_h=L_S,
        V_kmol_h=V_S,
        x_vap=spec.xB,
        comps=comps,
        P_atm=P,
        T_K=T_bot_tray,
        rho_L_assumida=rho_L_bot,
        frac_flood=frac_flood_tray,
        frac_area_ativa=frac_area_ativa,
        C_capacidade=C_capacidade,
    )

    D_coluna = max(sec_topo.D, sec_fundo.D)
    H_ativa, H_total = altura_coluna(N_pratos)

    tray_sizing = TrayColumnSizing(
        N_pratos=N_pratos,
        H_ativa=H_ativa,
        H_total=H_total,
        diametro=D_coluna,
        sec_topo=sec_topo,
        sec_fundo=sec_fundo,
    )

    # === Coluna recheada ===
    packed_sizing = dimensionar_coluna_recheada_intalox(
        spec=spec,
        RR=RR_oper,
        comps=comps,
        N_teorico=Nteo,
        P_atm=P,
        T_top_K=T_top_packed,
        rho_L=rho_L_packed,
        mu_L_cP=mu_L_packed,
        phi=phi_packed,
    )

    # === Análise de refluxo ===
    reflux_table: List[dict] = []
    for fR in fatores_R_sweep:
        RR_test = fR * RR_min
        Nteo_test = gilliland_N(Nmin, RR_test, RR_min)
        N_real_test = calcular_N_real(Nteo_test, eta_G)
        N_pratos_test = math.ceil(N_real_test)

        # dimensionar coluna de pratos para cada R
        L_R_t = RR_test * D_total
        V_R_t = (RR_test + 1.0) * D_total
        L_S_t = L_R_t + q_liq * F_total
        V_S_t = V_R_t + (1.0 - q_liq) * F_total

        sec_topo_t = dimensionar_secao(
            "topo",
            L_kmol_h=L_R_t,
            V_kmol_h=V_R_t,
            x_vap=spec.xD,
            comps=comps,
            P_atm=P,
            T_K=T_top_tray,
            rho_L_assumida=rho_L_top,
            frac_flood=frac_flood_tray,
            frac_area_ativa=frac_area_ativa,
            C_capacidade=C_capacidade,
        )
        sec_fundo_t = dimensionar_secao(
            "fundo",
            L_kmol_h=L_S_t,
            V_kmol_h=V_S_t,
            x_vap=spec.xB,
            comps=comps,
            P_atm=P,
            T_K=T_bot_tray,
            rho_L_assumida=rho_L_bot,
            frac_flood=frac_flood_tray,
            frac_area_ativa=frac_area_ativa,
            C_capacidade=C_capacidade,
        )
        D_col_t = max(sec_topo_t.D, sec_fundo_t.D)
        _, H_total_t = altura_coluna(N_pratos_test)

        reflux_table.append(
            dict(
                f_R=fR,
                RR=RR_test,
                Nteo=Nteo_test,
                N_real=N_real_test,
                N_pratos=N_pratos_test,
                D_col=D_col_t,
                H_total=H_total_t,
            )
        )

    return ResultadosProjeto(
        F=F,
        z=z_norm,
        fracao_vapor=fracao_vapor,
        q_liq=q_liq,
        P=P,
        comps=comps,
        spec=spec,
        alpha=alpha,
        Nmin=Nmin,
        RR_min=RR_min,
        theta=theta,
        RR_oper=RR_oper,
        Nteo=Nteo,
        mu_F_cP=mu_F_cP,
        alpha_rel=alpha_rel,
        eta_G=eta_G,
        N_real=N_real,
        N_pratos=N_pratos,
        NR_min=NR_min,
        NS_min=NS_min,
        NR_op=NR_op,
        NS_op=NS_op,
        j_feed_teo=j_feed_teo,
        j_feed_real=j_feed_real,
        j_feed_real_int=j_feed_real_int,
        tray_sizing=tray_sizing,
        packed_sizing=packed_sizing,
        reflux_table=reflux_table,
    )


# =========================
#   GUI
# =========================

class ProjetoApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Projeto Destilação Multicomponente – OP2")
        self.geometry("980x640")

        # estilo bonito
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 12, "bold"))
        style.configure("Card.TLabelframe", background="#f5f5f5")
        style.configure("Card.TLabelframe.Label", font=("Segoe UI", 11, "bold"))

        self._build_widgets()

        self.resultados: ResultadosProjeto | None = None

    # -------- helpers --------

    def _update_text(self, widget: tk.Text, text: str) -> None:
        widget.configure(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.configure(state=tk.DISABLED)

    def _parse_float(self, entry: ttk.Entry, default: float) -> float:
        s = entry.get().strip()
        if not s:
            return default
        try:
            return float(s)
        except ValueError:
            messagebox.showwarning(
                "Entrada inválida",
                f"Valor '{s}' inválido, usando padrão {default}.",
                parent=self,
            )
            return default

    def _parse_fatores_R(self, s: str) -> List[float]:
        valores: List[float] = []
        for parte in s.split(","):
            p = parte.strip()
            if not p:
                continue
            try:
                valores.append(float(p))
            except ValueError:
                messagebox.showwarning(
                    "Entrada inválida",
                    f"Fator de refluxo '{p}' ignorado (não numérico).",
                    parent=self,
                )
        if not valores:
            valores = [1.10, 1.30, 1.50, 2.00]
        return valores

    def _build_plots_tab(self) -> None:
        frame = self.tab_plots
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        # Figura matplotlib com 2x2 subplots
        self.fig = Figure(figsize=(7.5, 5.5), dpi=100)
        self.ax_r_vs_N = self.fig.add_subplot(2, 2, 1)
        self.ax_r_vs_H = self.fig.add_subplot(2, 2, 2)
        self.ax_diametros = self.fig.add_subplot(2, 2, 3)
        self.ax_alturas = self.fig.add_subplot(2, 2, 4)

        self.canvas = FigureCanvasTkAgg(self.fig, master=frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        # Botão para atualizar graficamente (caso você mude algo sem recalcular)
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=1, column=0, sticky="e", padx=8, pady=4)
        btn_save = ttk.Button(
            btn_frame,
            text="Salvar figura como JPG",
            command=self._save_figure,
        )
        btn_save.pack(side=tk.RIGHT, padx=6)

        btn_update = ttk.Button(
            btn_frame,
            text="Atualizar gráficos",
            command=self._update_plots,
        )
        btn_update.pack(side=tk.RIGHT)

    def _update_plots(self) -> None:
        r = self.resultados
        if r is None:
            # nada pra plotar ainda
            return

        # Limpa todos os eixos
        self.ax_r_vs_N.clear()
        self.ax_r_vs_H.clear()
        self.ax_diametros.clear()
        self.ax_alturas.clear()

        # --------- 1) R_R vs N_pratos (a partir da tabela de refluxo) ---------
        Rs = [row["RR"] for row in r.reflux_table]
        Nps = [row["N_pratos"] for row in r.reflux_table]

        self.ax_r_vs_N.plot(Rs, Nps, marker="o")
        self.ax_r_vs_N.set_xlabel("R_R [-]")
        self.ax_r_vs_N.set_ylabel("N_pratos [-]")
        self.ax_r_vs_N.set_title("Razão de refluxo vs número de pratos")
        self.ax_r_vs_N.grid(True, alpha=0.3)

        # --------- 2) R_R vs H_total (coluna de pratos) ---------
        Hs = [row["H_total"] for row in r.reflux_table]

        self.ax_r_vs_H.plot(Rs, Hs, marker="o")
        self.ax_r_vs_H.set_xlabel("R_R [-]")
        self.ax_r_vs_H.set_ylabel("H_total [m]")
        self.ax_r_vs_H.set_title("Razão de refluxo vs altura da coluna (pratos)")
        self.ax_r_vs_H.grid(True, alpha=0.3)

        # --------- 3) Barras – diâmetro pratos vs recheada ---------
        labels = ["Pratos", "Recheada"]
        diametros = [r.tray_sizing.diametro, r.packed_sizing.D_m]

        self.ax_diametros.bar(labels, diametros)
        self.ax_diametros.set_ylabel("Diâmetro [m]")
        self.ax_diametros.set_title("Comparação de diâmetros")

        for i, val in enumerate(diametros):
            self.ax_diametros.text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

        # --------- 4) Barras – altura pratos vs recheada ---------
        alturas = [r.tray_sizing.H_total, r.packed_sizing.H_total_m]

        self.ax_alturas.bar(labels, alturas)
        self.ax_alturas.set_ylabel("Altura [m]")
        self.ax_alturas.set_title("Comparação de alturas")

        for i, val in enumerate(alturas):
            self.ax_alturas.text(i, val, f"{val:.2f}", ha="center", va="bottom", fontsize=9)

        self.fig.tight_layout()
        self.canvas.draw()


    def _save_figure(self) -> None:
        if not hasattr(self, "fig"):
            return
        try:
            self.fig.savefig("graficos_projeto.jpg", dpi=300, bbox_inches="tight")
            messagebox.showinfo(
                "Figura salva",
                "Arquivo 'graficos_projeto.jpg' salvo na pasta do script.",
                parent=self,
            )
        except Exception as e:
            messagebox.showerror(
                "Erro ao salvar figura",
                f"Ocorreu um erro ao salvar a figura:\n{e}",
                parent=self,
            )


    # -------- layout --------

    def _build_widgets(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # abas
        self.tab_input = ttk.Frame(notebook)
        self.tab_result_resumo = ttk.Frame(notebook)
        self.tab_result_refluxo = ttk.Frame(notebook)
        self.tab_plots = ttk.Frame(notebook)  # NOVA aba

        notebook.add(self.tab_input, text="Entrada de dados")
        notebook.add(self.tab_result_resumo, text="Resultados – Resumo")
        notebook.add(self.tab_result_refluxo, text="Resultados – Refluxo")
        notebook.add(self.tab_plots, text="Gráficos")  # adiciona aqui

        self._build_input_tab()
        self._build_result_resumo_tab()
        self._build_result_refluxo_tab()
        self._build_plots_tab()  # NOVO


    def _build_input_tab(self) -> None:
        # frame principal
        frame = self.tab_input
        frame.columnconfigure(0, weight=1)
        frame.columnconfigure(1, weight=1)

        # --- Alimentação ---
        lf_feed = ttk.LabelFrame(frame, text="Dados da Alimentação", style="Card.TLabelframe")
        lf_feed.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        # F
        ttk.Label(lf_feed, text="F [kmol/h]:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.ent_F = ttk.Entry(lf_feed, width=12)
        self.ent_F.insert(0, str(DEFAULT_F))
        self.ent_F.grid(row=0, column=1, sticky="w", padx=4, pady=4)

        # z
        ttk.Label(lf_feed, text="z (n-C5, n-C6, n-C7, n-C9, n-C10):").grid(
            row=1, column=0, sticky="w", padx=4, pady=4
        )
        z_frame = ttk.Frame(lf_feed)
        z_frame.grid(row=1, column=1, sticky="w", padx=4, pady=4)
        self.ent_z = []
        for i, val in enumerate(DEFAULT_Z):
            e = ttk.Entry(z_frame, width=6)
            e.insert(0, f"{val:.3f}")
            e.grid(row=0, column=i, padx=2, pady=2)
            self.ent_z.append(e)

        ttk.Label(lf_feed, text="Fração vaporizada (feed):").grid(
            row=2, column=0, sticky="w", padx=4, pady=4
        )
        self.ent_frac_vapor = ttk.Entry(lf_feed, width=12)
        self.ent_frac_vapor.insert(0, f"{DEFAULT_FRAC_VAPOR:.2f}")
        self.ent_frac_vapor.grid(row=2, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(lf_feed, text="Pressão da coluna [atm]:").grid(
            row=3, column=0, sticky="w", padx=4, pady=4
        )
        self.ent_P = ttk.Entry(lf_feed, width=12)
        self.ent_P.insert(0, f"{DEFAULT_P_ATM:.2f}")
        self.ent_P.grid(row=3, column=1, sticky="w", padx=4, pady=4)

        # --- Refluxo ---
        lf_refluxo = ttk.LabelFrame(frame, text="Razão de Refluxo", style="Card.TLabelframe")
        lf_refluxo.grid(row=0, column=1, sticky="nsew", padx=8, pady=8)

        ttk.Label(lf_refluxo, text="f_R (R = f_R · Rmin):").grid(
            row=0, column=0, sticky="w", padx=4, pady=4
        )
        self.ent_fator_R = ttk.Entry(lf_refluxo, width=12)
        self.ent_fator_R.insert(0, f"{DEFAULT_FATOR_R:.2f}")
        self.ent_fator_R.grid(row=0, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(lf_refluxo, text="fatores para análise (separados por vírgula):").grid(
            row=1, column=0, sticky="w", padx=4, pady=4
        )
        self.ent_fatores_R = ttk.Entry(lf_refluxo, width=32)
        self.ent_fatores_R.insert(0, DEFAULT_FATORES_R_SWEEP)
        self.ent_fatores_R.grid(row=1, column=1, sticky="w", padx=4, pady=4)

        # --- Propriedades (HYSYS / chutes) ---
        lf_props = ttk.LabelFrame(
            frame, text="Propriedades (HYSYS / chutes)", style="Card.TLabelframe"
        )
        lf_props.grid(row=1, column=0, columnspan=2, sticky="nsew", padx=8, pady=8)
        for c in range(4):
            lf_props.columnconfigure(c, weight=1)

        ttk.Label(lf_props, text="Pratos – T_top [K]:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.ent_T_top_tray = ttk.Entry(lf_props, width=10)
        self.ent_T_top_tray.insert(0, f"{DEFAULT_T_TOP_TRAY:.1f}")
        self.ent_T_top_tray.grid(row=0, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(lf_props, text="Pratos – T_bot [K]:").grid(row=0, column=2, sticky="w", padx=4, pady=4)
        self.ent_T_bot_tray = ttk.Entry(lf_props, width=10)
        self.ent_T_bot_tray.insert(0, f"{DEFAULT_T_BOT_TRAY:.1f}")
        self.ent_T_bot_tray.grid(row=0, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(lf_props, text="Pratos – ρ_L,top [kg/m³]:").grid(
            row=1, column=0, sticky="w", padx=4, pady=4
        )
        self.ent_rho_L_top = ttk.Entry(lf_props, width=10)
        self.ent_rho_L_top.insert(0, f"{DEFAULT_RHO_L_TOP:.1f}")
        self.ent_rho_L_top.grid(row=1, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(lf_props, text="Pratos – ρ_L,bot [kg/m³]:").grid(
            row=1, column=2, sticky="w", padx=4, pady=4
        )
        self.ent_rho_L_bot = ttk.Entry(lf_props, width=10)
        self.ent_rho_L_bot.insert(0, f"{DEFAULT_RHO_L_BOT:.1f}")
        self.ent_rho_L_bot.grid(row=1, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(lf_props, text="Pratos – fração de inundação:").grid(
            row=2, column=0, sticky="w", padx=4, pady=4
        )
        self.ent_frac_flood = ttk.Entry(lf_props, width=10)
        self.ent_frac_flood.insert(0, f"{DEFAULT_FRAC_FLOOD_TRAY:.2f}")
        self.ent_frac_flood.grid(row=2, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(lf_props, text="Pratos – fração área ativa:").grid(
            row=2, column=2, sticky="w", padx=4, pady=4
        )
        self.ent_frac_area_ativa = ttk.Entry(lf_props, width=10)
        self.ent_frac_area_ativa.insert(0, f"{DEFAULT_FRAC_AREA_ATIVA:.2f}")
        self.ent_frac_area_ativa.grid(row=2, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(lf_props, text="Pratos – C_capacidade:").grid(
            row=3, column=0, sticky="w", padx=4, pady=4
        )
        self.ent_C_capacidade = ttk.Entry(lf_props, width=10)
        self.ent_C_capacidade.insert(0, f"{DEFAULT_C_CAPACIDADE:.3f}")
        self.ent_C_capacidade.grid(row=3, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(lf_props, text="Recheio – T_top [K]:").grid(
            row=4, column=0, sticky="w", padx=4, pady=4
        )
        self.ent_T_top_packed = ttk.Entry(lf_props, width=10)
        self.ent_T_top_packed.insert(0, f"{DEFAULT_T_TOP_PACKED:.1f}")
        self.ent_T_top_packed.grid(row=4, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(lf_props, text="Recheio – ρ_L [kg/m³]:").grid(
            row=4, column=2, sticky="w", padx=4, pady=4
        )
        self.ent_rho_L_packed = ttk.Entry(lf_props, width=10)
        self.ent_rho_L_packed.insert(0, f"{DEFAULT_RHO_L_PACKED:.1f}")
        self.ent_rho_L_packed.grid(row=4, column=3, sticky="w", padx=4, pady=4)

        ttk.Label(lf_props, text="Recheio – μ_L [cP]:").grid(
            row=5, column=0, sticky="w", padx=4, pady=4
        )
        self.ent_mu_L_packed = ttk.Entry(lf_props, width=10)
        self.ent_mu_L_packed.insert(0, f"{DEFAULT_MU_L_PACKED:.3f}")
        self.ent_mu_L_packed.grid(row=5, column=1, sticky="w", padx=4, pady=4)

        ttk.Label(lf_props, text="Recheio – φ (fração inundação):").grid(
            row=5, column=2, sticky="w", padx=4, pady=4
        )
        self.ent_phi_packed = ttk.Entry(lf_props, width=10)
        self.ent_phi_packed.insert(0, f"{DEFAULT_PHI_PACKED:.2f}")
        self.ent_phi_packed.grid(row=5, column=3, sticky="w", padx=4, pady=4)

        # --- Botão Rodar ---
        btn_run = ttk.Button(
            frame, text="Rodar cálculos", command=self.on_run_clicked
        )
        btn_run.grid(row=2, column=0, columnspan=2, pady=12)

    def _build_result_resumo_tab(self) -> None:
        frame = self.tab_result_resumo
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        self.txt_resumo = tk.Text(
            frame,
            wrap=tk.WORD,
            font=("Consolas", 10),
        )
        self.txt_resumo.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        scroll = ttk.Scrollbar(
            frame, orient=tk.VERTICAL, command=self.txt_resumo.yview
        )
        scroll.grid(row=0, column=1, sticky="ns")
        self.txt_resumo.configure(yscrollcommand=scroll.set)

        self.txt_resumo.configure(state=tk.DISABLED)

    def _build_result_refluxo_tab(self) -> None:
        frame = self.tab_result_refluxo
        frame.rowconfigure(0, weight=1)
        frame.columnconfigure(0, weight=1)

        columns = ("f_R", "RR", "Nteo", "N_real", "N_pratos", "D_col", "H_total")
        self.tree_refluxo = ttk.Treeview(
            frame,
            columns=columns,
            show="headings",
            height=16,
        )
        headings = [
            "f_R",
            "R_R",
            "N_teo",
            "N_real",
            "N_pratos",
            "D [m]",
            "H_total [m]",
        ]
        for col, head in zip(columns, headings):
            self.tree_refluxo.heading(col, text=head)
            self.tree_refluxo.column(col, anchor="center", width=100)

        self.tree_refluxo.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        scroll = ttk.Scrollbar(
            frame, orient=tk.VERTICAL, command=self.tree_refluxo.yview
        )
        scroll.grid(row=0, column=1, sticky="ns")
        self.tree_refluxo.configure(yscrollcommand=scroll.set)

    # -------- callbacks --------

    def on_run_clicked(self) -> None:
        try:
            F = self._parse_float(self.ent_F, DEFAULT_F)

            z_vals = []
            all_empty = True
            for ent, z_def in zip(self.ent_z, DEFAULT_Z):
                s = ent.get().strip()
                if s:
                    all_empty = False
                    try:
                        z_vals.append(float(s))
                    except ValueError:
                        raise ValueError(f"Composição inválida: '{s}'")
                else:
                    z_vals.append(z_def)
            if all_empty:
                z_vals = DEFAULT_Z.copy()

            frac_vapor = self._parse_float(
                self.ent_frac_vapor, DEFAULT_FRAC_VAPOR
            )
            if not (0.0 <= frac_vapor <= 1.0):
                raise ValueError("Fração vaporizada deve estar entre 0 e 1.")

            P = self._parse_float(self.ent_P, DEFAULT_P_ATM)

            fator_R_oper = self._parse_float(
                self.ent_fator_R, DEFAULT_FATOR_R
            )

            fatores_R = self._parse_fatores_R(self.ent_fatores_R.get())

            T_top_tray = self._parse_float(
                self.ent_T_top_tray, DEFAULT_T_TOP_TRAY
            )
            T_bot_tray = self._parse_float(
                self.ent_T_bot_tray, DEFAULT_T_BOT_TRAY
            )
            rho_L_top = self._parse_float(
                self.ent_rho_L_top, DEFAULT_RHO_L_TOP
            )
            rho_L_bot = self._parse_float(
                self.ent_rho_L_bot, DEFAULT_RHO_L_BOT
            )
            frac_flood_tray = self._parse_float(
                self.ent_frac_flood, DEFAULT_FRAC_FLOOD_TRAY
            )
            frac_area_ativa = self._parse_float(
                self.ent_frac_area_ativa, DEFAULT_FRAC_AREA_ATIVA
            )
            C_capacidade = self._parse_float(
                self.ent_C_capacidade, DEFAULT_C_CAPACIDADE
            )

            T_top_packed = self._parse_float(
                self.ent_T_top_packed, DEFAULT_T_TOP_PACKED
            )
            rho_L_packed = self._parse_float(
                self.ent_rho_L_packed, DEFAULT_RHO_L_PACKED
            )
            mu_L_packed = self._parse_float(
                self.ent_mu_L_packed, DEFAULT_MU_L_PACKED
            )
            phi_packed = self._parse_float(
                self.ent_phi_packed, DEFAULT_PHI_PACKED
            )

            resultados = rodar_calculos(
                F=F,
                z=z_vals,
                fracao_vapor=frac_vapor,
                P=P,
                alpha=DEFAULT_ALPHA,
                fator_R_oper=fator_R_oper,
                fatores_R_sweep=fatores_R,
                T_top_tray=T_top_tray,
                T_bot_tray=T_bot_tray,
                rho_L_top=rho_L_top,
                rho_L_bot=rho_L_bot,
                frac_flood_tray=frac_flood_tray,
                frac_area_ativa=frac_area_ativa,
                C_capacidade=C_capacidade,
                T_top_packed=T_top_packed,
                rho_L_packed=rho_L_packed,
                mu_L_packed=mu_L_packed,
                phi_packed=phi_packed,
            )

        except Exception as e:
            messagebox.showerror(
                "Erro nos cálculos",
                f"Ocorreu um erro:\n{e}",
                parent=self,
            )
            return

        self.resultados = resultados
        self._preencher_resumo()
        self._preencher_refluxo()
        self._update_plots()
        messagebox.showinfo(
            "Cálculos concluídos",
            "Cálculos executados com sucesso.",
            parent=self,
        )

    # -------- preencher resultados --------

    def _preencher_resumo(self) -> None:
        r = self.resultados
        if r is None:
            return

        lines = []

        lines.append("=== Dados de Entrada ===")
        lines.append(f"F = {r.F:.2f} kmol/h")
        lines.append(
            "z = [n-C5, n-C6, n-C7, n-C9, n-C10] = "
            + "[" + ", ".join(f"{zi:.3f}" for zi in r.z) + "]"
        )
        lines.append(
            f"Fração vaporizada = {r.fracao_vapor:.2f}  ->  q = {r.q_liq:.2f}"
        )
        lines.append(f"P = {r.P:.2f} atm")
        lines.append("")

        lines.append("=== Especificação de Separação ===")
        lines.append(f"D = {r.spec.D:.2f} kmol/h, B = {r.spec.B:.2f} kmol/h")
        lines.append(
            f"LK = índice {r.spec.LK_index} ({r.comps[r.spec.LK_index].nome}), "
            f"HK = índice {r.spec.HK_index} ({r.comps[r.spec.HK_index].nome})"
        )
        lines.append("")
        lines.append("Comp    Fi       R_D      Di       Bi      xD       xB")
        for i, c in enumerate(r.comps):
            lines.append(
                f"{i+1:>3}  {c.nome:<10} "
                f"{r.spec.F_i[i]:7.2f}  "
                f"{r.spec.R_D[i]:7.4f}  "
                f"{r.spec.D_i[i]:7.2f}  "
                f"{r.spec.B_i[i]:7.2f}  "
                f"{r.spec.xD[i]:7.4f}  "
                f"{r.spec.xB[i]:7.4f}"
            )
        lines.append("")

        lines.append("=== Método FUG ===")
        lines.append(f"N_min (Fenske) = {r.Nmin:.2f}")
        lines.append(f"R_R,min (Underwood) = {r.RR_min:.3f}")
        lines.append(f"θ (Underwood) = {r.theta:.4f}")
        lines.append(f"f_R (operação) = {r.RR_oper / r.RR_min:.3f}")
        lines.append(f"R_R (operação) = {r.RR_oper:.3f}")
        lines.append(f"N_teo (Gilliland) = {r.Nteo:.2f}")
        lines.append("")

        lines.append("=== Eficiência Global (O'Connell) ===")
        lines.append(f"μ_F ≈ {r.mu_F_cP:.3f} cP")
        lines.append(f"α_rel (LK/HK) ≈ {r.alpha_rel:.3f}")
        lines.append(f"η_G ≈ {r.eta_G:.3f}  ->  {r.eta_G*100:.1f} %")
        lines.append(f"N_real ≈ {r.N_real:.2f}")
        lines.append(f"N_pratos adotado = {r.N_pratos}")
        lines.append("")

        lines.append("=== Prato Ótimo de Alimentação ===")
        lines.append(f"NR_min ≈ {r.NR_min:.2f}, NS_min ≈ {r.NS_min:.2f}")
        lines.append(f"NR_op ≈ {r.NR_op:.2f}, NS_op ≈ {r.NS_op:.2f}")
        lines.append(f"j_feed,teo ≈ {r.j_feed_teo:.1f} (contando do topo)")
        lines.append(
            f"Prato de alimentação na coluna real ≈ prato {r.j_feed_real_int} "
            f"de {r.N_pratos} (contando do topo)"
        )
        lines.append("")

        lines.append("=== Coluna de Pratos Válvulados ===")
        lines.append(f"D_coluna ≈ {r.tray_sizing.diametro:.2f} m")
        lines.append(f"H_ativa ≈ {r.tray_sizing.H_ativa:.1f} m")
        lines.append(f"H_total ≈ {r.tray_sizing.H_total:.1f} m")
        lines.append("")
        st = r.tray_sizing.sec_topo
        lines.append("--- Seção de Topo ---")
        lines.append(f"V_topo = {st.V_kmol_h:.1f} kmol/h")
        lines.append(f"MW_vap_topo ≈ {st.MW_vap:.1f} kg/kmol")
        lines.append(f"ρ_vap_topo ≈ {st.rho_vap:.2f} kg/m³")
        lines.append(f"ρ_liq_topo (chute) ≈ {st.rho_liq:.1f} kg/m³")
        lines.append(f"u_flood_topo ≈ {st.u_flood:.2f} m/s")
        lines.append(f"u_op_topo ≈ {st.u_op:.2f} m/s")
        lines.append(f"D_topo (se isolado) ≈ {st.D:.2f} m")
        lines.append("")
        sf = r.tray_sizing.sec_fundo
        lines.append("--- Seção de Fundo ---")
        lines.append(f"V_fundo = {sf.V_kmol_h:.1f} kmol/h")
        lines.append(f"MW_vap_fundo ≈ {sf.MW_vap:.1f} kg/kmol")
        lines.append(f"ρ_vap_fundo ≈ {sf.rho_vap:.2f} kg/m³")
        lines.append(f"ρ_liq_fundo (chute) ≈ {sf.rho_liq:.1f} kg/m³")
        lines.append(f"u_flood_fundo ≈ {sf.u_flood:.2f} m/s")
        lines.append(f"u_op_fundo ≈ {sf.u_op:.2f} m/s")
        lines.append(f"D_fundo (se isolado) ≈ {sf.D:.2f} m")
        lines.append("")

        lines.append("=== Coluna Recheada (Intalox Saddles 1 in) ===")
        p = r.packed_sizing
        lines.append(f"MM_top ≈ {p.MM_top:.2f} kg/kmol")
        lines.append(f"ρ_V(top) ≈ {p.rho_V:.2f} kg/m³")
        lines.append(f"ρ_L (chute) ≈ {p.rho_L:.1f} kg/m³")
        lines.append(f"F_LV ≈ {p.FLV:.3f}")
        lines.append(f"Y (Leva) ≈ {p.Y:.3f}")
        lines.append(f"u_V,flood ≈ {p.uV_flood_m_s:.3f} m/s")
        lines.append(f"u_V,op ≈ {p.uV_op_m_s:.3f} m/s")
        lines.append(f"A_seção ≈ {p.A_m2:.2f} m²")
        lines.append(f"D_recheada ≈ {p.D_m:.2f} m")
        lines.append(f"H_recheio ≈ {p.H_recheio_m:.2f} m")
        lines.append(f"H_total sugerida ≈ {p.H_total_m:.2f} m")
        lines.append("")

        lines.append("Obs.: Todos os resultados foram gerados a partir do código Python "
                     "organizado no repositório OP2_PROJETO_2.")

        texto = "\n".join(lines)
        self._update_text(self.txt_resumo, texto)

    def _preencher_refluxo(self) -> None:
        self.tree_refluxo.delete(*self.tree_refluxo.get_children())
        r = self.resultados
        if r is None:
            return

        for row in r.reflux_table:
            self.tree_refluxo.insert(
                "",
                tk.END,
                values=(
                    f"{row['f_R']:.2f}",
                    f"{row['RR']:.3f}",
                    f"{row['Nteo']:.2f}",
                    f"{row['N_real']:.2f}",
                    f"{row['N_pratos']:d}",
                    f"{row['D_col']:.2f}",
                    f"{row['H_total']:.2f}",
                ),
            )


if __name__ == "__main__":
    app = ProjetoApp()
    app.mainloop()


