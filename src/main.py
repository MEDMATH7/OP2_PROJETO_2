# src/main.py
from __future__ import annotations

from pathlib import Path

from componentes import carregar_componentes
from especificacao import definir_especificacao_inicial


def definir_alimentacao():
    """
    Define as condicoes da corrente de alimentacao segundo o enunciado:
    """
    F = 1000.0  # kmol/h
    z = [0.05, 0.10, 0.25, 0.30, 0.30]
    q = 0.2
    P = 2.0     # atm
    return F, z, q, P


def main():
    # 1) Carregar componentes
    base_dir = Path(__file__).resolve().parent.parent
    caminho_componentes = base_dir / "data" / "componentes.csv"
    comps = carregar_componentes(caminho_componentes)

    print("Componentes carregados:")
    for c in comps:
        print(c)

    # 2) Alimentação
    F, z, q, P = definir_alimentacao()
    print("\nAlimentação:")
    print(f"F = {F} kmol/h")
    print(f"z = {z}")
    print(f"q = {q}")
    print(f"P = {P} atm")

    # 3) Especificação (placeholder)
    spec = definir_especificacao_inicial(n_comp=len(comps))
    print("\nEspecificação inicial (placeholder):")
    print(spec)


if __name__ == "__main__":
    main()


