import numpy as np
from spiral_network import build_spiral_graph, solve_potentials


def total_resistance(G, src=0, sink=None, V_source=1.0):
    """Return effective resistance between src and sink."""
    potentials = solve_potentials(G, src=src, sink=sink, V_source=V_source)
    if sink is None:
        sink = max(G.nodes)
    current = 0.0
    for nbr in G.neighbors(src):
        r = G[src][nbr].get("resistance", 1.0)
        current += (potentials[src] - potentials[nbr]) / r
    return float("inf") if current == 0 else V_source / current


def impedance_scan(max_turns=6):
    """Print effective resistance for varying turns and cross options."""
    print("turns\tcross\tresistance")
    for turns in range(2, max_turns + 1):
        for cross in (False, True):
            G = build_spiral_graph(n_turns=turns, cross=cross)
            R = total_resistance(G)
            flag = 1 if cross else 0
            print(f"{turns}\t{flag}\t{R:.4f}")


if __name__ == "__main__":
    impedance_scan()
