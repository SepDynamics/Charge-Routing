import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp

from spiral_network import build_spiral_graph


def build_rc_network(n_turns=4, step=1.0, cross=True, R=1.0, C=1.0):
    """Return a spiral graph with per-edge resistance and capacitance."""
    G = build_spiral_graph(n_turns=n_turns, step=step, cross=cross)
    for u, v in G.edges:
        G[u][v]["resistance"] = R
        G[u][v]["capacitance"] = C
    return G


def rc_odes(t, v, G, src, sink, V_source):
    """ODE system for RC network."""
    v[src] = V_source
    v[sink] = 0.0
    dv = np.zeros_like(v)
    for u, w in G.edges:
        r = G[u][w]["resistance"]
        c = G[u][w]["capacitance"]
        i = (v[u] - v[w]) / r
        dv[u] -= i / c
        dv[w] += i / c
    dv[src] = 0.0
    dv[sink] = 0.0
    return dv


def run(duration=1.0, n_turns=3, cross=True, V_source=1.0):
    G = build_rc_network(n_turns=n_turns, cross=cross)
    src = 0
    sink = max(G.nodes)
    n = G.number_of_nodes()
    v0 = np.zeros(n)
    v0[src] = V_source
    sol = solve_ivp(rc_odes, [0, duration], v0, args=(G, src, sink, V_source),
                    max_step=0.05)
    final = sol.y[:, -1]
    for node in G.nodes:
        pos = G.nodes[node]["pos"]
        print(f"{node}\t{pos[0]}\t{pos[1]}\t{final[node]:.3f}")


if __name__ == "__main__":
    run()
