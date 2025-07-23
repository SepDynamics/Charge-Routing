import numpy as np
import networkx as nx

from spiral_demo import square_spiral


def build_spiral_graph(n_turns=4, step=1.0, cross=True):
    """Return a graph representation of a spiral with optional midpoint cross.

    Parameters
    ----------
    n_turns : int
        Number of turns in the square spiral.
    step : float
        Step size between points.
    cross : bool
        If True, add a cross connection between the start and midpoint
        to mimic role flipping.
    """
    coords = square_spiral(n_turns, step)
    G = nx.Graph()
    for i, p in enumerate(coords):
        G.add_node(i, pos=p)
        if i > 0:
            G.add_edge(i - 1, i, resistance=1.0)
    if cross:
        mid = len(coords) // 2
        G.add_edge(0, mid, resistance=1.0)
    return G


def solve_potentials(G, src=0, sink=None, V_source=1.0):
    """Solve node potentials with a DC source between src and sink."""
    if sink is None:
        sink = max(G.nodes)
    n = G.number_of_nodes()
    A = nx.to_numpy_array(G, nodelist=range(n))
    L = np.diag(A.sum(axis=1)) - A
    b = np.zeros(n)
    # Dirichlet boundary conditions
    L[src, :] = 0
    L[src, src] = 1
    b[src] = V_source
    L[sink, :] = 0
    L[sink, sink] = 1
    b[sink] = 0.0
    v = np.linalg.solve(L, b)
    return {node: float(v[i]) for i, node in enumerate(G.nodes())}


if __name__ == "__main__":
    G = build_spiral_graph(n_turns=3, cross=True)
    potentials = solve_potentials(G)
    for node, value in potentials.items():
        pos = G.nodes[node]["pos"]
        print(f"{node}\t{pos[0]}\t{pos[1]}\t{value:.3f}")
