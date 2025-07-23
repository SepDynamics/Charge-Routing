import numpy as np
from spiral_network import build_spiral_graph, solve_potentials
from spiral_sweep import uniformity_metric


def analyze(max_turns=6):
    results = []
    for turns in range(2, max_turns + 1):
        for cross in (False, True):
            G = build_spiral_graph(n_turns=turns, cross=cross)
            potentials = solve_potentials(G)
            std = uniformity_metric(potentials)
            results.append({"turns": turns, "cross": cross, "stddev": std})
    return results


if __name__ == "__main__":
    results = analyze()
    cross = [r["stddev"] for r in results if r["cross"]]
    nocross = [r["stddev"] for r in results if not r["cross"]]
    print("Average stddev with cross:", np.mean(cross))
    print("Average stddev without cross:", np.mean(nocross))
