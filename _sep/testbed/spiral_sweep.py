import numpy as np
from spiral_network import build_spiral_graph, solve_potentials


def uniformity_metric(potentials):
    """Return standard deviation of potentials."""
    values = list(potentials.values())
    return float(np.std(values))


def sweep(max_turns=6):
    """Print uniformity metric for varying turns and cross option."""
    print("turns\tcross\tstddev")
    for turns in range(2, max_turns + 1):
        for cross in (False, True):
            G = build_spiral_graph(n_turns=turns, cross=cross)
            potentials = solve_potentials(G)
            std = uniformity_metric(potentials)
            flag = 1 if cross else 0
            print(f"{turns}\t{flag}\t{std:.4f}")


if __name__ == "__main__":
    sweep()
