import math


def square_spiral(n_turns=5, step=1.0):
    """Generate coordinates for a square spiral."""
    x = y = 0
    dx, dy = step, 0
    coords = [(x, y)]
    steps = 1
    for turn in range(1, 2 * n_turns + 1):
        for _ in range(steps):
            x += dx
            y += dy
            coords.append((x, y))
        dx, dy = -dy, dx  # rotate 90 degrees
        if turn % 2 == 0:
            steps += 1
    return coords


if __name__ == "__main__":
    pts = square_spiral(4, 1)
    for p in pts:
        print(f"{p[0]}\t{p[1]}")
