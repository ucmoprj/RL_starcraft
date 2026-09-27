"""Compare Lesson 2's Q-table with Lesson 3's DQN on MoveToBeacon.

Run after both have been trained:

    python 02_q_learning/q_learning.py --env sc2 --episodes 300
    python 03_dqn/dqn.py --env sc2 --episodes 300
    python 03_dqn/compare.py

For every one of the 289 states we know the right answer: walk toward the beacon.
A state counts as "correct" if its greedy action points within 45° of the beacon.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DIRECTIONS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
RINGS = {"near (1-2 cells)": (1, 2), "middle (3-5)": (3, 5), "far (6-8)": (6, 8)}


def correct_map(Q):
    """True where the greedy action points within 45 degrees of the beacon."""
    width, height, _ = Q.shape
    r = width // 2
    ok = np.zeros((width, height), dtype=bool)
    for x in range(width):
        for y in range(height):
            beacon = np.array([x - r, y - r], dtype=float)
            if not beacon.any():
                continue
            move = np.array(DIRECTIONS[int(np.argmax(Q[x, y]))], dtype=float)
            cos = beacon @ move / (np.linalg.norm(beacon) * np.linalg.norm(move))
            ok[x, y] = cos >= np.cos(np.radians(45)) - 1e-9
    return ok


def ring_scores(ok):
    r = ok.shape[0] // 2
    xs, ys = np.indices(ok.shape)
    dist = np.maximum(abs(xs - r), abs(ys - r))
    return {name: ok[(dist >= lo) & (dist <= hi)].mean() for name, (lo, hi) in RINGS.items()}


def draw(ax, Q, ok, title):
    width, height, _ = Q.shape
    ax.imshow(Q.max(axis=2).T, cmap="viridis", origin="upper")
    for x in range(width):
        for y in range(height):
            if x == y == width // 2:
                continue
            dx, dy = DIRECTIONS[int(np.argmax(Q[x, y]))]
            n = np.hypot(dx, dy)
            ax.arrow(x - 0.3 * dx / n, y - 0.3 * dy / n, 0.45 * dx / n, 0.45 * dy / n,
                     head_width=0.22, head_length=0.18, length_includes_head=True,
                     color="w" if ok[x, y] else "#ff5050")
    ax.plot(width // 2, height // 2, marker="*", color="red", markersize=14)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title)


def main():
    paths = {"Q-table (Lesson 2)": ROOT / "02_q_learning" / "results" / "sc2_q_table.npy",
             "DQN (Lesson 3)": ROOT / "03_dqn" / "results" / "sc2_dqn_q_table.npy"}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        sys.exit("Train first. Missing: " + ", ".join(missing))

    fig, axes = plt.subplots(1, 2, figsize=(12, 6.4))
    print(f"{'':20}" + "".join(f"{name:>20}" for name in RINGS) + f"{'all 288':>12}")
    for ax, (name, path) in zip(axes, paths.items()):
        Q = np.load(path)
        ok = correct_map(Q)
        scores = ring_scores(ok)
        overall = ok.sum() / (ok.size - 1)
        print(f"{name:20}" + "".join(f"{v:20.0%}" for v in scores.values()) + f"{overall:12.0%}")
        draw(ax, Q, ok, f"{name}: {overall:.0%} of states point at the beacon\n"
                        + "   ".join(f"{k.split(' ')[0]} {v:.0%}" for k, v in scores.items()))
    fig.suptitle("Greedy action in each of the 289 states (red arrow = not toward the beacon)", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = ROOT / "03_dqn" / "results" / "compare_table_vs_dqn.png"
    fig.savefig(out, dpi=110)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
