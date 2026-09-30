"""Compare three agents on MoveToBeacon: Q-table (Lesson 2), DQN (Lesson 3), REINFORCE (Lesson 4).

Run after all three have been trained:

    python 04_reinforce/compare.py

For each of the 289 states, the agent's preferred action (highest Q, or highest
probability) counts as correct if it points within 45 degrees of the beacon.
"""
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("lesson3_compare", ROOT / "03_dqn" / "compare.py")
lesson3 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson3)


def main():
    paths = {"Q-table (Lesson 2)": ROOT / "02_q_learning" / "results" / "sc2_q_table.npy",
             "DQN (Lesson 3)": ROOT / "03_dqn" / "results" / "sc2_dqn_q_table.npy",
             "REINFORCE (Lesson 4)": ROOT / "04_reinforce" / "results" / "sc2_policy_table.npy"}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        sys.exit("Train first. Missing: " + ", ".join(missing))

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.6))
    print(f"{'':22}" + "".join(f"{name:>20}" for name in lesson3.RINGS) + f"{'all 288':>12}")
    for ax, (name, path) in zip(axes, paths.items()):
        table = np.load(path)
        ok = lesson3.correct_map(table)
        scores = lesson3.ring_scores(ok)
        overall = ok.sum() / (ok.size - 1)
        print(f"{name:22}" + "".join(f"{v:20.0%}" for v in scores.values()) + f"{overall:12.0%}")
        lesson3.draw(ax, table, ok, f"{name}: {overall:.0%} point at the beacon\n"
                                    + "   ".join(f"{k.split(' ')[0]} {v:.0%}" for k, v in scores.items()))
    fig.suptitle("Preferred action in each of the 289 states (red arrow = not toward the beacon). "
                 "Colour: highest Q (left, middle) or highest probability (right).", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = ROOT / "04_reinforce" / "results" / "compare_three_agents.png"
    fig.savefig(out, dpi=100)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
