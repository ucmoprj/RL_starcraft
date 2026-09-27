"""Lesson 3 walkthrough: what changes when a table becomes a function.

The corridor from Lesson 2, now 8 cells long, beacon at the right end.
We only look at the action "right", so each cell has one number, Q(s).

    [0] [1] [2] [3] [4] [5] [6] [7] [B]

Run from the repository root:

    python 03_dqn/walkthrough.py --stage 1   # one update: a table changes one cell, a function changes all
    python 03_dqn/walkthrough.py --stage 2   # train on cells 5-7 only; what does it guess for 0-4?

The "network" here is the smallest one possible: a straight line,
    Q(s) = w * x + b,   with x = s / 7  (the cell number scaled to [0, 1]).
It has two parameters (w, b) instead of eight table entries. A real DQN has
thousands of parameters and bends, but the update works exactly the same way.
"""
import argparse

import numpy as np

CELLS = 8
GAMMA = 0.9
TRUE_Q = GAMMA ** (CELLS - 1 - np.arange(CELLS))   # what Q-learning converges to: 0.9^(steps left - 1)


def x_of(s):
    return s / (CELLS - 1)


def show(label, values):
    print(f"  {label:<12}" + " ".join(f"{v:6.3f}" for v in values))


def header():
    print("  " + " " * 12 + " ".join(f"  [{s}] " for s in range(CELLS)))


def stage1(alpha):
    s, y = 7, 1.0     # cell 7 -> right -> beacon: target y = r = 1 (Lesson 2, Eq. 3)
    print(f"One update: cell {s}, action right, reaches the beacon, target y = 1, alpha = {alpha}\n")

    print("TABLE (Lesson 2): one number per cell")
    table = np.zeros(CELLS)
    header()
    show("before", table)
    table[s] += alpha * (y - table[s])
    show("after", table)
    print(f"  Only Q[{s}] moved: 0 + {alpha} x (1 - 0) = {table[s]:.3f}. Every other cell is still 0.\n")

    print("FUNCTION (Lesson 3): Q(s) = w * x + b, two parameters shared by all cells")
    w, b = 0.0, 0.0
    q_all = w * x_of(np.arange(CELLS)) + b
    header()
    show("before", q_all)
    q = w * x_of(s) + b
    delta = y - q
    # loss L = 1/2 (y - Q)^2.  dL/dw = -(y - Q) * x,  dL/db = -(y - Q)   (README Eq. 2)
    w_new, b_new = w + alpha * delta * x_of(s), b + alpha * delta
    q_all_new = w_new * x_of(np.arange(CELLS)) + b_new
    show("after", q_all_new)
    print(f"\n  error    delta = y - Q(7) = 1 - {q:.3f} = {delta:.3f}")
    print(f"  w <- w + alpha * delta * x(7) = {w:.3f} + {alpha} x {delta:.3f} x {x_of(s):.3f} = {w_new:.3f}")
    print(f"  b <- b + alpha * delta        = {b:.3f} + {alpha} x {delta:.3f}         = {b_new:.3f}")
    print("  The update was aimed at cell 7, but w and b are shared, so EVERY cell moved.")
    print("  Cells close to 7 moved more (larger x), cells far away moved less.")


def stage2(alpha, steps):
    trained = [5, 6, 7]
    print(f"Train only on cells {trained} (the ones next to the beacon), {steps} gradient steps,")
    print("using their converged values from Lesson 2 as targets. Cells 0-4 are never visited.\n")
    w, b = 0.0, 0.0
    for _ in range(steps):
        for s in trained:
            delta = TRUE_Q[s] - (w * x_of(s) + b)
            w += alpha * delta * x_of(s)
            b += alpha * delta
    guess = w * x_of(np.arange(CELLS)) + b
    table = np.where(np.isin(np.arange(CELLS), trained), TRUE_Q, 0.0)
    header()
    show("true Q", TRUE_Q)
    show("table", table)
    show("function", guess)
    print(f"\n  w = {w:.3f}, b = {b:.3f}")
    print("  The table knows nothing about cells 0-4: they stay at 0, and 'right' there is no")
    print("  better than any other direction. The function never saw them either, but it")
    print("  GUESSES: further from the beacon means a lower value. The ordering is right,")
    print("  which is what choosing an action needs. The numbers are not exact: a straight")
    print("  line cannot bend like 0.9^k. That is why DQN uses a bigger, bendier network.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", type=int, choices=[1, 2], default=1)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--steps", type=int, default=2000)
    args = p.parse_args()
    if args.stage == 1:
        stage1(args.alpha)
    else:
        stage2(args.alpha, args.steps)


if __name__ == "__main__":
    main()
