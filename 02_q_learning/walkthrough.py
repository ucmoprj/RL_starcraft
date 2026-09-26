"""Lesson 2 walkthrough: watch every single Q-learning update, by hand.

The world is a corridor of 4 cells. The marine starts in cell 0, the beacon is in cell 3.

    [0] [1] [2] [B]
     M

Actions: 0 = left, 1 = right. Reaching B gives +1 and ends the episode.
The Q-table has 3 states x 2 actions = 6 numbers (B is terminal, no row needed).

Run from the repository root:

    python 02_q_learning/walkthrough.py --stage 1   # always walk right, see value flow back
    python 02_q_learning/walkthrough.py --stage 2   # epsilon-greedy: the agent must find the way itself
"""
import argparse
import random

import numpy as np

N_CELLS = 4          # cells 0, 1, 2 and the beacon (3)
BEACON = 3
NAMES = ["left", "right"]


def step(s, a):
    """The environment: move one cell; the wall at the left edge stops you."""
    s2 = max(0, s - 1) if a == 0 else s + 1
    reward = 1.0 if s2 == BEACON else 0.0
    return s2, reward, s2 == BEACON


def show(Q):
    print("          left    right")
    for s in range(BEACON):
        print(f"  cell {s}  {Q[s, 0]:6.3f}  {Q[s, 1]:6.3f}")


def update(Q, s, a, r, s2, done, alpha, gamma):
    """One Q-learning update (README Eq. 3 and 4), printed in full."""
    future = 0.0 if done else Q[s2].max()
    target = r + gamma * future
    error = target - Q[s, a]
    old = Q[s, a]
    Q[s, a] += alpha * error
    where = "B (terminal)" if done else f"cell {s2}"
    print(f"  cell {s} --{NAMES[a]:>5}--> {where}, reward {r:.0f}")
    if done:
        print(f"    target y = r = {r:.3f}")
    else:
        print(f"    target y = r + gamma * max Q(cell {s2}) = {r:.0f} + {gamma} * {future:.3f} = {target:.3f}")
    print(f"    error  d = y - Q = {target:.3f} - {old:.3f} = {error:.3f}")
    print(f"    Q(cell {s}, {NAMES[a]}) = {old:.3f} + {alpha} * {error:.3f} = {Q[s, a]:.3f}")


def stage1(alpha, gamma, episodes):
    """Always walk right. No exploration, so we can follow the numbers."""
    Q = np.zeros((BEACON, 2))
    print("Start: every Q value is 0.\n")
    show(Q)
    for ep in range(1, episodes + 1):
        print(f"\n===== Episode {ep} =====")
        s, done = 0, False
        while not done:
            s2, r, done = step(s, 1)
            update(Q, s, 1, r, s2, done, alpha, gamma)
            s = s2
        print(f"\nQ-table after episode {ep}:")
        show(Q)


def stage2(alpha, gamma, episodes, epsilon, seed):
    """Epsilon-greedy. The agent does not know that 'right' is good."""
    random.seed(seed)
    Q = np.zeros((BEACON, 2))
    for ep in range(1, episodes + 1):
        s, done, steps, path = 0, False, 0, ["0"]
        while not done:
            if random.random() < epsilon:
                a, why = random.randrange(2), "random"
            else:
                best = np.flatnonzero(Q[s] == Q[s].max())
                a = int(random.choice(best))
                why = "greedy" if len(best) == 1 else "tie->random"
            s2, r, done = step(s, a)
            steps += 1
            path.append(f"-{NAMES[a][0].upper()}({why})-> {'B' if done else s2}")
            # Same update as stage 1 (Eq. 3 and 4), just not printed.
            future = 0.0 if done else Q[s2].max()
            Q[s, a] += alpha * (r + gamma * future - Q[s, a])
            s = s2
        print(f"\n===== Episode {ep}: {steps} steps =====")
        print("  " + " ".join(path))
        show(Q)
    print("\nGreedy policy: " + ", ".join(
        f"cell {s} -> {NAMES[int(np.argmax(Q[s]))]}" for s in range(BEACON)))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", type=int, choices=[1, 2], default=1)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--epsilon", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.stage == 1:
        stage1(args.alpha, args.gamma, args.episodes or 3)
    else:
        stage2(args.alpha, args.gamma, args.episodes or 8, args.epsilon, args.seed)


if __name__ == "__main__":
    main()
