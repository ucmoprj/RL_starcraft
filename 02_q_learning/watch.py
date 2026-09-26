"""Watch the trained Q-learning marine play MoveToBeacon at human speed.

Run from the repository root, after training with `q_learning.py --env sc2`:

    python 02_q_learning/watch.py
    python 02_q_learning/watch.py --random     # compare with a random marine
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.move_to_beacon import MoveToBeaconEnv  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--random", action="store_true", help="ignore the Q-table")
    args = parser.parse_args()

    q_path = ROOT / "02_q_learning" / "results" / "sc2_q_table.npy"
    if not args.random and not q_path.exists():
        sys.exit(f"{q_path} not found. Train first: python 02_q_learning/q_learning.py --env sc2")
    Q = None if args.random else np.load(q_path)

    env = MoveToBeaconEnv(realtime=True)
    try:
        for episode in range(args.episodes):
            state, total, done = env.reset(), 0.0, False
            while not done:
                # Greedy policy (Eq. 5 with epsilon = 0): always the best known action.
                action = random.randrange(env.n_actions) if Q is None else int(np.argmax(Q[state]))
                state, reward, done = env.step(action)
                total += reward
            print(f"episode {episode + 1}: total reward {total:.0f}")
    finally:
        env.close()
