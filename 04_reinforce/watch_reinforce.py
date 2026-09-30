"""Watch the trained REINFORCE marine play MoveToBeacon at human speed.

Run from the repository root, after training with `reinforce.py --env sc2`:

    python 04_reinforce/watch_reinforce.py              # sample actions, as in training
    python 04_reinforce/watch_reinforce.py --greedy     # always the most likely action
    python 04_reinforce/watch_reinforce.py --grid       # draw the state grid and the 8 probabilities on the game
    python 04_reinforce/watch_reinforce.py --random     # compare with a random marine

With --grid, the numbers around the marine are probabilities π(a | s), not Q values.
They add up to 1. The chosen direction is in green.
"""
import argparse
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "04_reinforce"))

from envs.move_to_beacon import MoveToBeaconEnv  # noqa: E402
from reinforce import ReinforceAgent  # noqa: E402

# The drawing helper is shared with Lesson 2's watch.py.
_spec = importlib.util.spec_from_file_location("lesson2_watch", ROOT / "02_q_learning" / "watch.py")
lesson2_watch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson2_watch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--greedy", action="store_true", help="always pick the most likely action")
    parser.add_argument("--random", action="store_true", help="ignore the policy")
    parser.add_argument("--grid", action="store_true", help="draw the state grid and probabilities on the game")
    args = parser.parse_args()

    path = ROOT / "04_reinforce" / "results" / "sc2_reinforce.pt"
    if not path.exists():
        sys.exit(f"{path} not found. Train first: python 04_reinforce/reinforce.py --env sc2")

    env = MoveToBeaconEnv(realtime=True)
    agent = ReinforceAgent(env.state_shape, env.n_actions)
    agent.policy.load_state_dict(torch.load(path))
    agent.policy.eval()
    P = agent.prob_table()                  # only for drawing
    grid = lesson2_watch.GameGrid(env) if args.grid else None
    try:
        for episode in range(args.episodes):
            state, total, done = env.reset(), 0.0, False
            while not done:
                action = random.randrange(env.n_actions) if args.random else agent.act(state, greedy=args.greedy)
                if grid:
                    grid.draw(state, action, P)
                state, reward, done = env.step(action)
                total += reward
            print(f"episode {episode + 1}: total reward {total:.0f}")
    finally:
        env.close()
