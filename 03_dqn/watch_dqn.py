"""Watch the trained DQN marine play MoveToBeacon at human speed.

Run from the repository root, after training with `dqn.py --env sc2`:

    python 03_dqn/watch_dqn.py
    python 03_dqn/watch_dqn.py --grid              # draw the state grid and Q values on the game
    python 03_dqn/watch_dqn.py --show-q            # extra window: state grid and Q bar chart
    python 03_dqn/watch_dqn.py --model no_replay   # the network trained with --no-replay
    python 03_dqn/watch_dqn.py --random            # compare with a random marine
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
sys.path.insert(0, str(ROOT / "03_dqn"))

from dqn import DQNAgent  # noqa: E402
from envs.move_to_beacon import MoveToBeaconEnv  # noqa: E402

# The drawing helpers are shared with Lesson 2's watch.py.
_spec = importlib.util.spec_from_file_location("lesson2_watch", ROOT / "02_q_learning" / "watch.py")
lesson2_watch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lesson2_watch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--model", choices=["dqn", "no_replay", "no_target"], default="dqn",
                        help="which trained network to load")
    parser.add_argument("--random", action="store_true", help="ignore the network")
    parser.add_argument("--grid", action="store_true", help="draw the state grid and Q values on the game")
    parser.add_argument("--show-q", action="store_true", help="open a window with the state grid and Q values")
    args = parser.parse_args()

    tag = "" if args.model == "dqn" else f"_{args.model}"
    path = ROOT / "03_dqn" / "results" / f"sc2_dqn{tag}.pt"
    if not path.exists():
        extra = "" if args.model == "dqn" else f" --{args.model.replace('_', '-')}"
        sys.exit(f"{path} not found. Train first: python 03_dqn/dqn.py --env sc2{extra}")

    env = MoveToBeaconEnv(realtime=True)
    agent = DQNAgent(env.state_shape, env.n_actions, epsilon=0.0)
    agent.net.load_state_dict(torch.load(path))
    agent.net.eval()
    Q = agent.q_table()           # only for drawing; actions come from the network below
    view = lesson2_watch.QView(Q) if args.show_q else None
    grid = lesson2_watch.GameGrid(env) if args.grid else None
    try:
        for episode in range(args.episodes):
            state, total, done, step = env.reset(), 0.0, False, 0
            while not done:
                # Greedy policy: the action with the highest Q(s, a; θ) from the network.
                q = agent.q_values(state)
                action = random.randrange(env.n_actions) if args.random else int(np.argmax(q))
                step += 1
                if view:
                    view.update(state, action, step, total)
                if grid:
                    grid.draw(state, action, Q)
                state, reward, done = env.step(action)
                total += reward
            print(f"episode {episode + 1}: total reward {total:.0f}")
    finally:
        env.close()
