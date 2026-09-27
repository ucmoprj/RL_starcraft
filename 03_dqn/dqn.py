"""Lesson 3: Deep Q-Networks (DQN).

Run from the repository root:

    python 03_dqn/dqn.py --env gridworld                      # seconds
    python 03_dqn/dqn.py --env sc2 --episodes 300             # ~8 minutes
    python 03_dqn/dqn.py --env gridworld --no-replay          # switch off experience replay
    python 03_dqn/dqn.py --env gridworld --no-target          # switch off the target network

Everything outside the network, the replay buffer and the target network is the
same as Lesson 2's q_learning.py. Equation numbers refer to 03_dqn/README.md.
"""
import argparse
import copy
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # lets us import `envs` and `common`

from common.plots import plot_learning_curve, plot_policy_map  # noqa: E402


def to_input(state, state_shape):
    """Turn a table index like (11, 7) into two numbers in [-1, 1], e.g. (0.375, -0.125).

    For MoveToBeacon this is (dx / 8, dy / 8): the same information the Q-table used,
    but as numbers the network can compare, instead of a row number.
    """
    half = (np.array(state_shape, dtype=np.float32) - 1) / 2
    return (np.array(state, dtype=np.float32) - half) / half


class QNetwork(nn.Module):
    """Q(s, ·; θ): 2 numbers in, one Q value per action out (Eq. 1)."""

    def __init__(self, n_inputs, n_actions, hidden=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.layers(x)


class ReplayBuffer:
    """The last `capacity` transitions; learning samples a random batch from them (section 5)."""

    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def add(self, *transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    """Lesson 2's agent with the table replaced by a network."""

    def __init__(self, state_shape, n_actions, gamma=0.9, lr=1e-3, epsilon=1.0,
                 replay=True, target=True, buffer_size=10_000, batch_size=64,
                 learning_starts=500, target_every=250):
        self.state_shape, self.n_actions = state_shape, n_actions
        self.gamma, self.epsilon = gamma, epsilon
        self.net = QNetwork(len(state_shape), n_actions)
        # The target network is a frozen copy of `net`, refreshed every `target_every` steps (Eq. 4).
        self.target_net = copy.deepcopy(self.net) if target else self.net
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.replay = replay
        self.buffer = ReplayBuffer(buffer_size if replay else 1)
        self.batch_size = batch_size if replay else 1
        self.learning_starts = learning_starts if replay else 1
        self.target_every = target_every
        self.steps = 0

    def q_values(self, state):
        with torch.no_grad():
            return self.net(torch.from_numpy(to_input(state, self.state_shape))).numpy()

    def act(self, state):
        """Epsilon-greedy action selection (Lesson 2, Eq. 5)."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)       # explore
        q = self.q_values(state)
        return int(random.choice(np.flatnonzero(q == q.max())))  # exploit

    def learn(self, state, action, reward, next_state, terminal):
        """Store the transition, then take one gradient step on a batch (Eq. 2 and 3)."""
        self.buffer.add(to_input(state, self.state_shape), action, reward,
                        to_input(next_state, self.state_shape), float(terminal))
        self.steps += 1
        if len(self.buffer) < self.learning_starts:
            return None

        batch = self.buffer.sample(self.batch_size)
        s, a, r, s2, term = (torch.as_tensor(np.array(x)) for x in zip(*batch))
        s, s2, r, term = s.float(), s2.float(), r.float(), term.float()

        # TD target, exactly as in Lesson 2 but read from the (target) network.
        # no_grad: we do not differentiate through the target (semi-gradient).
        with torch.no_grad():
            future = self.target_net(s2).max(dim=1).values * (1.0 - term)
            y = r + self.gamma * future                                   # Eq. 3
        q = self.net(s).gather(1, a.long().unsqueeze(1)).squeeze(1)      # Q(s, a; θ)
        loss = 0.5 * ((y - q) ** 2).mean()                                # Eq. 2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()                                             # θ ← θ − α ∇θ L

        if self.target_net is not self.net and self.steps % self.target_every == 0:
            self.target_net.load_state_dict(self.net.state_dict())        # Eq. 4
        return loss.item()

    def q_table(self):
        """Evaluate the network at every table cell, so we can draw it like Lesson 2's table."""
        cells = np.indices(self.state_shape).reshape(len(self.state_shape), -1).T
        x = torch.from_numpy(np.stack([to_input(c, self.state_shape) for c in cells]))
        with torch.no_grad():
            q = self.net(x).numpy()
        return q.reshape(*self.state_shape, self.n_actions)


def linear_schedule(start, end, fraction, episode, total):
    """Decay epsilon linearly from `start` to `end` over the first `fraction` of training."""
    progress = min(1.0, episode / max(1, fraction * total))
    return start + progress * (end - start)


def train(env, agent, episodes, eps_start=1.0, eps_end=0.05, eps_fraction=0.6,
          terminal_on_reward=False, log_every=10):
    """The same loop as Lesson 2's train()."""
    episode_rewards, episode_lengths = [], []
    for episode in range(episodes):
        agent.epsilon = linear_schedule(eps_start, eps_end, eps_fraction, episode, episodes)
        state = env.reset()
        total, steps, done = 0.0, 0, False
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            terminal = (reward > 0) if terminal_on_reward else env.is_terminal(next_state)
            agent.learn(state, action, reward, next_state, terminal)
            state = next_state
            total += reward
            steps += 1
        episode_rewards.append(total)
        episode_lengths.append(steps)
        if (episode + 1) % log_every == 0:
            print(f"episode {episode + 1:4d} | epsilon {agent.epsilon:.2f} | "
                  f"avg reward {np.mean(episode_rewards[-log_every:]):6.2f} | "
                  f"avg steps {np.mean(episode_lengths[-log_every:]):6.1f}")
    return episode_rewards, episode_lengths


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=["gridworld", "sc2"], default="gridworld")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3, help="the optimiser's learning rate (Lesson 2's α)")
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--no-replay", action="store_true", help="learn from each transition once, in order")
    parser.add_argument("--no-target", action="store_true", help="compute targets with the network being trained")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = ROOT / "03_dqn" / "results"
    out.mkdir(exist_ok=True)
    tag = ("_no_replay" if args.no_replay else "") + ("_no_target" if args.no_target else "")

    if args.env == "gridworld":
        from envs.gridworld import ACTIONS, GridWorld
        env = GridWorld()
        agent = DQNAgent(env.state_shape, env.n_actions, args.gamma, args.lr,
                         replay=not args.no_replay, target=not args.no_target)
        rewards, lengths = train(env, agent, args.episodes or 300, log_every=50)
        Q = agent.q_table()
        plot_learning_curve(lengths, out / f"gridworld_learning_curve{tag}.png",
                            title=f"DQN on GridWorld{tag.replace('_', ' ')}",
                            ylabel="steps to reach the beacon", baselines={"shortest path = 5": 5})
        plot_policy_map(Q, ACTIONS, out / f"gridworld_policy{tag}.png",
                        title="DQN on GridWorld: greedy policy", blocked=env.walls, goal=env.beacon)
        print("\nLearned greedy policy:")
        env.render_policy({s: int(np.argmax(Q[s])) for s in env.states()})
    else:
        from envs.move_to_beacon import DIRECTIONS, MoveToBeaconEnv
        env = MoveToBeaconEnv()
        agent = DQNAgent(env.state_shape, env.n_actions, args.gamma, args.lr,
                         replay=not args.no_replay, target=not args.no_target)
        try:
            rewards, _ = train(env, agent, args.episodes or 300, terminal_on_reward=True)
        finally:
            env.close()
        Q = agent.q_table()
        np.save(out / f"sc2_dqn_q_table{tag}.npy", Q)
        torch.save(agent.net.state_dict(), out / f"sc2_dqn{tag}.pt")
        plot_learning_curve(rewards, out / f"sc2_learning_curve{tag}.png",
                            title=f"DQN on MoveToBeacon{tag.replace('_', ' ')}",
                            baselines={"random ≈ 1": 1, "Q-table (Lesson 2) ≈ 20": 20, "scripted ≈ 23": 23})
        plot_policy_map(Q, DIRECTIONS, out / f"sc2_policy{tag}.png",
                        title="DQN on MoveToBeacon: greedy policy\n"
                              "(star = beacon, each cell = where the beacon is relative to the marine)",
                        center=(env.radius, env.radius))
        print(f"Saved the network and its Q-table to {out}")
    print(f"Saved plots to {out}")


if __name__ == "__main__":
    main()
