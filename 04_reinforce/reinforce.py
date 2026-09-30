"""Lesson 4: REINFORCE, learning a policy directly.

Run from the repository root:

    python 04_reinforce/reinforce.py --env gridworld                 # under a minute
    python 04_reinforce/reinforce.py --env sc2 --episodes 600        # ~15 minutes

There is no Q-table, no TD target and no epsilon here. The network outputs the
probability of each action, the agent samples from it, and after each episode the
probabilities of the actions taken are pushed up in proportion to the return that
followed them. Equation numbers refer to 04_reinforce/README.md.
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # lets us import `envs` and `common`

from common.plots import plot_learning_curve, plot_policy_map  # noqa: E402


def to_input(state, state_shape):
    """The same input as Lesson 3: a table index like (11, 7) becomes two numbers in [-1, 1]."""
    half = (np.array(state_shape, dtype=np.float32) - 1) / 2
    return (np.array(state, dtype=np.float32) - half) / half


class PolicyNetwork(nn.Module):
    """π(a | s; θ): 2 numbers in, one probability per action out (Eq. 1).

    The layers are the same as Lesson 3's QNetwork. Only the meaning of the output
    changes: softmax turns the 8 outputs into probabilities that add up to 1.
    """

    def __init__(self, n_inputs, n_actions, hidden=64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return torch.softmax(self.layers(x), dim=-1)


class ReinforceAgent:
    def __init__(self, state_shape, n_actions, gamma=0.9, lr=1e-3):
        self.state_shape, self.n_actions, self.gamma = state_shape, n_actions, gamma
        self.policy = PolicyNetwork(len(state_shape), n_actions)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.episode = []          # (state input, action, reward, terminal) of the current episode

    def probabilities(self, state):
        with torch.no_grad():
            return self.policy(torch.from_numpy(to_input(state, self.state_shape))).numpy()

    def act(self, state, greedy=False):
        """Sample an action from π(· | s). Exploration comes from the randomness itself."""
        p = self.probabilities(state)
        if greedy:
            return int(np.argmax(p))
        return int(np.random.choice(self.n_actions, p=p / p.sum()))

    def remember(self, state, action, reward, terminal):
        self.episode.append((to_input(state, self.state_shape), action, reward, terminal))

    def returns(self):
        """G_t = r_t + γ r_{t+1} + γ² r_{t+2} + ...  computed backwards (Eq. 2).

        A terminal step cuts the sum, like Lesson 2's "no future after a terminal state".
        """
        G, out = 0.0, []
        for _, _, reward, terminal in reversed(self.episode):
            G = reward + (0.0 if terminal else self.gamma * G)
            out.append(G)
        return np.array(out[::-1], dtype=np.float32)

    def learn(self):
        """One gradient step at the end of an episode (Eq. 3 and 4)."""
        G = torch.from_numpy(self.returns())
        s = torch.from_numpy(np.stack([e[0] for e in self.episode]))
        a = torch.tensor([e[1] for e in self.episode])
        probs = self.policy(s)
        log_prob = torch.log(probs.gather(1, a.unsqueeze(1)).squeeze(1) + 1e-8)
        # Maximise Σ G_t log π(a_t|s_t)  ==  minimise its negative.
        loss = -(G * log_prob).sum() / len(self.episode)                 # Eq. 4
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.episode = []
        # Entropy: how random the policy was in the states it met (log 8 = 2.08 is uniform, 0 is certain).
        with torch.no_grad():
            return -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean().item()

    def prob_table(self):
        """π(· | s) at every table cell, so we can draw it like a Q-table."""
        cells = np.indices(self.state_shape).reshape(len(self.state_shape), -1).T
        x = torch.from_numpy(np.stack([to_input(c, self.state_shape) for c in cells]))
        with torch.no_grad():
            p = self.policy(x).numpy()
        return p.reshape(*self.state_shape, self.n_actions)


def train(env, agent, episodes, terminal_on_reward=False, log_every=10):
    episode_rewards, episode_lengths, entropies = [], [], []
    for episode in range(episodes):
        state = env.reset()
        total, steps, done = 0.0, 0, False
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            terminal = (reward > 0) if terminal_on_reward else env.is_terminal(next_state)
            agent.remember(state, action, reward, terminal)
            state = next_state
            total += reward
            steps += 1
        entropies.append(agent.learn())  # Monte Carlo: learn only once the episode is over
        episode_rewards.append(total)
        episode_lengths.append(steps)
        if (episode + 1) % log_every == 0:
            print(f"episode {episode + 1:4d} | avg reward {np.mean(episode_rewards[-log_every:]):6.2f} | "
                  f"avg steps {np.mean(episode_lengths[-log_every:]):6.1f} | "
                  f"entropy {np.mean(entropies[-log_every:]):.2f}")
    return episode_rewards, episode_lengths


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=["gridworld", "sc2"], default="gridworld")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = ROOT / "04_reinforce" / "results"
    out.mkdir(exist_ok=True)

    if args.env == "gridworld":
        from envs.gridworld import ACTIONS, GridWorld
        env = GridWorld()
        agent = ReinforceAgent(env.state_shape, env.n_actions, args.gamma, args.lr)
        _, lengths = train(env, agent, args.episodes or 1000, log_every=100)
        P = agent.prob_table()
        plot_learning_curve(lengths, out / "gridworld_learning_curve.png", title="REINFORCE on GridWorld",
                            ylabel="steps to reach the beacon", baselines={"shortest path = 5": 5})
        plot_policy_map(P, ACTIONS, out / "gridworld_policy.png",
                        title="REINFORCE on GridWorld: most likely action", blocked=env.walls, goal=env.beacon)
        print("\nMost likely action in each cell:")
        env.render_policy({s: int(np.argmax(P[s])) for s in env.states()})
        print("\nProbability of that action:")
        env.render_values({s: float(P[s].max()) for s in env.states()})
    else:
        from envs.move_to_beacon import DIRECTIONS, MoveToBeaconEnv
        env = MoveToBeaconEnv()
        agent = ReinforceAgent(env.state_shape, env.n_actions, args.gamma, args.lr)
        try:
            rewards, _ = train(env, agent, args.episodes or 600, terminal_on_reward=True)
        finally:
            env.close()
        P = agent.prob_table()
        np.save(out / "sc2_policy_table.npy", P)
        torch.save(agent.policy.state_dict(), out / "sc2_reinforce.pt")
        plot_learning_curve(rewards, out / "sc2_learning_curve.png", title="REINFORCE on MoveToBeacon",
                            baselines={"random ≈ 1": 1, "Q-table ≈ 20": 20, "DQN ≈ 23": 23})
        plot_policy_map(P, DIRECTIONS, out / "sc2_policy.png",
                        title="REINFORCE on MoveToBeacon: most likely action\n"
                              "(star = beacon, each cell = where the beacon is relative to the marine)",
                        center=(env.radius, env.radius))
        print(f"Saved the policy network and its probability table to {out}")
    print(f"Saved plots to {out}")


if __name__ == "__main__":
    main()
