"""Lesson 2: tabular Q-learning.

Run from the repository root:

    python 02_q_learning/q_learning.py --env gridworld            # seconds
    python 02_q_learning/q_learning.py --env sc2 --episodes 300   # ~6 minutes

Every important line is labelled with the equation it implements.
The equation numbers refer to 02_q_learning/README.md.
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))  # lets us import `envs` and `common`

from common.plots import plot_learning_curve, plot_policy_map  # noqa: E402


class QLearningAgent:
    """A Q-table plus the two things an agent must do: act and learn."""

    def __init__(self, state_shape, n_actions, alpha=0.1, gamma=0.9, epsilon=1.0):
        # Q[s][a]: our current estimate of how good action a is in state s.
        # One number per (state, action) pair, all starting at 0.
        self.Q = np.zeros(state_shape + (n_actions,))
        self.n_actions = n_actions
        self.alpha = alpha      # learning rate: how far to move toward the target
        self.gamma = gamma      # discount factor: how much the future matters
        self.epsilon = epsilon  # exploration rate: chance of a random action

    def act(self, state):
        """Epsilon-greedy action selection (Eq. 5)."""
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)       # explore
        q = self.Q[state]
        best = np.flatnonzero(q == q.max())               # exploit (break ties randomly)
        return int(random.choice(best))

    def learn(self, state, action, reward, next_state, terminal):
        """One Q-learning update (Eq. 3 and 4)."""
        # TD target: reward now + discounted value of the best next action.
        # If the episode really ended there is no future, so the target is just r.
        future = 0.0 if terminal else np.max(self.Q[next_state])
        td_target = reward + self.gamma * future                      # Eq. 3
        td_error = td_target - self.Q[state][action]                  # Eq. 3
        self.Q[state][action] += self.alpha * td_error                # Eq. 4
        return td_error


def linear_schedule(start, end, fraction, episode, total):
    """Decay epsilon linearly from `start` to `end` over the first `fraction` of training."""
    progress = min(1.0, episode / max(1, fraction * total))
    return start + progress * (end - start)


def train(env, agent, episodes, eps_start=1.0, eps_end=0.05, eps_fraction=0.6,
          terminal_on_reward=False, log_every=10):
    episode_rewards, episode_lengths = [], []
    for episode in range(episodes):
        agent.epsilon = linear_schedule(eps_start, eps_end, eps_fraction, episode, episodes)
        state = env.reset()
        total, steps, done = 0.0, 0, False
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            # In GridWorld, reaching the beacon ends the episode (true terminal).
            # In MoveToBeacon the episode continues after the reward, but the
            # beacon jumps to a new place, so we treat each "reach" as the end
            # of a sub-task (see README, section "Terminal states").
            terminal = (reward > 0) if terminal_on_reward else env_is_terminal(env, next_state)
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


def env_is_terminal(env, state):
    return env.is_terminal(state) if hasattr(env, "is_terminal") else False


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", choices=["gridworld", "sc2"], default="gridworld")
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    out = ROOT / "02_q_learning" / "results"
    out.mkdir(exist_ok=True)

    if args.env == "gridworld":
        from envs.gridworld import ACTIONS, GridWorld
        env = GridWorld()
        episodes = args.episodes or 300
        agent = QLearningAgent(env.state_shape, env.n_actions, args.alpha, args.gamma)
        rewards, lengths = train(env, agent, episodes, log_every=50)
        plot_learning_curve(lengths, out / "gridworld_learning_curve.png",
                            title="Q-learning on GridWorld", ylabel="steps to reach the beacon",
                            baselines={"shortest path = 5": 5})
        plot_policy_map(agent.Q, ACTIONS, out / "gridworld_policy.png",
                        title="Q-learning on GridWorld: greedy policy",
                        blocked=env.walls, goal=env.beacon)
        print("\nLearned greedy policy (compare with Lesson 1's value iteration):")
        env.render_policy({s: int(np.argmax(agent.Q[s])) for s in env.states()})
    else:
        from envs.move_to_beacon import DIRECTIONS, MoveToBeaconEnv
        env = MoveToBeaconEnv()
        episodes = args.episodes or 300
        agent = QLearningAgent(env.state_shape, env.n_actions, args.alpha, args.gamma)
        try:
            rewards, _ = train(env, agent, episodes, terminal_on_reward=True)
        finally:
            env.close()
        np.save(out / "sc2_q_table.npy", agent.Q)
        plot_policy_map(agent.Q, DIRECTIONS, out / "sc2_policy.png",
                        title="Q-learning on MoveToBeacon: greedy policy\n"
                              "(star = beacon, each cell = where the beacon is relative to the marine)",
                        center=(env.radius, env.radius))
        print(f"Saved the Q-table to {out / 'sc2_q_table.npy'}")
        plot_learning_curve(rewards, out / "sc2_learning_curve.png",
                            title="Q-learning on MoveToBeacon",
                            baselines={"random ≈ 1": 1, "scripted ≈ 23": 23})
    print(f"Saved plots to {out}")


if __name__ == "__main__":
    main()
