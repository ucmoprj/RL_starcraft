"""Lesson 1, part B: what is the best possible policy? (Value iteration)

Run from the repository root:

    python 01_rl_basics/value_iteration.py
    python 01_rl_basics/value_iteration.py --slip 0.2   # a slippery world
    python 01_rl_basics/value_iteration.py --gamma 0.5  # a short-sighted agent

We apply the Bellman OPTIMALITY equation (Eq. 8 in the README) until the
values stop changing, then read off the optimal policy (Eq. 9).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.gridworld import GridWorld  # noqa: E402


def q_from_v(env, V, s, a, gamma):
    """Q(s,a) = sum_{s',r} p(s',r|s,a) * [ r + gamma * V(s') ]   (Eq. 7)"""
    q = 0.0
    for p, s_next, r, done in env.transitions(s, a):
        future = 0.0 if done else V[s_next]
        q += p * (r + gamma * future)
    return q


def value_iteration(env, gamma=0.9, tolerance=1e-6):
    V = {s: 0.0 for s in env.states()}
    sweep = 0
    while True:
        biggest_change = 0.0
        for s in env.states():
            if env.is_terminal(s):
                continue
            # Bellman optimality equation (Eq. 8): V(s) = max_a Q(s,a)
            new_value = max(q_from_v(env, V, s, a, gamma) for a in range(env.n_actions))
            biggest_change = max(biggest_change, abs(new_value - V[s]))
            V[s] = new_value
        sweep += 1
        if biggest_change < tolerance:
            print(f"Converged after {sweep} sweeps.\n")
            break
    # Optimal policy (Eq. 9): in every state, pick the action with the largest Q.
    policy = {}
    for s in env.states():
        if not env.is_terminal(s):
            qs = [q_from_v(env, V, s, a, gamma) for a in range(env.n_actions)]
            policy[s] = qs.index(max(qs))
    return V, policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--slip", type=float, default=0.0)
    args = parser.parse_args()

    env = GridWorld(slip=args.slip)
    V, policy = value_iteration(env, gamma=args.gamma)
    print(f"V* (gamma = {args.gamma}, slip = {args.slip}):")
    env.render_values(V)
    print("\nOptimal policy (B = beacon, # = wall):")
    env.render_policy(policy)
