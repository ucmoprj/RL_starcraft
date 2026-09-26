"""Lesson 1, part A: how good is a policy? (Iterative policy evaluation)

Run from the repository root:

    python 01_rl_basics/policy_evaluation.py

We evaluate the *random* policy (each of the 4 moves with probability 1/4)
by applying the Bellman expectation equation (Eq. 6 in the README) over and
over until the values stop changing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.gridworld import GridWorld  # noqa: E402


def evaluate_policy(env, policy, gamma=0.9, tolerance=1e-6):
    """Return V^pi as a dict {state: value}.

    policy(state) must return a list of action probabilities pi(a|s).
    """
    V = {s: 0.0 for s in env.states()}
    sweep = 0
    while True:
        biggest_change = 0.0
        for s in env.states():
            if env.is_terminal(s):
                continue  # V(terminal) = 0: nothing more can happen there
            # Bellman expectation equation (Eq. 6):
            # V(s) = sum_a pi(a|s) * sum_{s',r} p(s',r|s,a) * [ r + gamma * V(s') ]
            new_value = 0.0
            for a, pi_a in enumerate(policy(s)):
                for p, s_next, r, done in env.transitions(s, a):
                    future = 0.0 if done else V[s_next]
                    new_value += pi_a * p * (r + gamma * future)
            biggest_change = max(biggest_change, abs(new_value - V[s]))
            V[s] = new_value
        sweep += 1
        if biggest_change < tolerance:
            print(f"Converged after {sweep} sweeps.\n")
            return V


def random_policy(state):
    return [0.25, 0.25, 0.25, 0.25]


if __name__ == "__main__":
    env = GridWorld()
    V = evaluate_policy(env, random_policy)
    print("V^pi for the RANDOM policy (gamma = 0.9):")
    env.render_values(V)
    print("\nNotice: values are small everywhere, and larger near the beacon")
    print("(bottom-right). A random marine does reach the beacon, just slowly.")
