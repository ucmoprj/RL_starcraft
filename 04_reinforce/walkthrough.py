"""Lesson 4 walkthrough: watch REINFORCE change probabilities, by hand.

The corridor from Lesson 2. The marine starts in cell 0; the beacon is after cell 2.

    [0] [1] [2] [B]

In each cell the policy has two probabilities, pi(left) and pi(right). They come from
two "preferences" h through softmax:  pi(a) = exp(h(a)) / (exp(h(left)) + exp(h(right))).
All h start at 0, so every cell starts at 50% / 50%.

Run from the repository root:

    python 04_reinforce/walkthrough.py --stage 1   # one episode, every number printed
    python 04_reinforce/walkthrough.py --stage 2   # 30 episodes, watch pi(right) climb
"""
import argparse

import numpy as np

BEACON, GAMMA = 3, 0.9
NAMES = ["left", "right"]


def softmax(h):
    e = np.exp(h - h.max())
    return e / e.sum()


def step(s, a):
    s2 = max(0, s - 1) if a == 0 else s + 1
    return s2, (1.0 if s2 == BEACON else 0.0), s2 == BEACON


def returns(rewards):
    """G_t = r_t + gamma G_{t+1}, computed backwards (README Eq. 2)."""
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + GAMMA * G
        out.append(G)
    return out[::-1]


def update(h, episode, alpha, verbose):
    """REINFORCE (README Eq. 3): h(s) += alpha G_t (onehot(a_t) - pi(*|s_t)) for every step."""
    G = returns([r for _, _, r in episode])
    old = {s: softmax(h[s]) for s in range(BEACON)}   # the policy that played this episode
    for t, ((s, a, _), g) in enumerate(zip(episode, G)):
        grad = -old[s].copy()
        grad[a] += 1.0                                # grad  log pi(a|s) for a softmax
        h[s] += alpha * g * grad
        if verbose:
            print(f"  t={t}  cell {s} --{NAMES[a]:>5}   G_t = {g:.3f}   "
                  f"push {NAMES[a]} up by alpha*G*(1-pi) = {alpha} x {g:.3f} x {1 - old[s][a]:.2f} = {alpha * g * (1 - old[s][a]):.3f}")


def show(h):
    print("          pi(left)  pi(right)")
    for s in range(BEACON):
        p = softmax(h[s])
        print(f"  cell {s}   {p[0]:6.1%}   {p[1]:6.1%}")


def stage1(alpha):
    h = np.zeros((BEACON, 2))
    # One possible episode, with a detour: right, LEFT, right, right, right.
    actions = [1, 0, 1, 1, 1]
    s, episode = 0, []
    for a in actions:
        s2, r, _ = step(s, a)
        episode.append((s, a, r))
        s = s2
    print("Start: every cell is 50% / 50%.\n")
    show(h)
    print("\nThe episode:  0 -right-> 1 -LEFT-> 0 -right-> 1 -right-> 2 -right-> B  (reward 1 at the end)\n")
    print("Returns (Eq. 2): the reward 1, discounted back to each step:")
    print("  " + "  ".join(f"G_{t} = {g:.3f}" for t, g in enumerate(returns([r for _, _, r in episode]))))
    print("\nUpdates (Eq. 3):")
    update(h, episode, alpha, verbose=True)
    print("\nAfter one episode:")
    show(h)
    print("\nNotice t=1: the step LEFT in cell 1 was a mistake, but its G is 0.729 > 0,")
    print("so REINFORCE pushed 'left' UP too. It only knows 'this episode ended well'.")
    print("Cell 1 still ends up preferring right, because 'right' was pushed with a larger G (0.900).")
    print("Mistakes get rewarded a little whenever the episode succeeds anyway. That noise is")
    print("why REINFORCE needs many episodes, and what Lesson 5 (a baseline / critic) fixes.")


def stage2(alpha, episodes, seed):
    rng = np.random.default_rng(seed)
    h = np.zeros((BEACON, 2))
    print(f"{episodes} episodes, actions sampled from the policy.\n")
    print("  episode  steps   pi(right) in cell 0     cell 1     cell 2")
    for ep in range(1, episodes + 1):
        s, done, episode = 0, False, []
        while not done and len(episode) < 100:
            a = int(rng.choice(2, p=softmax(h[s])))
            s2, r, done = step(s, a)
            episode.append((s, a, r))
            s = s2
        update(h, episode, alpha, verbose=False)
        p = [softmax(h[c])[1] for c in range(BEACON)]
        print(f"  {ep:7d}  {len(episode):5d}   {p[0]:16.1%}   {p[1]:8.1%}   {p[2]:8.1%}")
    print("\npi(right) climbs in every cell and most episodes shrink to the shortest path, 3 steps.")
    print("But not smoothly: look for episodes with more than 3 steps. Their detour steps were")
    print("rewarded too (the episode still reached the beacon), so some probabilities DROPPED.")
    print("That noise is REINFORCE's main weakness, and the reason for Lesson 5.")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", type=int, choices=[1, 2], default=1)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.stage == 1:
        stage1(args.alpha)
    else:
        stage2(args.alpha, args.episodes, args.seed)


if __name__ == "__main__":
    main()
