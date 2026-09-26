"""Plotting helpers shared by all lessons.

You can skip this file: it only draws pictures of the results.
"""
import matplotlib

matplotlib.use("Agg")  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(episode_rewards, path, title="Learning curve", window=20,
                        baselines=None, ylabel="total reward"):
    """A value per episode (e.g. total reward), plus a moving average to see the trend.

    baselines: optional dict {label: value} drawn as horizontal dashed lines.
    """
    rewards = np.asarray(episode_rewards, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rewards, color="#9ab", linewidth=1, label="per episode")
    if len(rewards) >= window:
        avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window - 1, len(rewards)), avg, color="#1f5fbf",
                linewidth=2, label=f"moving average ({window} episodes)")
    for label, value in (baselines or {}).items():
        ax.axhline(value, linestyle="--", linewidth=1, color="#888")
        ax.text(0, value, f" {label}", va="bottom", fontsize=9, color="#555")
    ax.set_xlabel("episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_policy_map(Q, directions, path, title="Greedy policy", center=None,
                    blocked=(), goal=None):
    """Draw the greedy action argmax_a Q(s, a) as an arrow in every cell.

    Q has shape (width, height, n_actions) and is indexed Q[x, y, a].
    directions: list of (dx, dy) for each action (y grows downwards).
    Cells never updated (all Q values zero) are left empty.
    The colour shows max_a Q(s, a): brighter = more valuable.
    """
    width, height, _ = Q.shape
    values = Q.max(axis=2)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(values.T, cmap="viridis", origin="upper")
    for x in range(width):
        for y in range(height):
            if (x, y) in blocked:
                ax.add_patch(plt.Rectangle((x - 0.5, y - 0.5), 1, 1, color="#444"))
                continue
            if goal is not None and (x, y) == goal:
                ax.text(x, y, "B", ha="center", va="center", color="w",
                        fontsize=12, fontweight="bold")
                continue
            if np.all(Q[x, y] == 0):
                continue
            dx, dy = directions[int(np.argmax(Q[x, y]))]
            norm = np.hypot(dx, dy)
            ax.arrow(x - 0.3 * dx / norm, y - 0.3 * dy / norm,
                     0.45 * dx / norm, 0.45 * dy / norm,
                     head_width=0.22, head_length=0.18, color="w",
                     length_includes_head=True)
    if center is not None:
        ax.plot(*center, marker="*", color="red", markersize=14)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
