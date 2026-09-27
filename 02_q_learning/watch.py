"""Watch the trained Q-learning marine play MoveToBeacon at human speed.

Run from the repository root, after training with `q_learning.py --env sc2`:

    python 02_q_learning/watch.py
    python 02_q_learning/watch.py --random     # compare with a random marine
    python 02_q_learning/watch.py --show-q     # also show the state grid and Q values live
    python 02_q_learning/watch.py --grid       # draw the state grid on the game itself
    python 02_q_learning/watch.py --grid --q-table 03_dqn/results/sc2_dqn_q_table.npy   # watch the DQN
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from envs.move_to_beacon import (ACTION_ARROWS, ACTION_NAMES, DIRECTIONS, SCREEN_SIZE,  # noqa: E402
                                 MoveToBeaconEnv)


class GameGrid:
    """Draws the 17 x 17 state grid inside the game, centred on the marine.

    Uses the StarCraft II debug-draw API, which paints lines, boxes and text into the
    game world (not into the feature layers the agent sees). Each step it shows:
    the grid (it moves with the marine), the current state's cell in orange, and the
    8 Q values around the marine, the chosen one in green.
    """

    def __init__(self, env):
        from s2clientprotocol import common_pb2, debug_pb2
        self.pb, self.debug = common_pb2, debug_pb2
        self.sc2 = env._env                    # the PySC2 environment inside our wrapper
        self.radius = env.radius
        # 64 screen pixels show 24 world units (PySC2's default camera width)
        self.cell = env.cell_pixels * 24 / SCREEN_SIZE

    def _point(self, x, y, z):
        return self.pb.Point(x=x, y=y, z=z)

    def _color(self, r, g, b):
        return self.debug.Color(r=r, g=g, b=b)

    def draw(self, state, action, Q):
        raw = self.sc2._obs[0].observation.raw_data
        marine = next((u for u in raw.units if u.alliance == 1), None)
        if marine is None:
            return
        mx, my, z = marine.pos.x, marine.pos.y, marine.pos.z + 0.1
        c, r, d = self.cell, self.radius, self.debug
        # Screen y grows downwards but world y grows upwards, so screen offset (i, j)
        # is at world (mx + i*c, my - j*c).
        lines, edge = [], (r + 0.5) * c
        grey = self._color(110, 110, 110)
        for k in range(-r, r + 2):
            v = (k - 0.5) * c
            lines.append(d.DebugLine(color=grey, line=d.Line(
                p0=self._point(mx + v, my - edge, z), p1=self._point(mx + v, my + edge, z))))
            lines.append(d.DebugLine(color=grey, line=d.Line(
                p0=self._point(mx - edge, my + v, z), p1=self._point(mx + edge, my + v, z))))
        i, j = state[0] - r, state[1] - r      # the current state: beacon offset in cells
        bx, by = mx + i * c, my - j * c
        boxes = [d.DebugBox(color=self._color(255, 160, 0),
                            min=self._point(bx - c / 2, by - c / 2, z),
                            max=self._point(bx + c / 2, by + c / 2, z + 0.2))]
        q = Q[state]
        text = [d.DebugText(color=self._color(255, 255, 255), size=14,
                            text=f"state ({i:+d}, {j:+d})  picks {ACTION_NAMES[action]}",
                            world_pos=self._point(mx - 2 * c, my + 2.2 * c, z))]
        for a, (dx, dy) in enumerate(DIRECTIONS):
            chosen = a == action
            text.append(d.DebugText(
                color=self._color(80, 255, 80) if chosen else self._color(220, 220, 220),
                size=13 if chosen else 10, text=f"{q[a]:.2f}",
                world_pos=self._point(mx + dx * 1.1 * c - 0.3, my - dy * 1.1 * c, z)))
        chosen_dx, chosen_dy = DIRECTIONS[action]
        lines.append(d.DebugLine(color=self._color(80, 255, 80), line=d.Line(
            p0=self._point(mx, my, z), p1=self._point(mx + chosen_dx * 0.8 * c, my - chosen_dy * 0.8 * c, z))))
        self.sc2._controllers[0].debug(d.DebugCommand(draw=d.DebugDraw(lines=lines, boxes=boxes, text=text)))


class QView:
    """A live window next to the game: where the agent thinks it is, and what it knows there.

    Left: the 17 x 17 state grid (the marine is always in the centre, each cell is a place
    the beacon could be). The colour and arrows are the learned policy, like sc2_policy.png.
    The orange ring marks the current state. Right: the 8 Q values of that state (Eq. 5 with
    epsilon = 0 picks the highest one, drawn in orange).
    """

    def __init__(self, Q):
        import matplotlib.pyplot as plt
        self.plt, self.Q = plt, Q
        width, height, n_actions = Q.shape
        self.radius = width // 2
        plt.ion()
        self.fig, (self.ax_grid, self.ax_q) = plt.subplots(
            1, 2, figsize=(11, 5.4), gridspec_kw={"width_ratios": [1, 1.1]})

        # the learned policy on the state grid (same picture as sc2_policy.png)
        self.ax_grid.imshow(Q.max(axis=2).T, cmap="viridis", origin="upper")
        for x in range(width):
            for y in range(height):
                if np.all(Q[x, y] == 0):
                    continue
                dx, dy = DIRECTIONS[int(np.argmax(Q[x, y]))]
                norm = np.hypot(dx, dy)
                self.ax_grid.arrow(x - 0.3 * dx / norm, y - 0.3 * dy / norm,
                                   0.45 * dx / norm, 0.45 * dy / norm, head_width=0.22,
                                   head_length=0.18, color="w", alpha=0.45,
                                   length_includes_head=True)
        r = self.radius
        self.ax_grid.plot(r, r, "o", ms=15, color="#20446E", mec="w")
        self.ax_grid.text(r, r, "M", ha="center", va="center", color="w", fontsize=9, fontweight="bold")
        self.beacon, = self.ax_grid.plot([], [], "o", ms=17, mfc="none", mec="#FFA000", mew=3)
        ticks = range(0, width, 2)
        self.ax_grid.set_xticks(ticks, [f"{t - r:+d}" if t != r else "0" for t in ticks], fontsize=8)
        self.ax_grid.set_yticks(ticks, [f"{t - r:+d}" if t != r else "0" for t in ticks], fontsize=8)
        self.ax_grid.set_xlabel("dx: beacon left (−) / right (+), in cells")
        self.ax_grid.set_ylabel("dy: beacon above (−) / below (+)")

        # the Q values of the current state
        self.bars = self.ax_q.bar(range(n_actions), np.zeros(n_actions), color="#9ab")
        self.ax_q.set_xticks(range(n_actions), [f"{a}\n{name}" for a, name in zip(ACTION_ARROWS, ACTION_NAMES)])
        self.ax_q.set_ylim(0, max(Q.max() * 1.15, 1e-6))
        self.ax_q.set_ylabel("Q(s, a)")
        self.ax_q.grid(axis="y", alpha=0.3)
        self.labels = [self.ax_q.text(a, 0, "", ha="center", va="bottom", fontsize=9) for a in range(n_actions)]
        self.fig.tight_layout(rect=(0, 0, 1, 0.94))

    def update(self, state, action, step, beacons):
        if not self.plt.fignum_exists(self.fig.number):
            return  # the window was closed; keep playing without it
        x, y = state
        dx, dy = x - self.radius, y - self.radius
        self.beacon.set_data([x], [y])
        where = "here" if dx == dy == 0 else ", ".join(
            f"{abs(v)} {name}" for v, name in ((dx, "right" if dx > 0 else "left"),
                                                (dy, "down" if dy > 0 else "up")) if v)
        self.ax_grid.set_title(f"state ({dx:+d}, {dy:+d}): beacon {where}")
        q = self.Q[x, y]
        for a, (bar, label) in enumerate(zip(self.bars, self.labels)):
            bar.set_height(q[a])
            bar.set_color("#E07B00" if a == action else "#9ab")
            label.set_position((a, q[a]))
            label.set_text(f"{q[a]:.2f}")
        self.ax_q.set_title(f"Q values in this state → picks {ACTION_ARROWS[action]} {ACTION_NAMES[action]}")
        self.fig.suptitle(f"step {step}   beacons {beacons:.0f}")
        self.plt.pause(0.001)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--random", action="store_true", help="ignore the Q-table")
    parser.add_argument("--show-q", action="store_true",
                        help="open a second window with the state grid and Q values")
    parser.add_argument("--grid", action="store_true",
                        help="draw the state grid and Q values on the game itself")
    parser.add_argument("--q-table", type=Path, default=ROOT / "02_q_learning" / "results" / "sc2_q_table.npy",
                        help="which Q-table to play, e.g. 03_dqn/results/sc2_dqn_q_table.npy")
    args = parser.parse_args()

    q_path = args.q_table
    if (args.show_q or args.grid or not args.random) and not q_path.exists():
        sys.exit(f"{q_path} not found. Train first: python 02_q_learning/q_learning.py --env sc2")
    Q = np.load(q_path) if q_path.exists() else None
    view = QView(Q) if args.show_q else None

    env = MoveToBeaconEnv(realtime=True)
    grid = GameGrid(env) if args.grid else None
    try:
        for episode in range(args.episodes):
            state, total, done, step = env.reset(), 0.0, False, 0
            while not done:
                # Greedy policy (Eq. 5 with epsilon = 0): always the best known action.
                action = random.randrange(env.n_actions) if args.random else int(np.argmax(Q[state]))
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
