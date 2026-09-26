"""A beginner-friendly wrapper around the StarCraft II MoveToBeacon mini-game.

The real PySC2 interface is huge: dozens of image layers as observations and
hundreds of actions with arguments. That is great for research, but it hides
the reinforcement learning ideas we want to study. This wrapper shrinks the
game down to the same simple interface as `envs/gridworld.py`:

    state = env.reset()
    next_state, reward, done = env.step(action)

State:
    Where is the beacon *relative to* the marine, measured in grid cells?
    state = (dx_cell, dy_cell), each in [-R, R], shifted to [0, 2R] so it can
    index a table. (R, R) means "the beacon is right here".

Actions (8 directions):
    0=N, 1=NE, 2=E, 3=SE, 4=S, 5=SW, 6=W, 7=NW
    The marine is ordered to move `step_pixels` pixels in that direction.

Reward:
    +1 every time the marine reaches the beacon (the beacon then jumps to a new
    random place). An episode lasts 120 game seconds (about 240 agent steps).
"""
import numpy as np
from absl import flags
from pysc2.env import sc2_env
from pysc2.lib import actions, features

from envs import _pysc2_fixes  # noqa: F401  (applies small bug fixes)

# PySC2 reads its settings from command-line flags. Parse an empty command line
# once so the library works inside ordinary Python scripts.
if not flags.FLAGS.is_parsed():
    flags.FLAGS(["course"])

SCREEN_SIZE = 64
DIRECTIONS = [(0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1)]
ACTION_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
ACTION_ARROWS = ["↑", "↗", "→", "↘", "↓", "↙", "←", "↖"]

_SELF = features.PlayerRelative.SELF
_NEUTRAL = features.PlayerRelative.NEUTRAL


class MoveToBeaconEnv:
    def __init__(self, cell_pixels=4, radius=8, step_pixels=8,
                 visualize=False, realtime=False, step_mul=8):
        """
        Args:
            cell_pixels: size of one grid cell, in screen pixels.
            radius: the state grid covers [-radius, radius] cells in x and y.
                Anything farther away is clipped to the edge.
            step_pixels: how far each move order sends the marine.
            visualize: open the PySC2 feature-layer viewer window.
            realtime: play at human speed (nice for watching, slow for training).
            step_mul: game frames between two agent decisions (8 = ~0.36s).
        """
        self.cell_pixels = cell_pixels
        self.radius = radius
        self.step_pixels = step_pixels
        self.state_shape = (2 * radius + 1, 2 * radius + 1)
        self.n_actions = len(DIRECTIONS)

        self._env = sc2_env.SC2Env(
            map_name="MoveToBeacon",
            players=[sc2_env.Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(
                    screen=SCREEN_SIZE, minimap=SCREEN_SIZE),
                use_feature_units=True),
            step_mul=step_mul,
            visualize=visualize,
            realtime=realtime,
        )

    # ------------------------------------------------------------------
    def reset(self):
        self._timestep = self._env.reset()[0]
        # The marine must be selected before it will obey move orders.
        self._timestep = self._env.step([actions.FUNCTIONS.select_army("select")])[0]
        return self._state()

    def step(self, action):
        marine = self._position(_SELF)
        dx, dy = DIRECTIONS[action]
        target = np.clip(marine + np.array([dx, dy]) * self.step_pixels,
                         0, SCREEN_SIZE - 1)

        if actions.FUNCTIONS.Move_screen.id in self._timestep.observation.available_actions:
            order = actions.FUNCTIONS.Move_screen("now", target.tolist())
        else:
            order = actions.FUNCTIONS.select_army("select")

        self._timestep = self._env.step([order])[0]
        reward = float(self._timestep.reward)
        done = self._timestep.last()
        return self._state(), reward, done

    def close(self):
        self._env.close()

    # ------------------------------------------------------------------
    def _position(self, alliance):
        """Screen (x, y) of the first unit with the given alliance."""
        for unit in self._timestep.observation.feature_units:
            if unit.alliance == alliance:
                return np.array([unit.x, unit.y])
        return np.array([SCREEN_SIZE // 2, SCREEN_SIZE // 2])

    def _state(self):
        offset = self._position(_NEUTRAL) - self._position(_SELF)  # beacon - marine
        cells = np.round(offset / self.cell_pixels).astype(int)
        cells = np.clip(cells, -self.radius, self.radius)
        return (int(cells[0] + self.radius), int(cells[1] + self.radius))
