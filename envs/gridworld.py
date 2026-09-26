"""GridWorld: a tiny, pure-Python version of the MoveToBeacon mini-game.

A marine stands on a small grid and must walk to a beacon.

    . . . . .        M = marine (the agent)
    . . # . .        B = beacon (the goal, reward +1)
    . M # . .        # = wall  (cannot be entered)
    . . . . .
    . . . . B

Because the grid is tiny, we can compute everything exactly and check our math.
The environment runs instantly, so it is perfect for learning the concepts
before we move to the (much slower) StarCraft II game.

Interface (shared with `envs/move_to_beacon.py`):
    state = env.reset()                         # state is a tuple of ints
    next_state, reward, done = env.step(action) # action is an int
    env.state_shape                             # used to size the Q-table
    env.n_actions
"""
import random

# Actions: index -> (dx, dy). y grows downwards, like on a screen.
ACTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
ACTION_NAMES = ["up", "right", "down", "left"]
ACTION_ARROWS = ["^", ">", "v", "<"]


class GridWorld:
    def __init__(self, size=5, start=(1, 2), beacon=(4, 4),
                 walls=((2, 1), (2, 2)), slip=0.0, max_steps=50):
        """
        Args:
            size: the grid is size x size cells.
            start: (x, y) where the marine starts each episode.
            beacon: (x, y) of the goal cell.
            walls: cells the marine cannot enter.
            slip: probability that the marine moves in a random direction
                instead of the chosen one (0.0 = fully deterministic).
            max_steps: the episode is cut off after this many steps.
        """
        self.size = size
        self.start = start
        self.beacon = beacon
        self.walls = set(walls)
        self.slip = slip
        self.max_steps = max_steps
        self.state_shape = (size, size)
        self.n_actions = len(ACTIONS)

    # ------------------------------------------------------------------
    # The model of the world: P(s', r | s, a)
    # ------------------------------------------------------------------
    def states(self):
        """All non-wall cells."""
        return [(x, y) for y in range(self.size) for x in range(self.size)
                if (x, y) not in self.walls]

    def is_terminal(self, state):
        return state == self.beacon

    def _move(self, state, action):
        """Where would we end up if the move succeeds? Walls/edges block."""
        dx, dy = ACTIONS[action]
        x, y = state[0] + dx, state[1] + dy
        if not (0 <= x < self.size and 0 <= y < self.size) or (x, y) in self.walls:
            return state  # bumped into something: stay in place
        return (x, y)

    def transitions(self, state, action):
        """The full model: a list of (probability, next_state, reward, done).

        Only planning algorithms (Lesson 1) are allowed to call this.
        Learning algorithms (Lesson 2+) must discover the world by `step`.
        """
        outcomes = {}
        for a in range(self.n_actions):
            # Chosen action: (1 - slip) + slip/4.  Any other action: slip/4.
            p = self.slip / self.n_actions + (1.0 - self.slip if a == action else 0.0)
            if p == 0.0:
                continue
            nxt = self._move(state, a)
            outcomes[nxt] = outcomes.get(nxt, 0.0) + p
        return [(p, nxt, 1.0 if self.is_terminal(nxt) else 0.0, self.is_terminal(nxt))
                for nxt, p in outcomes.items()]

    # ------------------------------------------------------------------
    # The interactive interface: what a learning agent sees
    # ------------------------------------------------------------------
    def reset(self):
        self.state = self.start
        self.t = 0
        return self.state

    def step(self, action):
        if random.random() < self.slip:
            action = random.randrange(self.n_actions)
        self.state = self._move(self.state, action)
        self.t += 1
        reward = 1.0 if self.is_terminal(self.state) else 0.0
        done = self.is_terminal(self.state) or self.t >= self.max_steps
        return self.state, reward, done

    # ------------------------------------------------------------------
    # Pretty printing
    # ------------------------------------------------------------------
    def render_values(self, V):
        """Print a table of state values V[(x, y)]."""
        for y in range(self.size):
            row = []
            for x in range(self.size):
                if (x, y) in self.walls:
                    row.append("  ### ")
                elif self.is_terminal((x, y)):
                    row.append("    B ")  # terminal state: V = 0 by definition
                else:
                    row.append(f"{V[(x, y)]:6.3f}")
            print(" ".join(row))

    def render_policy(self, policy):
        """Print an arrow map. policy[(x, y)] is an action index."""
        for y in range(self.size):
            row = []
            for x in range(self.size):
                if (x, y) in self.walls:
                    row.append("#")
                elif self.is_terminal((x, y)):
                    row.append("B")
                else:
                    row.append(ACTION_ARROWS[policy[(x, y)]])
            print(" ".join(row))
