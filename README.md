# Reinforcement Learning with StarCraft II

**Learn reinforcement learning from the ground up by teaching a StarCraft II marine to play.**

This course is for beginners. You write every algorithm yourself, with the math next
to the code, and watch it learn inside a real game. No prior reinforcement learning or
deep learning experience is needed.

- **Math, explained.** Every equation is derived step by step and labelled
  (Eq. 1, Eq. 2, ...). The code says which line implements which equation.
- **Small first, then StarCraft.** Each idea is tried in a tiny GridWorld that runs
  instantly, then in StarCraft II.
- **No black boxes.** We do not call ready-made RL libraries. Each algorithm is about
  100–200 lines you can read in one sitting.

## Prerequisites

- Basic Python (functions, loops, lists, dictionaries).
- High-school math. Probability and derivatives are reviewed where they are needed.
- A Windows or macOS computer. No graphics card needed.

## Lessons

| # | Lesson | Algorithm | Environment | Status |
|---|---|---|---|---|
| 0 | [Setup](00_setup/) | — | StarCraft II | ✅ |
| 1 | [The language of RL](01_rl_basics/) | policy evaluation, value iteration | GridWorld | ✅ |
| 2 | [Learning from experience](02_q_learning/) | Q-learning | GridWorld → MoveToBeacon | ✅ |
| 3 | Deep Q-networks | DQN | CollectMineralShards | planned |
| 4 | Learning a policy directly | REINFORCE | CollectMineralShards | planned |
| 5 | Actor and critic | A2C | DefeatRoaches | planned |
| 6 | Stable policy updates | PPO | DefeatRoaches | planned |

## How to use this course

1. Follow [Lesson 0](00_setup/) to install everything.
   Just want the commands? See [COMMANDS.md](COMMANDS.md).
2. Read each lesson's `README.md` from top to bottom. Run the commands as you meet them.
3. Before running an experiment, **predict** what will happen. Then check.
4. Do the exercises at the end of each lesson. Answers are hidden under "Answers".

All commands are run from the repository root, with the virtual environment activated.

## Repository layout

```
RL_starcraft/
├── 00_setup/        installation guide and a check script
├── 01_rl_basics/    lesson notes + code
├── 02_q_learning/   lesson notes + code + results/
├── envs/            the environments: GridWorld and MoveToBeacon
└── common/          plotting helpers
```

## Acknowledgements

- [PySC2](https://github.com/google-deepmind/pysc2) by DeepMind, the StarCraft II
  learning environment.
- Sutton & Barto, [*Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html),
  the textbook this course follows for notation.

StarCraft is a trademark of Blizzard Entertainment. This course is not affiliated with
or endorsed by Blizzard.
