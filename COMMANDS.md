# Command reference

Every command in the course, in order, with no explanations.
For the *why*, read each lesson's `README.md`.

Run all commands from the repository folder (`RL_starcraft`).
Commands marked **[SC2]** start StarCraft II. The others need no game.

---

## Every time you open a new terminal

```bash
cd RL_starcraft
.venv\Scripts\activate
```

macOS: `source .venv/bin/activate`

Windows PowerShell may refuse with *"running scripts is disabled on this system"*.
Either skip activation and write `.venv\Scripts\python.exe` wherever this page says
`python`, or allow local scripts once with
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

---

## Lesson 0: Setup (once)

```bash
git clone https://github.com/ucmoprj/RL_starcraft.git
cd RL_starcraft
python -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

Unzip [`mini_games.zip`](https://github.com/deepmind/pysc2/releases/download/v1.2/mini_games.zip)
into `StarCraft II\Maps\`, then:

```bash
python 00_setup/check_setup.py                         # [SC2]
```

---

## Lesson 1: The language of RL

```bash
python 01_rl_basics/policy_evaluation.py
python 01_rl_basics/value_iteration.py
```

---

## Lesson 2: Q-learning

Warm-up (a 4-cell corridor, every update printed):

```bash
python 02_q_learning/walkthrough.py --stage 1
python 02_q_learning/walkthrough.py --stage 2
```

Simulator: open [the Q-Learning Lab](https://ucmoprj.github.io/RL_starcraft/02_q_learning/q_learning_lab.html)
or double-click `02_q_learning/q_learning_lab.html`.

GridWorld:

```bash
python 02_q_learning/q_learning.py --env gridworld
```

StarCraft II:

```bash
python 02_q_learning/q_learning.py --env sc2 --episodes 300   # [SC2] train, ~6-7 min
python 02_q_learning/watch.py                                 # [SC2] watch the trained marine
python 02_q_learning/watch.py --random                        # [SC2] compare: random marine
python 02_q_learning/watch.py --grid                          # [SC2] state grid + Q values drawn on the game
python 02_q_learning/watch.py --show-q                        # [SC2] extra window: state grid + Q bar chart
python 02_q_learning/watch.py --grid --random                 # [SC2] grid view of a random marine
```

Experiments (section 9):

```bash
python 02_q_learning/q_learning.py --env gridworld --alpha 0.5
python 02_q_learning/q_learning.py --env gridworld --alpha 0.01
python 02_q_learning/q_learning.py --env gridworld --gamma 0.5
python 02_q_learning/q_learning.py --env gridworld --gamma 0.99
```

Outputs are written to `02_q_learning/results/`. Training in StarCraft II overwrites
`sc2_q_table.npy`, so copy it first if you want to keep it.

---

## Lesson 3: Deep Q-Networks

Warm-up (table vs function, every number printed):

```bash
python 03_dqn/walkthrough.py --stage 1
python 03_dqn/walkthrough.py --stage 2
```

GridWorld:

```bash
python 03_dqn/dqn.py --env gridworld
```

StarCraft II:

```bash
python 03_dqn/dqn.py --env sc2 --episodes 300                 # [SC2] train, ~8-10 min
python 03_dqn/compare.py                                      # Q-table vs DQN on all 289 states
python 03_dqn/watch_dqn.py                                    # [SC2] watch the trained DQN marine
python 03_dqn/watch_dqn.py --grid                             # [SC2] state grid + Q values drawn on the game
python 03_dqn/watch_dqn.py --show-q                           # [SC2] extra window: state grid + Q bar chart
python 03_dqn/watch_dqn.py --model no_replay                  # [SC2] the network trained without replay
python 03_dqn/watch_dqn.py --random                           # [SC2] compare: random marine
```

Do replay and the target network matter? (section 10):

```bash
python 03_dqn/dqn.py --env sc2 --episodes 300 --no-replay     # [SC2]
python 03_dqn/dqn.py --env sc2 --episodes 300 --no-target     # [SC2]
```

Outputs are written to `03_dqn/results/`.
