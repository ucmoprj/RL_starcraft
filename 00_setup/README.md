# Lesson 0 — Setup

> **Goal:** install StarCraft II, Python and the course code, and check that an agent
> can control a marine in the game. **Time:** 30–60 minutes (mostly downloading).

These instructions are written for **Windows**. macOS works the same way with the
macOS paths shown in the notes. (Linux needs Blizzard's separate headless build and
an x86-64 CPU; it is not covered here.)

---

## Step 1 — Install StarCraft II (free)

1. Download the Battle.net app from <https://www.blizzard.com/apps/battle.net/desktop>
   and create a free Blizzard account.
2. In Battle.net, find **StarCraft II** and click **Install**. The free version is
   enough for the whole course.
3. Keep the default install location:
   - Windows: `C:\Program Files (x86)\StarCraft II`
   - macOS: `/Applications/StarCraft II`

   If you install somewhere else, set an environment variable `SC2PATH` to that folder.

## Step 2 — Install Python 3.11

Download Python **3.11** from <https://www.python.org/downloads/>.
On Windows, tick **"Add python.exe to PATH"** in the installer.

Check it in a new terminal:

```bash
python --version      # should print Python 3.11.x
```

## Step 3 — Get the course code and its packages

```bash
git clone https://github.com/ucmoprj/RL_starcraft.git
cd RL_starcraft

python -m venv .venv
.venv\Scripts\activate          # macOS: source .venv/bin/activate

pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

> **What is `.venv`?** A *virtual environment*: a private folder of Python packages
> for this project only, so it cannot break other Python projects on your computer.
> Run the `activate` line every time you open a new terminal.
>
> **Why a special line for `torch`?** It installs the smaller CPU-only version of
> PyTorch. The course does not need a graphics card. If you have an NVIDIA GPU, you
> can install the CUDA version from <https://pytorch.org> instead.

## Step 4 — Download the mini-game maps

1. Download [`mini_games.zip`](https://github.com/deepmind/pysc2/releases/download/v1.2/mini_games.zip).
2. Inside your StarCraft II folder, create a folder named `Maps` if it does not exist.
3. Unzip so that you get `StarCraft II\Maps\mini_games\MoveToBeacon.SC2Map`.

## Step 5 — Check everything

```bash
python 00_setup/check_setup.py
```

A StarCraft II window opens, a marine wanders around randomly for a few seconds,
and you should see:

```
[ OK ] numpy 2.x
[ OK ] matplotlib 3.x
[ OK ] pysc2
[ OK ] torch 2.x
[ OK ] StarCraft II found at C:\Program Files (x86)\StarCraft II
[ OK ] mini-game maps found
       Launching StarCraft II (the first launch can take a minute)...
[ OK ] played one episode: 239 steps, total reward 1

All good! Continue with Lesson 1: 01_rl_basics/README.md
```

(The total reward will be around 0–2: a random marine rarely finds the beacon.)

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `TypeError: Random.shuffle() takes 2 positional arguments` | You installed PySC2 from PyPI. Run `pip install -r requirements.txt` again; it installs the fixed GitHub version. |
| `TypeError: Popen.__init__() got an unexpected keyword argument 'extra_ports'` | A PySC2 bug on Windows. The course code fixes it automatically (`envs/_pysc2_fixes.py`), but PySC2's own command-line tools such as `python -m pysc2.bin.agent` still hit it. Use the course scripts instead. |
| `StarCraft II not found` | Set `SC2PATH` to your install folder, e.g. `setx SC2PATH "D:\Games\StarCraft II"` (Windows), then open a new terminal. |
| The game window opens and closes immediately | Start StarCraft II once from Battle.net, log in, then close it and try again. |

**Next:** [Lesson 1 — The language of reinforcement learning →](../01_rl_basics/)
