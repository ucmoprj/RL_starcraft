"""Check that everything needed for the course is installed.

Run from the repository root:

    python 00_setup/check_setup.py

It checks Python packages, finds StarCraft II and the mini-game maps, and finally
plays one short episode of MoveToBeacon with random actions.
"""
import importlib
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

OK, FAIL = "[ OK ]", "[FAIL]"


def check_packages():
    good = True
    for name in ["numpy", "matplotlib", "pysc2", "torch"]:
        try:
            module = importlib.import_module(name)
            print(f"{OK} {name} {getattr(module, '__version__', '')}")
        except ImportError:
            print(f"{FAIL} {name} is missing. Run: pip install -r requirements.txt")
            good = False
    return good


def find_sc2():
    candidates = [os.environ.get("SC2PATH"),
                  r"C:\Program Files (x86)\StarCraft II",
                  r"C:\Program Files\StarCraft II",
                  "/Applications/StarCraft II",
                  str(Path.home() / "StarCraftII")]
    for path in filter(None, candidates):
        if Path(path, "Versions").is_dir():
            print(f"{OK} StarCraft II found at {path}")
            return Path(path)
    print(f"{FAIL} StarCraft II not found. Install it, or set the SC2PATH "
          "environment variable to its folder.")
    return None


def check_maps(sc2_dir):
    maps = sc2_dir / "Maps" / "mini_games" / "MoveToBeacon.SC2Map"
    if maps.exists():
        print(f"{OK} mini-game maps found")
        return True
    print(f"{FAIL} {maps} not found. See 00_setup/README.md, step 4.")
    return False


def play_one_episode():
    import random
    from envs.move_to_beacon import MoveToBeaconEnv

    print("       Launching StarCraft II (the first launch can take a minute)...")
    env = MoveToBeaconEnv()
    try:
        env.reset()
        total, steps, done = 0.0, 0, False
        while not done:
            _, reward, done = env.step(random.randrange(env.n_actions))
            total += reward
            steps += 1
    finally:
        env.close()
    print(f"{OK} played one episode: {steps} steps, total reward {total:.0f}")


if __name__ == "__main__":
    ready = check_packages()
    sc2_dir = find_sc2()
    ready = ready and sc2_dir is not None and check_maps(sc2_dir)
    if not ready:
        sys.exit("\nFix the items marked [FAIL] and run this script again.")
    play_one_episode()
    print("\nAll good! Continue with Lesson 1: 01_rl_basics/README.md")
