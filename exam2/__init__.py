"""Shared code for the ENGR 6010 Second Test. Project code is imported from the pinned `project/` submodule."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

# Unitree's MuJoCo models (unitreerobotics/unitree_mujoco 1eb6642). Set GO2X_UNITREE_MUJOCO, or run
# project/scripts/robot/setup_laptop.sh, which clones it into project/third_party/unitree_mujoco.
MODEL_DIR = os.environ.get("GO2X_UNITREE_MUJOCO", str(PROJECT / "third_party/unitree_mujoco"))
WALKER = PROJECT / "results/phase2/mujoco_walk/mjx_s0/policy.pt"          # MuJoCo-trained walker (job 110)
ISAAC_WALKER = PROJECT / "results/phase2/walk_train/policy.pt"            # Isaac Lab walker (job 030)
DT = 0.05                                                                 # model and navigation step, s (20 Hz)
