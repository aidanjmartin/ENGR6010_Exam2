"""Loader for the real Go2 X navigation runs of 2026-11-06 (q2_data/dataset/real_runs_20261106/).

Each run folder was written by the project's robot runner (project/scripts/robot/run_nav_policy.py) over WebRTC on the
robot's Wi-Fi hotspot, with the overhead camera giving the robot pose. Times are the MacBook's monotonic clock.

- state.jsonl: sport-mode state at 20 Hz. `velocity` is in the body frame (vx, vy; checked by integrating it against
  `position`), `yaw_speed` is the yaw rate, `position` and `imu_state.rpy[2]` are the robot's odometry.
- commands.jsonl: every command sent at 20 Hz, before (`requested`) and after (`sent`) the safety layer's limits.
- events.jsonl: one `policy_step` event per policy step (10 Hz), with the pose the policy used and its source
  (`camera` or `odometry` when the camera had no fresh tag fix).
- policy.npz: per policy step, the same pose, the raw odometry (x, y, yaw) and the commands.
"""
import json
import math
from pathlib import Path

import numpy as np

from exam2 import ROOT

RUNS_DIR = ROOT / "q2_data/dataset/real_runs_20261106"
MOTION_DIR = ROOT / "q2_data/dataset/real_motion_20261017"
COLORS = ["red", "blue", "green", "yellow"]           # chair color index in a scenario (project config/atrium.yaml)


def wrap(a):
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def odom_to_square(start_square, first_odom, odom):
    """project/go2x/deploy/nav_runtime.py OdomToSquare, vectorized: odometry (N,3) into the square frame."""
    s, o, odom = np.asarray(start_square, float), np.asarray(first_odom, float), np.atleast_2d(odom)
    dx, dy = odom[:, 0] - o[0], odom[:, 1] - o[1]
    c, si = math.cos(-o[2]), math.sin(-o[2])
    lx, ly = c * dx - si * dy, si * dx + c * dy
    c2, s2 = math.cos(s[2]), math.sin(s[2])
    return np.stack([s[0] + c2 * lx - s2 * ly, s[1] + s2 * lx + c2 * ly, wrap(s[2] + odom[:, 2] - o[2])], 1)


def load_run(folder):
    """Everything one analysis needs from a run folder, or None when the run has no policy steps."""
    folder = Path(folder)
    if not (folder / "run.json").exists():
        return None
    run = json.loads((folder / "run.json").read_text())
    events = read_jsonl(folder / "events.jsonl")
    steps = [e for e in events if e["event"] == "policy_step"]
    if not steps:
        return None
    z = np.load(folder / "policy.npz")
    states = read_jsonl(folder / "state.jsonl")
    cmds = read_jsonl(folder / "commands.jsonl")
    perc = read_jsonl(folder / "perception.jsonl")
    end = [e for e in events if e["event"] == "policy_end"]
    st_t = np.array([s["t_mono_ns"] for s in states]) * 1e-9
    d = [s["data"] for s in states]
    return {
        "name": folder.name.split("_")[0],
        "folder": folder,
        "target": run.get("source", "").split("target '")[-1].split("'")[0],
        "outcome": run.get("final", run.get("outcome", "")),
        "success": str(run.get("final", "")).startswith("success"),
        "scenario": perc[-1]["scenario"] if perc else run["scenario"],
        "t": np.array([s["t_mono_ns"] for s in steps]) * 1e-9,
        "pose": np.array([s["pose"] for s in steps], float),
        "camera": np.array([s["pose_source"] == "camera" for s in steps]),
        "odom": z["odom"].astype(float),
        "t_end": end[0]["t_mono_ns"] * 1e-9 if end else steps[-1]["t_mono_ns"] * 1e-9,
        "cmd_t": np.array([c["t_mono_ns"] for c in cmds]) * 1e-9,
        "cmd": np.array([c["sent"] for c in cmds], float),
        "st_t": st_t,
        "vel": np.array([[s["velocity"][0], s["velocity"][1], s["yaw_speed"]] for s in d], float),
        "st_odom": np.array([[s["position"][0], s["position"][1], s["imu_state"]["rpy"][2]] for s in d], float),
    }


def load_runs():
    runs = [load_run(f) for f in sorted(RUNS_DIR.glob("*_nav_policy_webrtc"))]
    return [r for r in runs if r is not None]


def path_length(xy):
    return float(np.hypot(*np.diff(np.asarray(xy)[:, :2], axis=0).T).sum()) if len(xy) > 1 else 0.0


def load_motion():
    """The Phase 1 motion tests of 2026-10-17 that sent motion commands (forward, rotate, lateral, the Phase 1 RL
    policy, L-path trials 1 to 4): per run, the state and command streams."""
    out = []
    for folder in sorted(MOTION_DIR.glob("*_webrtc")):
        cmds = read_jsonl(folder / "commands.jsonl")
        states = read_jsonl(folder / "state.jsonl")
        if not cmds or not states or not any(any(abs(v) > 0 for v in c["sent"]) for c in cmds):
            continue
        d = [s["data"] for s in states]
        cmd_t = np.array([c["t_mono_ns"] for c in cmds]) * 1e-9
        out.append({"name": folder.name.split("_", 1)[1].rsplit("_", 1)[0], "folder": folder,
                    "cmd_t": cmd_t, "cmd": np.array([c["sent"] for c in cmds], float),
                    "st_t": np.array([s["t_mono_ns"] for s in states]) * 1e-9,
                    "vel": np.array([[s["velocity"][0], s["velocity"][1], s["yaw_speed"]] for s in d], float),
                    "t_from": float(cmd_t[0]), "t_to": float(cmd_t[-1])})
    return out
