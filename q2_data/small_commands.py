"""Q2: the Phase 1 single-axis motion tests (project main, scripts/motion/02_forward.py, 03_rotate.py, 04_lateral.py)
replayed in each model and compared with the real Go2 X runs of 2026-10-17 (project main, LOG.md, Mac robot session).

Each test drives one axis open loop for distance / speed seconds through the safety layer's rate limits, then holds
zero for 1 s; the travelled distance (or angle) is compared with the commanded one. The real values are single runs
(n = 1 each), the robot's own odometry and IMU.

    python q2_data/small_commands.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "q2_data"))
from exam2 import ISAAC_WALKER, WALKER  # noqa: E402
from exam2.dyn import LearnedModel, MujocoModel, PhysicsModel  # noqa: E402
from l_path_replay import Runner  # noqa: E402

# (name, axis, speed, target, real result) ; real results from project main LOG.md
TESTS = [("forward 0.2 m/s, 0.5 m", 0, 0.2, 0.5, 0.327),
         ("rotate 0.3 rad/s, 90 deg", 2, 0.3, math.pi / 2, math.radians(13)),
         ("rotate 0.5 rad/s, 90 deg", 2, 0.5, math.pi / 2, math.radians(42)),
         ("lateral 0.15 m/s, 0.3 m", 1, 0.15, 0.3, 0.009),
         ("lateral 0.2 m/s, 0.3 m", 1, 0.2, 0.3, 0.04),
         ("lateral 0.3 m/s, 0.3 m", 1, 0.3, 0.3, 0.17)]


def run_test(model, rng, axis, speed, target):
    model.reset(rng, np.array([0.0, 0.0, float(rng.uniform(-math.pi, math.pi))]))
    r = Runner(model)
    p0, y0 = model.pose[:2].copy(), model.pose[2]
    cmd = np.zeros(3)
    cmd[axis] = speed
    rise, t = None, 0.0
    yaw_acc, yp = 0.0, y0
    for _ in range(int(round(target / speed / 0.05))):
        r.send(cmd)
        t += 0.05
        yaw_acc += (model.pose[2] - yp + math.pi) % (2 * math.pi) - math.pi
        yp = model.pose[2]
        if rise is None and model.vel[axis] >= 0.5 * speed:
            rise = t
    for _ in range(20):
        r.send((0, 0, 0))
        yaw_acc += (model.pose[2] - yp + math.pi) % (2 * math.pi) - math.pi
        yp = model.pose[2]
    if axis == 2:
        return abs(yaw_acc), rise
    d = model.pose[:2] - p0
    c, s = math.cos(y0), math.sin(y0)
    return abs((c * d[0] + s * d[1]) if axis == 0 else (-s * d[0] + c * d[1])), rise


def main():
    fit = json.loads((ROOT / "q1_vision_ml/results/ensemble_eval.json").read_text())["lag_delay_fitted"]
    models = {"physics model, Phase 1 ranges": (lambda: PhysicsModel(), 10),
              "physics model, fitted to the MuJoCo dog": (lambda: PhysicsModel(fixed=(fit["tau_s"], fit["delay_s"])), 1),
              "learned ensemble": (lambda: LearnedModel(ROOT / "q1_vision_ml/results/ensemble.npz"), 10),
              "MuJoCo dog, MuJoCo walker (mjx_s0)": (lambda: MujocoModel(policy=str(WALKER)), 5),
              "MuJoCo dog, Isaac Lab walker": (lambda: MujocoModel(policy=str(ISAAC_WALKER)), 5)}
    res = {"real Go2 X (Unitree controller)": {name: {"fraction": real / tgt, "n": 1} for name, _, _, tgt, real in TESTS}}
    for mname, (make, n) in models.items():
        m = make()
        res[mname] = {}
        for name, axis, speed, target, _ in TESTS:
            fr = []
            for i in range(n):
                got, rise = run_test(m, np.random.default_rng(500 + i), axis, speed, target)
                fr.append(got / target)
            res[mname][name] = {"fraction": float(np.mean(fr)), "sd": float(np.std(fr, ddof=1)) if n > 1 else 0.0, "n": n}
        print(mname, {k: round(v["fraction"], 2) for k, v in res[mname].items() if isinstance(v, dict)}, flush=True)
    out = ROOT / "q2_data/results"
    (out / "small_commands.json").write_text(json.dumps(res, indent=1))
    names = [t[0] for t in TESTS]
    lines = ["# Small single-axis commands: share of the commanded travel delivered", "",
             "Real: one run each on the Go2 X (2026-10-17, odometry and IMU). Simulated: mean over n runs "
             "(sd in the JSON). 1.00 means the full distance or angle.", "",
             "| world | " + " | ".join(names) + " |", "| :--- | " + " | ".join("---:" for _ in names) + " |"]
    for mname, r in res.items():
        lines.append(f"| {mname} | " + " | ".join(f"{r[n]['fraction']:.2f}" for n in names) + " |")
    (out / "small_commands.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
