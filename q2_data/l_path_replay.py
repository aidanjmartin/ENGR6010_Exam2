"""Q2: the Phase 1 L-path test (project main, scripts/motion/07_l_path.py) replayed in simulation and compared with
the 4 real trials on the Go2 X (project/results/robot/l_path_trials_webrtc.md, 2026-10-17, robot odometry).

The script is reproduced step for step: 20 Hz commands through the safety layer's rate limits (0.5 m/s^2, 1.0 rad/s^2),
forward until odometry shows 0.5 m minus the predicted coast (speed x 0.2 s plus the ramp-down), turn left at
0.5 rad/s until the yaw has changed 90 degrees minus the coast, forward again, a 1 s hold after each segment,
8 s and 10 s timeouts. Odometry is the model's own pose, as on the robot.

    python q2_data/l_path_replay.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exam2 import ISAAC_WALKER, WALKER  # noqa: E402
from exam2.dyn import LearnedModel, MujocoModel, PhysicsModel  # noqa: E402

VX, WZ, DIST, ANGLE, T_RESP = 0.2, 0.5, 0.5, 90.0, 0.2
ACC = np.array([0.5, 0.5, 1.0])
PERIOD = 0.05
# project/results/robot/l_path_trials_webrtc.md, trials 1 to 4 (trial 5 never connected)
REAL = {"leg1_m": [0.412, 0.428, 0.341, 0.294], "turn_deg": [82.5, 80.5, 80.5, 85.8], "leg2_m": [0.450, 0.445, 0.456, 0.062],
        "end_forward_m": [0.454, 0.489, 0.425, 0.286], "end_left_m": [0.477, 0.530, 0.521, 0.127],
        "heading_deg": [82.3, 85.3, 82.7, 85.4], "timeouts": 3}


class Runner:
    def __init__(self, model):
        self.m = model
        self.cmd = np.zeros(3)

    def send(self, target):
        step = ACC * PERIOD
        self.cmd = self.cmd + np.clip(np.asarray(target, float) - self.cmd, -step, step)
        self.m.step(self.cmd, PERIOD)

    def hold(self, s):
        for _ in range(int(round(s / PERIOD))):
            self.send((0, 0, 0))

    def leg(self, timeout):
        p0 = self.m.pose[:2].copy()
        k, trav, lead = 0, 0.0, 0.0
        while trav + lead < DIST and k * PERIOD < timeout:
            self.send((VX, 0, 0))
            trav = float(np.hypot(*(self.m.pose[:2] - p0)))
            lead = abs(self.m.vel[0]) * T_RESP + VX ** 2 / (2 * ACC[0])
            k += 1
        timed_out = trav + lead < DIST
        self.hold(1.0)
        return float(np.hypot(*(self.m.pose[:2] - p0))), timed_out

    def turn(self, timeout):
        yp = self.m.pose[2]
        turned, k, lead = 0.0, 0, 0.0
        target = math.radians(ANGLE)
        while abs(turned) + lead < target and k * PERIOD < timeout:
            self.send((0, 0, WZ))
            turned += (self.m.pose[2] - yp + math.pi) % (2 * math.pi) - math.pi
            yp = self.m.pose[2]
            lead = abs(self.m.vel[2]) * T_RESP + WZ ** 2 / (2 * ACC[2])
            k += 1
        timed_out = abs(turned) + lead < target
        self.hold(1.0)
        turned += (self.m.pose[2] - yp + math.pi) % (2 * math.pi) - math.pi
        return math.degrees(turned), timed_out


def trial(model, rng):
    model.reset(rng, np.array([0.0, 0.0, float(rng.uniform(-math.pi, math.pi))]))
    r = Runner(model)
    p0, y0 = model.pose[:2].copy(), model.pose[2]
    l1, t1 = r.leg(8.0)
    tu, t2 = r.turn(10.0)
    l2, t3 = r.leg(8.0)
    d = model.pose[:2] - p0
    c, s = math.cos(y0), math.sin(y0)
    return {"leg1_m": l1, "turn_deg": tu, "leg2_m": l2, "end_forward_m": c * d[0] + s * d[1],
            "end_left_m": -s * d[0] + c * d[1],
            "heading_deg": math.degrees((model.pose[2] - y0 + math.pi) % (2 * math.pi) - math.pi),
            "timeouts": int(t1) + int(t2) + int(t3)}


def summary(rows):
    out = {}
    for k in ("leg1_m", "turn_deg", "leg2_m", "end_forward_m", "end_left_m", "heading_deg"):
        v = np.array([r[k] for r in rows])
        out[k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}
    out["timeouts"] = int(sum(r["timeouts"] for r in rows))
    return out


def main():
    ens = ROOT / "q1_vision_ml/results/ensemble.npz"
    fit = json.loads((ROOT / "q1_vision_ml/results/ensemble_eval.json").read_text())["lag_delay_fitted"]
    N = 10
    models = {
        "physics model, Phase 1 ranges": (lambda: PhysicsModel(), N),
        "physics model, fitted to the MuJoCo dog": (lambda: PhysicsModel(fixed=(fit["tau_s"], fit["delay_s"])), 1),
        "learned ensemble": (lambda: LearnedModel(ens), N),
        "MuJoCo dog, MuJoCo walker (mjx_s0)": (lambda: MujocoModel(policy=str(WALKER)), N),
        "MuJoCo dog, Isaac Lab walker": (lambda: MujocoModel(policy=str(ISAAC_WALKER)), N),
    }
    res = {"real Go2 X (Unitree controller)": {k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "n": 4}
                                               for k, v in REAL.items() if k != "timeouts"}}
    res["real Go2 X (Unitree controller)"]["timeouts"] = REAL["timeouts"]
    for name, (make, n) in models.items():
        m = make()
        res[name] = summary([trial(m, np.random.default_rng(100 + i)) for i in range(n)])
        print(name, {k: round(v["mean"], 3) for k, v in res[name].items() if isinstance(v, dict)}, flush=True)
    out = ROOT / "q2_data/results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "l_path_replay.json").write_text(json.dumps(res, indent=1))
    keys = [("leg1_m", "leg 1 (m)"), ("turn_deg", "turn (deg)"), ("leg2_m", "leg 2 (m)"), ("end_forward_m", "end forward (m)"),
            ("end_left_m", "end left (m)"), ("heading_deg", "heading change (deg)")]
    lines = ["# L-path test: real Go2 X against simulation", "",
             "Mean ± sd over trials (n per row). Ideal: legs 0.5 m, turn 90 deg, end +0.5 m forward and +0.5 m left. "
             "Real values are the robot's odometry and IMU, as in the simulated rows.", "",
             "| world | n | " + " | ".join(k[1] for k in keys) + " | segment timeouts |",
             "| :--- | ---: | " + " | ".join("---:" for _ in keys) + " | ---: |"]
    for name, r in res.items():
        n = r["leg1_m"]["n"]
        cells = [f"{r[k]['mean']:.3f} ± {r[k]['sd']:.3f}" if "deg" not in k else f"{r[k]['mean']:.1f} ± {r[k]['sd']:.1f}" for k, _ in keys]
        lines.append(f"| {name} | {n} | " + " | ".join(cells) + f" | {r['timeouts']} |")
    (out / "l_path_replay.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
