"""Simulated Go2 dataset from the MuJoCo dog (exam2/mjdog.py), in the fields of unitree_go2_create_dataset
(foot_force, motor_q, motor_dq, motor_ddq, motor_tau_est) plus the commanded body velocity, the body velocity and the
planar pose, at 20 Hz.

Commands: piecewise constant holds of 0.5 to 3 s, and "policy-like" stretches that change every 1 to 4 steps,
within vx [-0.5, 0.8] m/s, vy [-0.4, 0.4] m/s, wz [-1.0, 1.0] rad/s (covers the navigation limits and the 0.8 m/s
used on the real dog). Each axis is zero with probability 0.3; 15 % of holds are full stops.
Per episode: foot friction U(0.3, 1.0), payload U(0, 3) kg on the base. An episode ends early if the dog falls.

    python q2_data/collect_sim.py --split train --episodes 200 --seed0 0 [--walker isaac]
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

LIM_LO = np.array([-0.5, -0.4, -1.0])
LIM_HI = np.array([0.8, 0.4, 1.0])
FIELDS = ["cmd", "vel", "pose", "z", "rpy", "gyro", "foot_force", "motor_q", "motor_dq", "motor_ddq", "motor_tau_est"]


def command_sequence(rng, steps, dt=0.05):
    out = np.zeros((steps, 3))
    t = 0
    while t < steps:
        if rng.random() < 0.3:                                   # policy-like stretch
            end = min(steps, t + int(rng.uniform(1, 3) / dt))
            while t < end:
                c = rng.uniform(LIM_LO, LIM_HI) * (rng.random(3) > 0.3)
                n = int(rng.integers(1, 5))
                out[t:t + n] = c
                t += n
        else:                                                    # hold
            n = int(rng.uniform(0.5, 3.0) / dt)
            c = np.zeros(3) if rng.random() < 0.15 else rng.uniform(LIM_LO, LIM_HI) * (rng.random(3) > 0.3)
            out[t:t + n] = c
            t += n
    return out[:steps]


def run(job):
    from exam2 import ISAAC_WALKER, WALKER
    from exam2.mjdog import MujocoDog
    seeds, walker, seconds = job
    rows = []
    for s in seeds:
        rng = np.random.default_rng(s)
        friction, payload = float(rng.uniform(0.3, 1.0)), float(rng.uniform(0.0, 3.0))
        dog = MujocoDog(policy=str(ISAAC_WALKER if walker == "isaac" else WALKER), friction=friction, payload=payload)
        steps = int(round(seconds / 0.05))
        cmds = command_sequence(rng, steps)
        rec = {f: [] for f in FIELDS}
        st = dog.reset((0.0, 0.0, float(rng.uniform(-np.pi, np.pi))))
        fell = False
        for t in range(steps):
            for f in FIELDS[1:]:
                rec[f].append(st[f] if f != "z" else [st["z"]])
            rec["cmd"].append(cmds[t])
            st = dog.step(cmds[t])
            if dog.fallen():
                fell = True
                break
        n = len(rec["cmd"])
        rows.append({"seed": s, "friction": friction, "payload": payload, "fell": fell, "n": n,
                     **{f: np.asarray(rec[f], np.float32) for f in FIELDS}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--episodes", type=int, required=True)
    ap.add_argument("--seed0", type=int, required=True)
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--walker", default="mjx_s0", choices=["mjx_s0", "isaac"])
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "q2_data/dataset/sim"))
    args = ap.parse_args()
    seeds = list(range(args.seed0, args.seed0 + args.episodes))
    chunks = [seeds[i::args.workers] for i in range(args.workers)]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        rows = sorted([r for part in ex.map(run, [(c, args.walker, args.seconds) for c in chunks if c]) for r in part],
                      key=lambda r: r["seed"])
    T = int(round(args.seconds / 0.05))
    out = {"dt": np.float32(0.05), "seed": np.array([r["seed"] for r in rows]),
           "friction": np.array([r["friction"] for r in rows], np.float32),
           "payload_kg": np.array([r["payload"] for r in rows], np.float32),
           "fell": np.array([r["fell"] for r in rows]), "length": np.array([r["n"] for r in rows])}
    for f in FIELDS:
        a = np.zeros((len(rows), T) + rows[0][f].shape[1:], np.float32)
        for i, r in enumerate(rows):
            a[i, :r["n"]] = r[f]
        out[f] = a
    Path(args.out).mkdir(parents=True, exist_ok=True)
    path = Path(args.out) / f"{args.split}.npz"
    np.savez_compressed(path, **out)
    meta = {"split": args.split, "episodes": len(rows), "seconds": args.seconds, "walker": args.walker,
            "seeds": [seeds[0], seeds[-1]], "falls": int(out["fell"].sum()), "wall_s": round(time.time() - t0, 1),
            "size_mb": round(path.stat().st_size / 1e6, 1)}
    (Path(args.out) / f"{args.split}.json").write_text(json.dumps(meta, indent=1))
    print(meta)


if __name__ == "__main__":
    main()
