"""Q2: sensor statistics of the standing dog, real against simulated.

Real: Exam 1's 20 s WebRTC recording of the Go2 X standing still (q2_data/dataset/real_exam1): sport-mode state at
about 20 Hz (body height and the robot's own velocity estimate) and low state at about 1 Hz (IMU angles, foot force
sensors, battery). Simulated: the MuJoCo dog in the training split, in every stretch where the command was zero for
at least 1.5 s, using the last 1.0 s of each (after it has settled).

    python q2_data/sensor_stats.py
"""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REAL = ROOT / "q2_data/dataset/real_exam1"
OUT = ROOT / "q2_data/results"


def stat(x):
    x = np.asarray(x, float)
    return {"n": int(len(x)), "mean": float(x.mean()), "sd": float(x.std(ddof=1)),
            "se_mean": float(x.std(ddof=1) / math.sqrt(len(x)))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(REAL / "sportmodestate.csv")
    lo = pd.read_csv(REAL / "lowstate.csv")
    dt = np.diff(s["t_s"].values)
    slope = np.polyfit(s["t_s"], s["body_height_m"], 1)[0]
    real = {
        "sport_state": {"messages": int(len(s)), "duration_s": float(s["t_s"].iloc[-1] - s["t_s"].iloc[0]),
                        "period_s": {"median": float(np.median(dt)), "p95": float(np.percentile(dt, 95)), "max": float(dt.max())},
                        "body_height_m": stat(s["body_height_m"]), "body_height_drift_m_per_s": float(slope),
                        "vx": stat(s["vx"]), "vy": stat(s["vy"]), "vz": stat(s["vz"]),
                        "modes": sorted(set(map(int, s["mode"])))},
        "low_state": {"messages": int(len(lo)), "roll_rad": stat(lo["roll"]), "pitch_rad": stat(lo["pitch"]),
                      "yaw_rad": stat(lo["yaw"]),
                      "foot_force_raw": {f"foot_{i}": stat(lo[f"foot_force_{i}"]) for i in range(4)},
                      "battery_pct": [int(lo["battery_soc_pct"].min()), int(lo["battery_soc_pct"].max())]},
    }
    z = np.load(ROOT / "q2_data/dataset/sim/train.npz")
    cmd, vel, h, ff, rpy = z["cmd"], z["vel"], z["z"][..., 0], z["foot_force"], z["rpy"]
    vx, vy, hh, roll, pitch, f = [], [], [], [], [], []
    seg_sd = []
    for e in range(len(cmd)):
        still = np.all(cmd[e] == 0, axis=1)
        t = 0
        while t < len(still):
            if still[t]:
                u = t
                while u < len(still) and still[u]:
                    u += 1
                if (u - t) * 0.05 >= 1.5:
                    w = slice(u - 20, u)
                    vx += list(vel[e, w, 0]); vy += list(vel[e, w, 1]); hh += list(h[e, w])
                    roll += list(rpy[e, w, 0]); pitch += list(rpy[e, w, 1]); f += list(ff[e, w].sum(1))
                    seg_sd.append(vel[e, w, 0].std())
                t = u
            else:
                t += 1
    sim = {"stretches": len(seg_sd), "samples": len(vx), "vx": stat(vx), "vy": stat(vy), "body_height_m": stat(hh),
           "roll_rad": stat(roll), "pitch_rad": stat(pitch), "total_foot_force_N": stat(f)}
    res = {"real": real, "sim": sim}
    (OUT / "sensor_stats.json").write_text(json.dumps(res, indent=1))
    r = real["sport_state"]
    lines = ["# The standing dog: real sensor statistics against the MuJoCo dog", "",
             f"Real: Exam 1 WebRTC recording, {r['messages']} sport-state messages over {r['duration_s']:.1f} s "
             f"(median period {r['period_s']['median'] * 1000:.1f} ms, 95th percentile {r['period_s']['p95'] * 1000:.1f} ms, "
             f"longest {r['period_s']['max'] * 1000:.1f} ms) and {real['low_state']['messages']} low-state messages. "
             f"Simulated: {sim['stretches']} settled standing stretches, {sim['samples']} samples at 20 Hz.", "",
             "| quantity | real mean | real sd | simulated mean | simulated sd |", "| :--- | ---: | ---: | ---: | ---: |"]
    rows = [("body height (m)", r["body_height_m"], sim["body_height_m"]),
            ("velocity vx (m/s)", r["vx"], sim["vx"]), ("velocity vy (m/s)", r["vy"], sim["vy"]),
            ("roll (rad)", real["low_state"]["roll_rad"], sim["roll_rad"]),
            ("pitch (rad)", real["low_state"]["pitch_rad"], sim["pitch_rad"])]
    for name, a, b in rows:
        lines.append(f"| {name} | {a['mean']:+.4f} | {a['sd']:.2g} | {b['mean']:+.4f} | {b['sd']:.2g} |")
    lines.append(f"| velocity vz (m/s) | {r['vz']['mean']:+.4f} | {r['vz']['sd']:.2g} | not recorded | |")
    lines += ["", f"Real body height drift: {r['body_height_drift_m_per_s'] * 1000:+.3f} mm/s. Real foot force sensors "
              "(raw units, not newtons): " + ", ".join(f"{k} {v['mean']:.1f} ± sd {v['sd']:.1f}" for k, v in real['low_state']['foot_force_raw'].items())
              + f" (n = {real['low_state']['messages']}). Simulated total foot normal force {sim['total_foot_force_N']['mean']:.1f} N "
              f"(sd {sim['total_foot_force_N']['sd']:.1f})."]
    (OUT / "sensor_stats.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
