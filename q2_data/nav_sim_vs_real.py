"""Q2: the same navigation policy (Phase 2 attempt 4) in simulation and on the real Go2 X, by start distance.

Real: the 19 runs with policy steps on 2026-11-06 (results/robot/nav_camera_runs_webrtc.md), Unitree's walking
controller, overhead-camera pose. Simulation: the 200 shared evaluation scenarios (project
config/nav_eval_scenarios.json) in Isaac Lab (job 060) and in MuJoCo over both trained walkers (job 140).
Start distance is from the robot's front to the target's footprint in both (project go2x/atrium/scene.py
_target_distance; the real table reports the same quantity). Rates with binomial standard errors.

    python q2_data/nav_sim_vs_real.py
"""
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import exam2  # noqa: E402,F401
from go2x.atrium.scene import _target_distance, layout_from_dict  # noqa: E402

CUT = 1.9
SIMS = {"Isaac Lab (job 060)": "results/phase2/nav_eval_attempt4/nav_eval_policy.json",
        "MuJoCo, MuJoCo walker (job 140)": "project/results/phase2/mujoco_nav_eval/mjx_walker/nav_eval_isaac_attempt4.json",
        "MuJoCo, Isaac Lab walker (job 140)": "project/results/phase2/mujoco_nav_eval/isaac_walker/nav_eval_isaac_attempt4.json"}


def rate(k, n):
    p = k / n if n else float("nan")
    return {"k": k, "n": n, "rate": p, "se": math.sqrt(p * (1 - p) / n) if n else float("nan")}


def main():
    scen = json.loads((ROOT / "project/config/nav_eval_scenarios.json").read_text())["scenarios"]
    dist = [_target_distance(layout_from_dict(s)) for s in scen]
    res = {}
    rows = []
    for line in (ROOT / "results/robot/nav_camera_runs_webrtc.md").read_text().splitlines():
        m = re.match(r"\| (\d{8}T\d{6}Z) \| [^|]+ \| ([^|]+) \| (\d+) \| ([\d.]+) → ([\d.]+) \|", line)
        if m and int(m.group(3)) > 0:
            rows.append((m.group(2).strip().startswith("success"), float(m.group(4))))
    res["real Go2 X (Unitree controller), 2026-11-06"] = {
        "all": rate(sum(s for s, _ in rows), len(rows)),
        f"start < {CUT} m": rate(sum(s for s, d in rows if d < CUT), sum(1 for _, d in rows if d < CUT)),
        f"start >= {CUT} m": rate(sum(s for s, d in rows if d >= CUT), sum(1 for _, d in rows if d >= CUT)),
        "start distance range m": [min(d for _, d in rows), max(d for _, d in rows)]}
    for name, path in SIMS.items():
        eps = json.loads((ROOT / path).read_text())["episodes"]
        ok = {e["scenario"]: e["success"] for e in eps}
        near = [i for i in ok if dist[i] < CUT]
        far = [i for i in ok if dist[i] >= CUT]
        res[name] = {"all": rate(sum(ok.values()), len(ok)),
                     f"start < {CUT} m": rate(sum(ok[i] for i in near), len(near)),
                     f"start >= {CUT} m": rate(sum(ok[i] for i in far), len(far)),
                     f"start >= {CUT} m, within the real range": rate(
                         sum(ok[i] for i in far if dist[i] <= 2.4), sum(1 for i in far if dist[i] <= 2.4))}
    res["simulated start distance m"] = {"min": min(dist), "median": sorted(dist)[100], "max": max(dist)}
    out = ROOT / "q2_data/results"
    (out / "nav_sim_vs_real.json").write_text(json.dumps(res, indent=1))
    keys = ["all", f"start < {CUT} m", f"start >= {CUT} m"]
    lines = ["# Navigation policy (attempt 4): simulation against the real Go2 X", "",
             "Success rate ± binomial standard error (successes / runs). Start distance: robot front to target footprint.", "",
             "| world | " + " | ".join(keys) + " |", "| :--- | " + " | ".join("---:" for _ in keys) + " |"]
    for name, r in res.items():
        if "all" in r:
            lines.append(f"| {name} | " + " | ".join(f"{r[k]['rate']:.3f} ± {r[k]['se']:.3f} ({r[k]['k']}/{r[k]['n']})" for k in keys) + " |")
    d = res["simulated start distance m"]
    lines += ["", f"Simulated start distances: {d['min']:.2f} to {d['max']:.2f} m (median {d['median']:.2f}); real: "
              f"{res['real Go2 X (Unitree controller), 2026-11-06']['start distance range m'][0]:.2f} to "
              f"{res['real Go2 X (Unitree controller), 2026-11-06']['start distance range m'][1]:.2f} m."]
    (out / "nav_sim_vs_real.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
