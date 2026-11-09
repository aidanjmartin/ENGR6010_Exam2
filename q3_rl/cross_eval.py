"""Q3: cross-evaluation. Every PPO policy (3 seeds per training world) and the Phase 1 go-to-goal baseline run on
the same 100 seeded scenarios (seeds 10000 to 10099, as the Phase 1 evaluation) in each of the three worlds:
physics model, learned ensemble, MuJoCo dog. Deterministic (mean) actions. A seed's success rate is over its 100
scenarios; the table gives the mean over the 3 seeds ± the standard error over seeds.

    python q3_rl/cross_eval.py
"""
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "q3_rl"))

WORLDS = ["physics", "learned", "mujoco"]
SEEDS = range(10000, 10100)


def run(job):
    import torch
    torch.set_num_threads(1)
    import exam2  # noqa: F401
    from go2x.sim.baselines import GoToGoalPolicy
    from go2x.sim.nav_env import Go2NavEnv
    from ppo import ActorCritic, make_dynamics
    ctrl, path, world = job
    env = Go2NavEnv(dynamics=make_dynamics(world))
    if path:
        ac = ActorCritic(env.observation_space.shape[0], 3, 256)
        ac.load_state_dict(torch.load(path, map_location="cpu"))
        act = lambda o: np.clip(ac.actor(torch.as_tensor(o[None])).detach().numpy()[0], -1, 1)  # noqa: E731
    else:
        g = GoToGoalPolicy()
        act = lambda o: g(o, env)  # noqa: E731
    out = []
    for s in SEEDS:
        o, _ = env.reset(seed=s)
        while True:
            o, r, term, trunc, info = env.step(act(o))
            fell = bool(getattr(env.dynamics, "fell", False))
            if term or trunc or fell:
                break
        out.append({"seed": s, "success": bool(info["is_success"]) and not fell, "collision": bool(info["collision"]),
                    "fell": fell, "timeout": bool(trunc), "time_s": float(info["time_s"])})
    return ctrl, path, world, out


def main():
    jobs = []
    for tw in WORLDS:
        for p in sorted((ROOT / "q3_rl/runs").glob(f"{tw}_s*/policy.pt")):
            for w in WORLDS:
                jobs.append((f"PPO trained in {tw}", str(p), w))
    for w in WORLDS:
        jobs.append(("go-to-goal baseline (Phase 1)", "", w))
    with ProcessPoolExecutor(22) as ex:
        results = list(ex.map(run, jobs))
    per = {}
    for ctrl, path, world, out in results:
        per.setdefault(ctrl, {}).setdefault(world, []).append(
            {"policy": str(Path(path).relative_to(ROOT)) if path else "baseline", "episodes": out,
             "success": float(np.mean([e["success"] for e in out])), "collision": float(np.mean([e["collision"] for e in out])),
             "fell": float(np.mean([e["fell"] for e in out])), "timeout": float(np.mean([e["timeout"] for e in out])),
             "time_s": float(np.mean([e["time_s"] for e in out if e["success"]])) if any(e["success"] for e in out) else None})
    summ = {}
    for k in ("success", "collision", "fell", "timeout"):
        summ[k] = {}
        for ctrl, d in per.items():
            summ[k][ctrl] = {}
            for w, runs in d.items():
                v = np.array([r[k] for r in runs])
                se = float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else float(math.sqrt(v[0] * (1 - v[0]) / 100))
                summ[k][ctrl][w] = {"mean": float(v.mean()), "se": se, "seeds": len(v), "scenarios": 100}
    res = {"eval_worlds": WORLDS, "scenario_seeds": [SEEDS[0], SEEDS[-1]], **summ, "runs": per}
    out = ROOT / "q3_rl/results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "cross_eval.json").write_text(json.dumps(res, indent=1))
    lines = ["# Q3 cross-evaluation", "",
             "100 seeded scenarios (seeds 10000 to 10099). PPO rows: mean over 3 training seeds ± standard error over "
             "seeds. Baseline row: one deterministic controller, binomial standard error over the 100 scenarios.", ""]
    for k in ("success", "collision", "fell", "timeout"):
        lines += [f"**{k} rate**", "", "| controller | " + " | ".join(f"in {w}" for w in WORLDS) + " |",
                  "| :--- | " + " | ".join("---:" for _ in WORLDS) + " |"]
        for ctrl in summ[k]:
            lines.append(f"| {ctrl} | " + " | ".join(f"{summ[k][ctrl][w]['mean']:.3f} ± {summ[k][ctrl][w]['se']:.3f}" for w in WORLDS) + " |")
        lines.append("")
    (out / "cross_eval.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
