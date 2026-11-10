"""Q1 and Q2 on the real Go2 X: the 19 navigation runs with policy steps of 2026-11-06 (q2_data/dataset/real_runs_20261106/),
with the overhead camera as ground truth for the robot's position.

1. Camera against odometry (Q2): the robot's odometry, anchored at the first camera pose of each run, against the
   camera pose at every policy step where the camera saw the robot tag.
2. Command replay with path error (Q2): the commands the robot received (commands.jsonl `sent`, 20 Hz) replayed open
   loop through each dynamics model from the first camera pose, and the model path compared with the camera path.
3. The learned ensemble on real motion (Q1 C): body-velocity prediction 0.1 s and 0.5 s ahead on the real runs and on
   the Phase 1 motion tests of 2026-10-17 (q2_data/dataset/real_motion_20261017/), the same test as
   q1_vision_ml/train_ensemble.py on simulated episodes, plus a lag-delay model fitted to the real dog.
4. Digital-twin figure: every run's camera path, odometry path and MuJoCo-dog replay on the floor plan.

    GO2X_UNITREE_MUJOCO=<unitree_mujoco> python q2_data/real_runs.py
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "q1_vision_ml"))
from exam2 import DT  # noqa: E402
from exam2.dyn import LearnedModel, MujocoModel, PhysicsModel  # noqa: E402
from exam2.models import H, Ensemble, LagDelay, features  # noqa: E402
from exam2.realruns import COLORS, load_motion, load_runs, odom_to_square, path_length, wrap  # noqa: E402
from train_ensemble import HORIZONS, fit_lag_delay, one_step_pairs, windows  # noqa: E402

OUT = ROOT / "q2_data/results"
FIG = ROOT / "figures"
ENS = ROOT / "q1_vision_ml/results/ensemble.npz"
FIT_SIM = json.loads((ROOT / "q1_vision_ml/results/ensemble_eval.json").read_text())["lag_delay_fitted"]
AXES = ("vx", "vy", "wz")


def mean_se(a):
    a = np.asarray(a, float)
    return {"mean": float(a.mean()), "se": float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0, "n": int(len(a))}


def fmt(d, nd=3):
    return f"{d['mean']:.{nd}f} ± {d['se']:.{nd}f}"


# ---------------------------------------------------------------- 1. camera against odometry

def drift(runs):
    per_run, pooled = [], []
    for r in runs:
        cam = r["camera"]
        c = r["pose"][cam]
        o = odom_to_square(c[0], r["odom"][cam][0], r["odom"][cam])
        err = np.hypot(*(o[:, :2] - c[:, :2]).T)
        yaw_err = np.degrees(np.abs(wrap(o[:, 2] - c[:, 2])))
        dist = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(c[:, :2], axis=0).T))])
        pooled += list(zip(dist, err, yaw_err))
        L = dist[-1]
        per_run.append({"run": r["name"], "camera_steps": int(cam.sum()), "camera_path_m": round(float(L), 3),
                        "odometry_path_m": round(path_length(o), 3), "end_error_m": round(float(err[-1]), 3),
                        "max_error_m": round(float(err.max()), 3), "end_heading_error_deg": round(float(yaw_err[-1]), 1),
                        "error_per_m": round(float(err[-1] / L), 3) if L >= 0.5 else None})
    pooled = np.array(pooled)
    bins = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.5)]
    by_dist = []
    for a, b in bins:
        m = (pooled[:, 0] >= a) & (pooled[:, 0] < b)
        by_dist.append({"from_m": a, "to_m": b, "steps": int(m.sum()), "median_error_m": float(np.median(pooled[m, 1])),
                        "p90_error_m": float(np.percentile(pooled[m, 1], 90)), "median_heading_deg": float(np.median(pooled[m, 2]))})
    moving = [p for p in per_run if p["error_per_m"] is not None]
    # least-squares slope of error against distance through the origin, pooled over steps
    slope = float((pooled[:, 0] * pooled[:, 1]).sum() / (pooled[:, 0] ** 2).sum())
    summary = {"runs": len(per_run), "steps": int(len(pooled)),
               "end_error_m": mean_se([p["end_error_m"] for p in per_run]),
               "end_heading_error_deg": mean_se([p["end_heading_error_deg"] for p in per_run]),
               "error_per_m_runs_over_0.5m": mean_se([p["error_per_m"] for p in moving]),
               "path_ratio_odometry_over_camera": mean_se([p["odometry_path_m"] / p["camera_path_m"] for p in moving]),
               "slope_m_per_m": slope}
    return {"summary": summary, "by_distance": by_dist, "per_run": per_run}, pooled


# ---------------------------------------------------------------- 2. command replay

class EnsembleMean(LearnedModel):
    """The Q1 ensemble with the members' mean and no sampled noise (the mean prediction, as in Q1's rollouts)."""

    def step(self, cmd, dt):
        full = np.concatenate([self.ch, np.asarray(cmd, float).reshape(1, 1, 3)], 1)
        dv = self.ens.predict(features(self.vh, full))[0][0]
        v_new = self.vel + dv
        from go2x.sim.dynamics import integrate_pose
        for i in range(10):
            self.pose = integrate_pose(self.pose, self.vel + (v_new - self.vel) * (i + 0.5) / 10, dt / 10)
        self.vel = v_new
        self.vh = np.concatenate([self.vh[:, 1:], v_new.reshape(1, 1, 3)], 1)
        self.ch = full[:, 1:]


def replay_one(model, r, rng):
    cam = r["camera"]
    t0 = r["t"][cam][0]
    start = r["pose"][cam][0]
    sel = (r["cmd_t"] >= t0) & (r["cmd_t"] <= r["t"][-1] + 0.05)
    cmd_t, cmd = r["cmd_t"][sel], r["cmd"][sel]
    model.reset(rng, start.copy())
    ts, poses = [t0], [start.copy()]
    for t, c in zip(cmd_t, cmd):
        model.step(c, DT)
        ts.append(t + DT)
        poses.append(np.array(model.pose, float))
    ts, poses = np.array(ts), np.array(poses)
    tq = r["t"][cam]
    xy = np.stack([np.interp(tq, ts, poses[:, i]) for i in range(2)], 1)
    yaw = np.interp(tq, ts, np.unwrap(poses[:, 2]))
    return np.concatenate([xy, yaw[:, None]], 1)


def replay(runs, mujoco_ok, real_fit):
    models = {
        "physics, Phase 1 (unfitted)": lambda: PhysicsModel(fixed=([0.25] * 3, [0.075] * 3)),
        "physics, fitted to the MuJoCo dog": lambda: PhysicsModel(fixed=(FIT_SIM["tau_s"], FIT_SIM["delay_s"])),
        "physics, fitted to the real dog (in-sample)": lambda: PhysicsModel(fixed=(real_fit["tau_s"], real_fit["delay_s"])),
        "learned ensemble (mean)": lambda: EnsembleMean(ENS, sample_noise=False),
    }
    if mujoco_ok:
        models["MuJoCo dog, MuJoCo walker"] = lambda: MujocoModel()
    paths = {r["name"]: {} for r in runs}
    rows = {name: [] for name in ["odometry (the dog's own estimate)"] + list(models)}
    for r in runs:
        cam = r["camera"]
        c = r["pose"][cam]
        L = path_length(c)
        o = odom_to_square(c[0], r["odom"][cam][0], r["odom"][cam])
        paths[r["name"]]["odometry (the dog's own estimate)"] = o
        for name, make in models.items():
            m = make()
            paths[r["name"]][name] = replay_one(m, r, np.random.default_rng(0))
        for name, p in paths[r["name"]].items():
            e = np.hypot(*(p[:, :2] - c[:, :2]).T)
            rows[name].append({"run": r["name"], "mean_error_m": float(e.mean()), "end_error_m": float(e[-1]),
                               "end_heading_error_deg": float(np.degrees(abs(wrap(p[-1, 2] - c[-1, 2])))),
                               "path_ratio": float(path_length(p) / L) if L >= 0.5 else None})
        print(f"[replay] {r['name']}: " + ", ".join(f"{k.split(',')[0][:8]} {rows[k][-1]['end_error_m']:.2f}" for k in rows), flush=True)
    summary = {}
    for name, rr in rows.items():
        summary[name] = {"mean_error_m": mean_se([x["mean_error_m"] for x in rr]),
                         "end_error_m": mean_se([x["end_error_m"] for x in rr]),
                         "end_heading_error_deg": mean_se([x["end_heading_error_deg"] for x in rr]),
                         "path_ratio": mean_se([x["path_ratio"] for x in rr if x["path_ratio"] is not None])}
    return {"summary": summary, "per_run": rows}, paths


# ---------------------------------------------------------------- 3. ensemble on real motion

def real_arrays(runs):
    """Per run, the 20 Hz command grid over [t_from, t_to]: body velocity (state interpolated at each command time) and
    the command sent then (held for the next 0.05 s). Navigation runs: 1 s before the first policy step to 0.5 s after
    the last. Motion tests: the whole command stream."""
    seqs = []
    for r in runs:
        t_from = r.get("t_from", r["t"][0] - 1.0 if "t" in r else None)
        t_to = r.get("t_to", r["t"][-1] + 0.5 if "t" in r else None)
        sel = (r["cmd_t"] >= t_from) & (r["cmd_t"] <= t_to)
        tg = r["cmd_t"][sel]
        vel = np.stack([np.interp(tg, r["st_t"], r["vel"][:, i]) for i in range(3)], 1)
        seqs.append((vel, r["cmd"][sel]))
    T = max(len(v) for v, _ in seqs)
    V, C = np.zeros((len(seqs), T, 3)), np.zeros((len(seqs), T, 3))
    for i, (v, c) in enumerate(seqs):
        V[i, :len(v)], C[i, :len(c)] = v, c
    return V, C, np.array([len(v) for v, _ in seqs])


def predict_eval(vel, cmd, length, real_models):
    """RMSE at 0.1 s and 0.5 s per run, mean ± SE over runs, as in q1_vision_ml/train_ensemble.py. real_models(e) gives
    the lag-delay model fitted to the real dog to use for run e."""
    phase1, fitted, ens = LagDelay([0.25] * 3, [0.075] * 3), LagDelay(FIT_SIM["tau_s"], FIT_SIM["delay_s"]), Ensemble(ENS)
    res = {}
    for hname, K in HORIZONS.items():
        e, vh, ch, cp, cf, truth = windows(vel, cmd, length, K, stride=1)
        real_pred = np.zeros_like(truth)
        for k in np.unique(e):
            m = e == k
            real_pred[m] = real_models(k).rollout(vh[m, -1], cp[m], cf[m])
        preds = {"constant velocity": np.repeat(vh[:, -1:], K, 1),
                 "lag-delay, Phase 1 (unfitted)": phase1.rollout(vh[:, -1], cp, cf),
                 "lag-delay, fitted to the MuJoCo dog": fitted.rollout(vh[:, -1], cp, cf),
                 "lag-delay, fitted to the real dog": real_pred,
                 "ensemble (5 networks)": ens.rollout(vh, ch, cf)}
        res[hname] = {}
        for name, p in preds.items():
            se = (p[:, -1] - truth[:, -1]) ** 2
            rmse = np.sqrt(np.array([se[e == k].mean(0) for k in np.unique(e)]))
            res[hname][name] = {ax: mean_se(rmse[:, i]) for i, ax in enumerate(AXES)}
        res[hname]["windows"] = int(len(e))
    X, Y = one_step_pairs(vel, cmd, length)
    mu, var = ens.predict(X)
    res["one-step 2-sigma coverage"] = {ax: float(v) for ax, v in zip(AXES, (np.abs(Y - mu) <= 2 * np.sqrt(var)).mean(0))}
    res["one-step pairs"] = int(len(X))
    res["grid_steps"] = int(length.sum())
    res["runs"] = int(len(length))
    return res


def ensemble_real(runs, motion):
    """Navigation runs (2026-11-06): the real-dog lag-delay model is fitted on the other 18 runs (leave one run out).
    Motion tests (2026-10-17): it is fitted on all 19 navigation runs, a different day and different commands."""
    vel, cmd, length = real_arrays(runs)
    E = len(runs)
    tau_all, delay_all = fit_lag_delay(vel, cmd, length)
    loo = [LagDelay(*fit_lag_delay(np.delete(vel, e, 0), np.delete(cmd, e, 0), np.delete(length, e))) for e in range(E)]
    res = {"navigation_20261106": predict_eval(vel, cmd, length, lambda k: loo[k])}
    mv, mc, ml = real_arrays(motion)
    all_nav = LagDelay(tau_all, delay_all)
    res["motion_20261017"] = predict_eval(mv, mc, ml, lambda k: all_nav)
    res["motion_runs"] = [m["name"] for m in motion]
    res["lag_delay_fitted_real_all_runs"] = {"tau_s": tau_all.tolist(), "delay_s": delay_all.tolist()}
    res["lag_delay_fitted_real_leave_one_out"] = {"tau_s": [l.tau.tolist() for l in loo], "delay_s": [l.delay.tolist() for l in loo]}
    return res


# ---------------------------------------------------------------- figures

def figures(runs, pooled, rep, paths, ens_res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    shades = ["#1a1a1a", "#5c5c5c", "#9a9a9a", "#d0d0d0", "#ffffff", "#efefef"]
    hatch = ["", "//", "..", "xx", "\\\\", "--"]

    # drift against distance
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].scatter(pooled[:, 0], pooled[:, 1], s=6, color="#5c5c5c")
    xs = np.linspace(0, pooled[:, 0].max(), 10)
    slope = rep["drift"]["summary"]["slope_m_per_m"]
    ax[0].plot(xs, slope * xs, color="#1a1a1a", lw=1.2, label=f"fit through origin: {slope:.2f} m per m")
    ax[0].set_xlabel("distance walked so far, camera (m)")
    ax[0].set_ylabel("odometry position error (m)")
    ax[0].legend(frameon=False, loc="upper left")
    ax[0].set_title("Odometry against the overhead camera", fontsize=10)
    ax[1].scatter(pooled[:, 0], pooled[:, 2], s=6, color="#5c5c5c")
    ax[1].set_xlabel("distance walked so far, camera (m)")
    ax[1].set_ylabel("odometry heading error (deg)")
    ax[1].set_title(f"{rep['drift']['summary']['steps']} camera steps, 19 runs", fontsize=10)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "q2_camera_vs_odometry.png", dpi=150)
    plt.close(fig)

    # replay errors
    s = rep["replay"]["summary"]
    names = list(s)
    fig, ax = plt.subplots(figsize=(10, 3.8))
    groups = ["mean position error (m)", "end position error (m)"]
    w = 0.8 / len(names)
    for i, n in enumerate(names):
        vals = [s[n]["mean_error_m"]["mean"], s[n]["end_error_m"]["mean"]]
        errs = [s[n]["mean_error_m"]["se"], s[n]["end_error_m"]["se"]]
        ax.bar(np.arange(2) + (i - (len(names) - 1) / 2) * w, vals, w * 0.92, yerr=errs, capsize=2, color=shades[i % 6],
               edgecolor="#1a1a1a", linewidth=0.6, hatch=hatch[i % 6], label=n)
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(groups)
    ax.set_ylabel("m, mean ± standard error over 19 runs")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_title("Recorded commands replayed open loop, compared with the camera path", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "q2_replay_path_error.png", dpi=150)
    plt.close(fig)

    # ensemble on real motion, 0.5 s ahead
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, key, title in ((axs[0], "navigation_20261106", "19 navigation runs, 2026-11-06"),
                           (axs[1], "motion_20261017", f"{ens_res['motion_20261017']['runs']} motion tests, 2026-10-17")):
        r5 = ens_res[key]["0.5 s"]
        mnames = [k for k in r5 if k != "windows"]
        w = 0.8 / len(mnames)
        for i, n in enumerate(mnames):
            ax.bar(np.arange(3) + (i - (len(mnames) - 1) / 2) * w, [r5[n][a]["mean"] for a in AXES], w * 0.92,
                   yerr=[r5[n][a]["se"] for a in AXES], capsize=2, color=shades[i % 6], edgecolor="#1a1a1a", linewidth=0.6,
                   hatch=hatch[i % 6], label=n)
        ax.set_xticks(np.arange(3))
        ax.set_xticklabels(["vx (m/s)", "vy (m/s)", "wz (rad/s)"])
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=10)
    axs[0].set_ylabel("RMSE 0.5 s ahead, mean ± SE over runs")
    axs[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Velocity prediction on the real Go2 X", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "q1_ensemble_real.png", dpi=150)
    plt.close(fig)

    # digital-twin paths on the floor plan, one panel per run
    side = 3.048
    n = len(runs)
    cols = 5
    rows_ = math.ceil(n / cols)
    fig, axs = plt.subplots(rows_, cols, figsize=(3.0 * cols, 3.2 * rows_))
    styles = {"camera": dict(color="#1a1a1a", lw=2.0), "odometry (the dog's own estimate)": dict(color="#1a1a1a", lw=1.0, ls="--"),
              "MuJoCo dog, MuJoCo walker": dict(color="#7a7a7a", lw=1.2, ls="-."),
              "physics, Phase 1 (unfitted)": dict(color="#9a9a9a", lw=1.0, ls=":")}
    for k, a in enumerate(axs.flat):
        a.set_aspect("equal")
        a.set_xticks([])
        a.set_yticks([])
        for sp in a.spines.values():
            sp.set_visible(False)
        if k >= n:
            a.axis("off")
            continue
        r = runs[k]
        a.add_patch(Rectangle((-side / 2, -side / 2), side, side, fill=False, lw=0.8, ec="#5c5c5c"))
        sc = r["scenario"]
        tgt = sc.get("target", [None, None])
        for j, ch in enumerate(sc.get("chairs", [])):
            x, y, yaw = ch[0], ch[1], ch[2]
            c, s_ = math.cos(yaw), math.sin(yaw)
            pts = [(x + c * dx - s_ * dy, y + s_ * dx + c * dy) for dx, dy in ((-.25, -.25), (.25, -.25), (.25, .25), (-.25, .25))]
            is_t = tgt[0] == "chair" and tgt[1] == j
            a.add_patch(Polygon(pts, closed=True, fc="#bdbdbd" if is_t else "#ececec", ec="#1a1a1a", lw=0.6, hatch="//" if is_t else ""))
            a.text(x, y, COLORS[int(ch[3])][0].upper(), ha="center", va="center", fontsize=7)
        for j, tb in enumerate(sc.get("table", [])):
            x, y, yaw = tb[0], tb[1], tb[2]
            c, s_ = math.cos(yaw), math.sin(yaw)
            pts = [(x + c * dx - s_ * dy, y + s_ * dx + c * dy) for dx, dy in ((-.6, -.375), (.6, -.375), (.6, .375), (-.6, .375))]
            is_t = tgt[0] == "table"
            a.add_patch(Polygon(pts, closed=True, fc="#bdbdbd" if is_t else "#ececec", ec="#1a1a1a", lw=0.6, hatch="//" if is_t else ""))
        cam = r["pose"][r["camera"]]
        a.plot(cam[:, 0], cam[:, 1], label="camera", **styles["camera"])
        for name in ("odometry (the dog's own estimate)", "MuJoCo dog, MuJoCo walker", "physics, Phase 1 (unfitted)"):
            if name in paths[r["name"]]:
                p = paths[r["name"]][name]
                a.plot(p[:, 0], p[:, 1], label=name, **styles[name])
        a.plot(*cam[0, :2], "o", color="#1a1a1a", ms=4)
        a.set_xlim(-side / 2 - 0.1, side / 2 + 0.1)
        a.set_ylim(-side / 2 - 0.1, side / 2 + 0.1)
        hh = r["name"][9:11] + ":" + r["name"][11:13] + ":" + r["name"][13:15]
        a.set_title(f"{hh} UTC, {r['target']}\n{'success' if r['success'] else r['outcome'].split(':')[0]}", fontsize=8)
    h, l = axs.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower right", frameon=False, fontsize=9, bbox_to_anchor=(0.98, 0.03))
    fig.suptitle("Real runs of 2026-11-06 on the 3.048 m square: camera path, odometry, and the recorded commands replayed in simulation"
                 "\n(target hatched; chair letters are tag colors; dot = start)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "q2_real_runs_paths.png", dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------- report

def write_report(rep):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "real_runs.json").write_text(json.dumps(rep, indent=1))
    d = rep["drift"]["summary"]
    L = ["# The real Go2 X runs of 2026-11-06 against the overhead camera", "",
         "19 runs with policy steps (q2_data/dataset/real_runs_20261106/). Ground truth: the overhead camera's robot pose at "
         "every policy step where it saw the robot's tag. Mean ± standard error over runs unless stated.", "",
         "## 1. Camera against odometry", "",
         f"Odometry anchored at each run's first camera pose ({d['steps']} camera steps):", "",
         "| quantity | value |", "| :--- | ---: |",
         f"| position error at the run's end (m) | {fmt(d['end_error_m'])} |",
         f"| heading error at the run's end (deg) | {fmt(d['end_heading_error_deg'], 1)} |",
         f"| end error per m walked, runs over 0.5 m (n = {d['error_per_m_runs_over_0.5m']['n']}) | {fmt(d['error_per_m_runs_over_0.5m'])} |",
         f"| odometry path / camera path, same runs | {fmt(d['path_ratio_odometry_over_camera'])} |",
         f"| least-squares slope of error against distance, all steps (m per m) | {d['slope_m_per_m']:.3f} |", "",
         "| distance walked (m) | steps | median error (m) | 90th percentile (m) | median heading error (deg) |",
         "| :--- | ---: | ---: | ---: | ---: |"]
    for b in rep["drift"]["by_distance"]:
        L.append(f"| {b['from_m']} to {b['to_m']} | {b['steps']} | {b['median_error_m']:.3f} | {b['p90_error_m']:.3f} | {b['median_heading_deg']:.1f} |")
    L += ["", "Per run:", "", "| run (UTC) | camera steps | camera path (m) | odometry path (m) | end error (m) | max error (m) | end heading error (deg) |",
          "| :--- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for p in rep["drift"]["per_run"]:
        L.append(f"| {p['run']} | {p['camera_steps']} | {p['camera_path_m']:.2f} | {p['odometry_path_m']:.2f} | {p['end_error_m']:.3f} | "
                 f"{p['max_error_m']:.3f} | {p['end_heading_error_deg']:.1f} |")
    s = rep["replay"]["summary"]
    L += ["", "## 2. Recorded commands replayed through the dynamics models", "",
          "Each run's sent commands (20 Hz) replayed open loop from its first camera pose, at rest; the model path is "
          "compared with the camera path at the camera steps. The odometry row is the dog's own estimate, for reference.", "",
          "| model | mean position error (m) | end position error (m) | end heading error (deg) | path length / camera path (runs over 0.5 m) |",
          "| :--- | ---: | ---: | ---: | ---: |"]
    for n, v in s.items():
        L.append(f"| {n} | {fmt(v['mean_error_m'])} | {fmt(v['end_error_m'])} | {fmt(v['end_heading_error_deg'], 1)} | {fmt(v['path_ratio'], 2)} |")
    e = rep["ensemble_real"]
    L += ["", "## 3. Velocity prediction on the real dog", "",
          "Body velocity on the 20 Hz command grid, predicted with the commands over the horizon known. RMSE at the horizon "
          "per run, then mean ± standard error over runs. vx and vy in m/s, wz in rad/s. The lag-delay model fitted to the "
          "real dog is fitted on the other 18 navigation runs for the navigation runs, and on all 19 navigation runs for "
          "the motion tests.", ""]
    for key, title in (("navigation_20261106", "navigation runs, 2026-11-06"), ("motion_20261017", "Phase 1 motion tests, 2026-10-17 (" + ", ".join(e["motion_runs"]) + ")")):
        ek = e[key]
        L += [f"### {title}", "", f"{ek['runs']} runs, {ek['grid_steps']} steps.", ""]
        for hname in HORIZONS:
            L += [f"**{hname} ahead** ({ek[hname]['windows']} windows)", "", "| model | vx | vy | wz |", "| :--- | ---: | ---: | ---: |"]
            for n, v in ek[hname].items():
                if n != "windows":
                    L.append(f"| {n} | " + " | ".join(fmt(v[a]) for a in AXES) + " |")
            L.append("")
        cov = ek["one-step 2-sigma coverage"]
        L += [f"One-step 2-sigma coverage ({ek['one-step pairs']} pairs): " + ", ".join(f"{a} {v:.3f}" for a, v in cov.items())
              + " (0.954 if Gaussian and calibrated).", ""]
    fr = e["lag_delay_fitted_real_all_runs"]
    L += [f"Lag-delay fitted to all 19 navigation runs: tau {np.round(fr['tau_s'], 3).tolist()} s, delay {np.round(fr['delay_s'], 3).tolist()} s "
          f"(vx, vy, wz). Fitted to the MuJoCo dog (Q1): tau {FIT_SIM['tau_s']} s, delay {FIT_SIM['delay_s']} s.", ""]
    (OUT / "real_runs.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


def main():
    import os
    from exam2 import MODEL_DIR
    runs = load_runs()
    print(f"[real runs] {len(runs)} runs with policy steps, {sum(int(r['camera'].sum()) for r in runs)} camera steps", flush=True)
    mujoco_ok = os.path.exists(os.path.join(MODEL_DIR, "unitree_robots/go2/go2.xml"))
    rep = {"runs": [r["name"] for r in runs]}
    rep["drift"], pooled = drift(runs)
    rep["ensemble_real"] = ensemble_real(runs, load_motion())
    rep["replay"], paths = replay(runs, mujoco_ok, rep["ensemble_real"]["lag_delay_fitted_real_all_runs"])
    write_report(rep)
    figures(runs, pooled, rep, paths, rep["ensemble_real"])


if __name__ == "__main__":
    main()
