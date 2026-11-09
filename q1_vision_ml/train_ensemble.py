"""Q1 (C), supervised learning: a probabilistic ensemble that predicts the dog's next body velocity from its recent
velocities and velocity commands, trained on the simulated dataset (q2_data/dataset/sim). Compared at 0.1 s and
0.5 s ahead with constant velocity, the Phase 1 lag-delay model (its unfitted mean parameters) and a lag-delay
model fitted to the training split, on the held-out test split (same walker) and on a walker it never saw
(the Isaac Lab walker, split shift_isaac).

    python q1_vision_ml/train_ensemble.py [--eval-only]
"""
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exam2.models import DT, H, Ensemble, LagDelay, features  # noqa: E402

DATA = ROOT / "q2_data/dataset/sim"
OUT = ROOT / "q1_vision_ml/results"
N_MEMBERS, HIDDEN, EPOCHS = 5, 200, 40
HORIZONS = {"0.1 s": 2, "0.5 s": 10}
P = 8                                   # command history kept for the lag-delay delay buffer (0.4 s)


def load(split):
    z = np.load(DATA / f"{split}.npz")
    return z["vel"].astype(np.float64), z["cmd"].astype(np.float64), z["length"]


def one_step_pairs(vel, cmd, length):
    X, Y = [], []
    for e in range(len(vel)):
        for t in range(H - 1, length[e] - 1):
            X.append(features(vel[e, t - H + 1:t + 1][None], cmd[e, t - H + 1:t + 1][None])[0])
            Y.append(vel[e, t + 1] - vel[e, t])
    return np.array(X), np.array(Y)


class Member(nn.Module):
    def __init__(self, d_in):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
                                 nn.Linear(HIDDEN, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, 6))
        self.max_logvar = nn.Parameter(torch.full((3,), 0.5))
        self.min_logvar = nn.Parameter(torch.full((3,), -10.0))

    def forward(self, x):
        h = self.net(x)
        mu, lv = h[:, :3], h[:, 3:]
        lv = self.max_logvar - nn.functional.softplus(self.max_logvar - lv)
        lv = self.min_logvar + nn.functional.softplus(lv - self.min_logvar)
        return mu, lv


def train(X, Y, Xv, Yv, seed):
    """Gaussian negative log likelihood with learned log-variance bounds (PETS); bootstrap resample per member."""
    x_mean, x_std = X.mean(0), X.std(0) + 1e-6
    y_mean, y_std = Y.mean(0), Y.std(0) + 1e-6
    nx = lambda a: torch.tensor((a - x_mean) / x_std, dtype=torch.float32)  # noqa: E731
    ny = lambda a: torch.tensor((a - y_mean) / y_std, dtype=torch.float32)  # noqa: E731
    members, curves = [], []
    for m in range(N_MEMBERS):
        torch.manual_seed(seed + m)
        rng = np.random.default_rng(seed + m)
        idx = rng.integers(0, len(X), len(X))                      # bootstrap
        xb, yb = nx(X[idx]), ny(Y[idx])
        xv, yv = nx(Xv), ny(Yv)
        net = Member(X.shape[1])
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-5)
        curve = []
        for ep in range(EPOCHS):
            perm = torch.randperm(len(xb))
            for i in range(0, len(xb), 256):
                b = perm[i:i + 256]
                mu, lv = net(xb[b])
                loss = (((mu - yb[b]) ** 2) * torch.exp(-lv) + lv).mean() + 0.01 * (net.max_logvar.sum() - net.min_logvar.sum())
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                mu, _ = net(xv)
                curve.append(float(((mu - yv) ** 2).mean()))
        members.append(net)
        curves.append(curve)
        print(f"[ensemble] member {m}: validation MSE (normalized) {curve[0]:.4f} -> {curve[-1]:.4f}", flush=True)
    save = {"n_members": N_MEMBERS, "n_layers": 4, "x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    for m, net in enumerate(members):
        lin = [l for l in net.net if isinstance(l, nn.Linear)]
        for i, l in enumerate(lin):
            save[f"m{m}_W{i}"] = l.weight.detach().numpy().T.astype(np.float64)
            save[f"m{m}_b{i}"] = l.bias.detach().numpy().astype(np.float64)
    # shared bounds (members learn nearly identical ones); use the widest
    save["max_logvar"] = np.max([n.max_logvar.detach().numpy() for n in members], 0).astype(np.float64)
    save["min_logvar"] = np.min([n.min_logvar.detach().numpy() for n in members], 0).astype(np.float64)
    return save, curves


def windows(vel, cmd, length, K, stride=5):
    """Start times t with H history and P past commands, and K future steps."""
    rows = []
    for e in range(len(vel)):
        for t in range(max(H - 1, P), length[e] - K, stride):
            rows.append((e, t))
    e, t = np.array(rows).T
    vh = np.stack([vel[e, t - H + 1 + i] for i in range(H)], 1)
    ch = np.stack([cmd[e, t - H + 1 + i] for i in range(H - 1)], 1)       # commands before t
    cp = np.stack([cmd[e, t - P + i] for i in range(P)], 1)
    cf = np.stack([cmd[e, t + i] for i in range(K)], 1)
    truth = np.stack([vel[e, t + 1 + i] for i in range(K)], 1)
    return e, vh, ch, cp, cf, truth


def fit_lag_delay(vel, cmd, length):
    """Per-axis grid search of tau and delay minimizing the 0.5 s rollout squared error on the training split."""
    K = HORIZONS["0.5 s"]
    _, vh, _, cp, cf, truth = windows(vel, cmd, length, K, stride=10)
    taus = np.round(np.arange(0.02, 0.81, 0.02), 3)
    delays = np.round(np.arange(0.0, 0.301, 0.025), 3)
    best = np.full(3, np.inf), np.zeros(3), np.zeros(3)
    for tau in taus:
        for dl in delays:
            pred = LagDelay([tau] * 3, [dl] * 3).rollout(vh[:, -1], cp, cf)
            err = ((pred - truth) ** 2).mean((0, 1))
            better = err < best[0]
            best[0][better], best[1][better], best[2][better] = err[better], tau, dl
    return best[1], best[2]


def evaluate(models, split):
    vel, cmd, length = load(split)
    res = {}
    for hname, K in HORIZONS.items():
        e, vh, ch, cp, cf, truth = windows(vel, cmd, length, K)
        preds = {
            "constant velocity": np.repeat(vh[:, -1:], K, 1),
            "lag-delay, Phase 1 (unfitted)": models["phase1"].rollout(vh[:, -1], cp, cf),
            "lag-delay, fitted": models["fitted"].rollout(vh[:, -1], cp, cf),
            "ensemble (5 networks)": models["ensemble"].rollout(vh, ch, cf),
        }
        res[hname] = {}
        for name, p in preds.items():
            se_ep = (p[:, -1] - truth[:, -1]) ** 2                     # squared error at the horizon
            per_ep = np.array([se_ep[e == k].mean(0) for k in np.unique(e)])
            rmse = np.sqrt(per_ep)                                     # per-episode RMSE per axis
            res[hname][name] = {ax: {"mean": float(rmse[:, i].mean()), "se": float(rmse[:, i].std(ddof=1) / math.sqrt(len(rmse)))}
                                for i, ax in enumerate(["vx", "vy", "wz"])}
            res[hname][name]["episodes"] = int(len(rmse))
            res[hname][name]["windows"] = int(len(p))
    # one-step calibration: share of true changes inside the ensemble's 2-sigma interval
    X, Y = one_step_pairs(vel, cmd, length)
    mu, var = models["ensemble"].predict(X)
    inside = (np.abs(Y - mu) <= 2 * np.sqrt(var)).mean(0)
    res["one-step 2-sigma coverage"] = {ax: float(inside[i]) for i, ax in enumerate(["vx", "vy", "wz"])}
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    vel, cmd, length = load("train")
    X, Y = one_step_pairs(vel, cmd, length)
    Xv, Yv = one_step_pairs(*load("val"))
    print(f"[ensemble] {len(X)} training pairs, {len(Xv)} validation pairs", flush=True)
    if "--eval-only" in sys.argv and (OUT / "ensemble.npz").exists():
        curves = json.loads((OUT / "ensemble_eval.json").read_text())["validation_curves"] if (OUT / "ensemble_eval.json").exists() else []
    else:
        save, curves = train(X, Y, Xv, Yv, seed=0)
        np.savez(OUT / "ensemble.npz", **save)
    tau, delay = fit_lag_delay(vel, cmd, length)
    print(f"[lag-delay] fitted tau {tau} s, delay {delay} s", flush=True)
    models = {"phase1": LagDelay([0.25] * 3, [0.075] * 3), "fitted": LagDelay(tau, delay),
              "ensemble": Ensemble(OUT / "ensemble.npz")}
    report = {"train_pairs": int(len(X)), "val_pairs": int(len(Xv)), "members": N_MEMBERS, "hidden": [HIDDEN] * 3,
              "epochs": EPOCHS, "history_steps": H, "validation_curves": curves,
              "lag_delay_fitted": {"tau_s": tau.tolist(), "delay_s": delay.tolist()},
              "lag_delay_phase1": {"tau_s": [0.25] * 3, "delay_s": [0.075] * 3,
                                   "note": "means of config/sim.yaml tau_range [0.1, 0.4] and delay_range [0, 0.15]"},
              "test": evaluate(models, "test"), "shift_isaac": evaluate(models, "shift_isaac"),
              "wall_s": round(time.time() - t0, 1)}
    (OUT / "ensemble_eval.json").write_text(json.dumps(report, indent=1))
    lines = ["# Velocity prediction on held-out simulated episodes",
             "",
             "RMSE of the predicted body velocity at the horizon, per episode, then mean ± standard error over episodes "
             "(n = 30 episodes per split). vx and vy in m/s, wz in rad/s. Commands over the horizon are known to every model.",
             ""]
    for split, title in (("test", "test: same walker as training (mjx_s0)"), ("shift_isaac", "shift: Isaac Lab walker, never seen in training")):
        for hname in HORIZONS:
            lines += [f"**{title}, {hname} ahead**", "", "| model | vx | vy | wz |", "| :--- | ---: | ---: | ---: |"]
            for name, r in report[split][hname].items():
                lines.append(f"| {name} | " + " | ".join(f"{r[a]['mean']:.3f} ± {r[a]['se']:.3f}" for a in ("vx", "vy", "wz")) + " |")
            lines.append("")
        cov = report[split]["one-step 2-sigma coverage"]
        lines += [f"One-step 2-sigma coverage ({split}): " + ", ".join(f"{a} {v:.3f}" for a, v in cov.items()) + " (0.954 if Gaussian and calibrated).", ""]
    lines.append(f"Fitted lag-delay: tau {np.round(tau, 3).tolist()} s, delay {np.round(delay, 3).tolist()} s (vx, vy, wz).")
    (OUT / "ensemble_eval.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
