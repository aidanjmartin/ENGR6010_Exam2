"""Figures for README.md, from the result files in this repository. Grayscale-safe: series differ by shade and
hatch, not by hue alone.

    python figures/make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
SHADES = ["#1a1a1a", "#5c5c5c", "#9a9a9a", "#d0d0d0", "#ffffff"]
HATCH = ["", "//", "..", "xx", "\\\\"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def bars(ax, groups, series, values, errors=None, ylabel=""):
    n = len(series)
    w = 0.8 / n
    x = np.arange(len(groups))
    for i, s in enumerate(series):
        ax.bar(x + (i - (n - 1) / 2) * w, values[i], w * 0.92, yerr=None if errors is None else errors[i], capsize=2,
               color=SHADES[i % 5], edgecolor="#1a1a1a", linewidth=0.6, hatch=HATCH[i % 5], label=s)
    ax.set_xticks(x)
    ax.set_xticklabels(groups)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)


def system_diagram():
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.6)
    ax.axis("off")
    boxes = {
        "lang": (0.1, 2.4, "plain-language target\n\"go to the red chair\""),
        "cam": (0.1, 0.5, "overhead camera\nArUco tags, OWLv2"),
        "img": (2.9, 1.45, "drawn top-down image\n(robot view, 64x64)"),
        "nav": (5.5, 1.45, "navigation policy\n(CNN, PPO), 10 Hz"),
        "walk": (8.1, 2.4, "walking policy (PPO), 50 Hz\nsimulation only"),
        "unitree": (8.1, 0.5, "Unitree walking controller\nreal robot"),
    }
    for k, (x, y, t) in boxes.items():
        ax.add_patch(FancyBboxPatch((x, y), 2.3, 0.85, boxstyle="round,pad=0.05", fc="#f2f2f2" if k != "nav" else "#d9d9d9",
                                    ec="#1a1a1a", lw=1.2))
        ax.text(x + 1.15, y + 0.42, t, ha="center", va="center", fontsize=9)
    arrows = [((2.45, 2.8), (3.6, 2.35), "target"), ((2.45, 0.9), (3.6, 1.45), "robot pose,\nobstacles"),
              ((5.25, 1.88), (5.5, 1.88), ""), ((7.85, 2.1), (8.1, 2.75), "velocity\ntarget"),
              ((7.85, 1.65), (8.1, 0.95), "velocity\ntarget")]
    for a, b, lab in arrows:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, lw=1.1, color="#1a1a1a"))
        if lab:
            ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.05, lab, fontsize=7.5, ha="center", va="bottom")
    ax.text(10.45, 2.82, "12 joint\ntargets", fontsize=7.5, ha="left", va="center")
    ax.text(10.45, 0.92, "Go2 X", fontsize=8.5, ha="left", va="center", weight="bold")
    ax.text(5.5, 0.25, "Q1: camera, language, learned dynamics    Q2: dog data and simulation    Q3: walking and navigation policies (PPO)",
            ha="center", fontsize=8.5, style="italic")
    fig.tight_layout()
    fig.savefig(FIG / "system_diagram.png", dpi=130)
    plt.close(fig)


def ensemble():
    r = json.loads((ROOT / "q1_vision_ml/results/ensemble_eval.json").read_text())
    models = list(r["test"]["0.5 s"].keys())
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, split, title in ((axs[0], "test", "held-out episodes, same walker"),
                             (axs[1], "shift_isaac", "a walker never seen in training")):
        vals = [[r[split]["0.5 s"][m][a]["mean"] for a in ("vx", "vy", "wz")] for m in models]
        errs = [[r[split]["0.5 s"][m][a]["se"] for a in ("vx", "vy", "wz")] for m in models]
        bars(ax, ["vx (m/s)", "vy (m/s)", "wz (rad/s)"], models, vals, errs, "RMSE 0.5 s ahead")
        ax.set_title(f"{title} (n = 30 episodes)", fontsize=10)
    axs[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Q1: predicting the MuJoCo dog's body velocity 0.5 s ahead", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG / "q1_ensemble_rmse.png", dpi=130)
    plt.close(fig)


def small_commands():
    r = json.loads((ROOT / "q2_data/results/small_commands.json").read_text())
    worlds = ["real Go2 X (Unitree controller)", "physics model, Phase 1 ranges", "learned ensemble",
              "MuJoCo dog, MuJoCo walker (mjx_s0)", "MuJoCo dog, Isaac Lab walker"]
    tests = [k for k in r[worlds[0]].keys()]
    short = [t.replace(", ", "\n") for t in tests]
    fig, ax = plt.subplots(figsize=(11, 4))
    bars(ax, short, worlds, [[r[w][t]["fraction"] for t in tests] for w in worlds], ylabel="share of commanded travel")
    ax.axhline(1.0, color="#1a1a1a", lw=0.8, ls="--")
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    ax.set_title("Q2: small single-axis commands, real Go2 X (n = 1 each) against simulation", fontsize=10, pad=34)
    fig.tight_layout()
    fig.savefig(FIG / "q2_small_commands.png", dpi=130)
    plt.close(fig)


def q3():
    p = ROOT / "q3_rl/results/cross_eval.json"
    if not p.exists():
        return
    r = json.loads(p.read_text())
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, w in enumerate(["physics", "learned", "mujoco"]):
        curves = [json.loads(f.read_text()) for f in sorted((ROOT / "q3_rl/runs").glob(f"{w}_s*/progress.json"))]
        if not curves:
            continue
        L = min(len(c) for c in curves)
        x = np.array([c["steps"] for c in curves[0][:L]]) / 1e6
        y = np.array([[c["success"] for c in cv[:L]] for cv in curves])
        m, s = y.mean(0), y.std(0, ddof=1) / np.sqrt(len(y)) if len(y) > 1 else np.zeros(L)
        ax.plot(x, m, ["-", "--", ":"][i], color=SHADES[i], lw=2, label=f"trained in {w} world (n = {len(y)} seeds)")
        ax.fill_between(x, m - s, m + s, color=SHADES[i], alpha=0.15)
    ax.set(xlabel="environment steps (millions)", ylabel="training success (last 200 episodes)",
           title="Q3: PPO navigation, training in three worlds")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "q3_training.png", dpi=130)
    plt.close(fig)
    worlds = r["eval_worlds"]
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ctrls = list(r["success"].keys())
    vals = [[r["success"][c][w]["mean"] for w in worlds] for c in ctrls]
    errs = [[r["success"][c][w]["se"] for w in worlds] for c in ctrls]
    bars(ax, [f"evaluated in {w}" for w in worlds], ctrls, vals, errs, "success rate (100 scenarios)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    ax.set_title("Q3: cross-evaluation (mean over seeds ± standard error)", fontsize=10, pad=40)
    fig.tight_layout()
    fig.savefig(FIG / "q3_cross_eval.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    system_diagram()
    ensemble()
    small_commands()
    q3()
    print("figures written")
