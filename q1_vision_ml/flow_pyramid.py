"""Q1 (A), vision: Gaussian pyramid and pyramidal Lucas-Kanade optical flow (Bouguet 2001) on the real Go2 X
front-camera frames recorded over WebRTC in Exam 1 (q2_data/dataset/real_exam1/camera_frames.jpg, four 640x360
frames from a 20 s recording of the dog standing still).

1. Real frames: flow between consecutive frames of the standing dog (camera sway plus noise).
2. Known motion: each real frame is shifted by a known translation (2 to 64 px, random direction) with sensor
   noise added; Shi-Tomasi corners are tracked with Lucas-Kanade using 0 to 4 pyramid levels. Endpoint error
   (distance between the tracked and true position) and the share of points tracked within 1 px show how far
   each pyramid depth can follow a motion. When the dog turns, the scene shifts across the image by tens of
   pixels per frame, which is the case the pyramid is for.

    python q1_vision_ml/flow_pyramid.py
"""
import json
import math
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "q2_data/dataset/real_exam1"
OUT = ROOT / "q1_vision_ml/results"
FIG = ROOT / "figures"
SHIFTS = [2, 4, 8, 16, 24, 32, 48, 64]
LEVELS = [0, 1, 2, 3, 4]
TRIALS = 20
WIN = (21, 21)


def frames():
    img = cv2.imread(str(SRC / "camera_frames.jpg"))
    h, w = img.shape[0] // 2, img.shape[1] // 2
    return [cv2.cvtColor(img[r * h:(r + 1) * h, c * w:(c + 1) * w], cv2.COLOR_BGR2GRAY) for r in (0, 1) for c in (0, 1)]


def corners(g):
    p = cv2.goodFeaturesToTrack(g, maxCorners=300, qualityLevel=0.01, minDistance=8, blockSize=7)
    return p.reshape(-1, 2).astype(np.float32)


def track(a, b, pts, levels):
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(a, b, pts.reshape(-1, 1, 2), None, winSize=WIN, maxLevel=levels, criteria=crit)
    return nxt.reshape(-1, 2), st.reshape(-1).astype(bool)


def pyramid_figure(g):
    levels = [g]
    for _ in range(4):
        levels.append(cv2.pyrDown(levels[-1]))
    fig, ax = plt.subplots(1, 5, figsize=(15, 2.6), gridspec_kw={"width_ratios": [16, 8, 4, 2, 1]})
    for i, (a, L) in enumerate(zip(ax, levels)):
        a.imshow(L, cmap="gray")
        a.set_title(f"level {i}: {L.shape[1]}x{L.shape[0]} px", fontsize=9)
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Gaussian pyramid of a real Go2 X front-camera frame (5x5 Gaussian blur, then half resolution per level)", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "q1_gaussian_pyramid.png", dpi=110)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    fs = frames()
    pyramid_figure(fs[0])
    res = {"source": "q2_data/dataset/real_exam1/camera_frames.jpg (Exam 1, WebRTC, 1280x720 stream, tiled 640x360)",
           "window_px": WIN[0], "trials_per_shift": TRIALS}
    # 1. consecutive real frames of the standing dog
    real = []
    for i in range(3):
        p = corners(fs[i])
        q, ok = track(fs[i], fs[i + 1], p, 3)
        d = np.linalg.norm(q[ok] - p[ok], axis=1)
        real.append({"pair": f"{i + 1}->{i + 2}", "points": int(ok.sum()), "median_px": float(np.median(d)),
                     "p95_px": float(np.percentile(d, 95))})
    res["real_consecutive"] = real
    # 2. known shifts on real frames
    rng = np.random.default_rng(0)
    table = {L: {s: [] for s in SHIFTS} for L in LEVELS}
    for s in SHIFTS:
        for t in range(TRIALS):
            g = fs[t % 4]
            a = rng.uniform(0, 2 * math.pi)
            dx, dy = s * math.cos(a), s * math.sin(a)
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            moved = cv2.warpAffine(g, M, (g.shape[1], g.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            noisy = lambda im: np.clip(im.astype(np.float32) + rng.normal(0, 2.0, im.shape), 0, 255).astype(np.uint8)  # noqa: E731
            a0, b0 = noisy(g), noisy(moved)
            p = corners(a0)
            keep = (p[:, 0] + dx > 10) & (p[:, 0] + dx < g.shape[1] - 10) & (p[:, 1] + dy > 10) & (p[:, 1] + dy < g.shape[0] - 10)
            p = p[keep]
            truth = p + [dx, dy]
            for L in LEVELS:
                q, ok = track(a0, b0, p, L)
                epe = np.linalg.norm(q - truth, axis=1)
                epe[~ok] = np.inf
                table[L][s].append({"median_epe": float(np.median(epe)), "within_1px": float(np.mean(epe < 1.0))})
    summ = {}
    for L in LEVELS:
        summ[L] = {}
        for s in SHIFTS:
            w = np.array([r["within_1px"] for r in table[L][s]])
            m = np.array([r["median_epe"] for r in table[L][s]])
            summ[L][s] = {"within_1px": {"mean": float(w.mean()), "se": float(w.std(ddof=1) / math.sqrt(len(w)))},
                          "median_epe_px": float(np.median(m))}
    res["known_shift"] = {str(L): {str(s): v for s, v in d.items()} for L, d in summ.items()}
    (OUT / "flow_pyramid.json").write_text(json.dumps(res, indent=1))
    # figure: share within 1 px against shift
    fig, ax = plt.subplots(figsize=(7, 4))
    styles = ["-o", "--s", "-.^", ":D", "-v"]
    for L, st in zip(LEVELS, styles):
        y = [summ[L][s]["within_1px"]["mean"] for s in SHIFTS]
        e = [summ[L][s]["within_1px"]["se"] for s in SHIFTS]
        ax.errorbar(SHIFTS, y, yerr=e, fmt=st, color=str(0.15 + 0.15 * L), capsize=3, label=f"{L} pyramid levels")
    ax.set(xlabel="true image shift (px)", ylabel="share of corners tracked within 1 px",
           title=f"Lucas-Kanade on real Go2 X frames, {WIN[0]}x{WIN[0]} px window (n = {TRIALS} per point)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(SHIFTS)
    ax.set_xticklabels(SHIFTS)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "q1_optical_flow_pyramid.png", dpi=120)
    plt.close(fig)
    # markdown
    lines = ["# Pyramidal Lucas-Kanade on real Go2 X frames", "",
             f"Share of Shi-Tomasi corners tracked within 1 px of the true position, mean ± standard error over {TRIALS} "
             f"trials per shift (real frames, known shift in a random direction, Gaussian noise sd 2 gray levels), "
             f"{WIN[0]}x{WIN[0]} px window.", "",
             "| shift (px) | " + " | ".join(f"{L} levels" for L in LEVELS) + " |",
             "| ---: | " + " | ".join("---:" for _ in LEVELS) + " |"]
    for s in SHIFTS:
        lines.append(f"| {s} | " + " | ".join(f"{summ[L][s]['within_1px']['mean']:.3f} ± {summ[L][s]['within_1px']['se']:.3f}" for L in LEVELS) + " |")
    lines += ["", "Consecutive real frames of the standing dog (3 pyramid levels):", "",
              "| frames | corners tracked | median flow (px) | 95th percentile (px) |", "| :--- | ---: | ---: | ---: |"]
    for r in real:
        lines.append(f"| {r['pair']} | {r['points']} | {r['median_px']:.2f} | {r['p95_px']:.2f} |")
    (OUT / "flow_pyramid.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
