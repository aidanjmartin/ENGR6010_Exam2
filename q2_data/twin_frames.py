"""Q2 digital twin: the overhead camera's frame at the last policy step of a real run beside the digital twin's frame
at the same moment. The twins (twin.mp4 in each run folder) were rendered on the MacBook from the run logs alone by
project/scripts/robot/render_twin.py: one frame per sport-mode state message (20 Hz), robot pose from the camera
poses the policy used, joint angles from the low-state messages, obstacles from the last perceived layout.

    python q2_data/twin_frames.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from exam2.realruns import RUNS_DIR, load_run  # noqa: E402

RUNS = ["20261107T015345Z", "20261107T020049Z", "20261107T015741Z", "20261107T015621Z"]
H = 300


def frame(video, index, out):
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vf", f"select=eq(n\\,{index})", "-frames:v", "1",
                    str(out)], check=True)
    return Image.open(out).convert("RGB")


def main():
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for name in RUNS:
            folder = next(RUNS_DIR.glob(f"{name}_*"))
            r = load_run(folder)
            k = len(r["t"]) - 1
            cam = frame(folder / "camera.mp4", k, Path(tmp) / "c.png")
            i = int(np.argmin(np.abs(r["st_t"] - r["t"][k])))
            twin = frame(folder / "twin.mp4", i, Path(tmp) / "t.png")
            cam = cam.resize((int(cam.width * H / cam.height), H))
            twin = twin.resize((int(twin.width * H / twin.height), H))
            row = Image.new("RGB", (cam.width + twin.width + 10, H + 22), "white")
            row.paste(cam, (0, 22))
            row.paste(twin, (cam.width + 10, 22))
            d = ImageDraw.Draw(row)
            d.text((4, 4), f"{name} UTC, '{r['target']}': {r['outcome']}. Left: overhead camera at the last policy step. "
                           f"Right: digital twin at the same time (fixed camera, chase camera, policy input).", fill="black")
            rows.append(row)
    W = max(r.width for r in rows)
    sheet = Image.new("RGB", (W, sum(r.height for r in rows) + 8 * len(rows)), "white")
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height + 8
    out = ROOT / "figures/q2_twin_vs_camera.jpg"
    sheet.save(out, quality=88)
    print(f"wrote {out} ({sheet.width} x {sheet.height})")


if __name__ == "__main__":
    main()
