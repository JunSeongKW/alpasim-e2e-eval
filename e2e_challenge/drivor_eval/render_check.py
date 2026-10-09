# SPDX-License-Identifier: Apache-2.0
"""Render check of the PAI -> nuPlan mapping (drivor-eval env).

    python render_check.py --clip-dir DIR [--clip-dir DIR ...] --out check.png \
        [--nuplan-scene /home/kaist5/Dataset/navsim/download/sensor_blobs/mini/<log>]

Each DIR comes from asl_frames.py. For every step folder the figure shows the
four views DrivoR will receive, in its slot order [cam_f0, cam_b0, cam_l0,
cam_r0], next to a real nuPlan frame of the same four cameras for comparison.
Prints per-view coverage (share of target pixels some PAI camera sees).
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from pai_views import NEEDED_CAMERAS, TARGET_ORDER, NuplanRig, SessionViews  # noqa: E402


def load_rgb(path):
    return np.asarray(Image.open(path).convert("RGB"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip-dir", type=Path, action="append", required=True)
    ap.add_argument("--nuplan-scene", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rig = NuplanRig()
    rays = rig.pixel_rays()
    rows = []
    for clip in a.clip_dir:
        req = json.loads((clip / "session_request.json").read_text())
        specs = {c["logical_id"]: c for c in req["rollout_spec"]["vehicle"]["available_cameras"]}
        views = SessionViews(specs, rig, rays)
        for step in sorted(clip.glob("step*")):
            images = {c: load_rgb(step / f"{c}.jpg") for c in NEEDED_CAMERAS}
            out = views.render_all(images)
            cov = {t: views.views[t].coverage for t in TARGET_ORDER}
            print(clip.name, step.name, "coverage", {k: round(v, 4) for k, v in cov.items()})
            rows.append((f"{clip.name[:14]} {step.name}", [out[t] for t in TARGET_ORDER], cov))
    if a.nuplan_scene:
        cams = {t: sorted((a.nuplan_scene / t.upper()).glob("*.jpg")) for t in TARGET_ORDER}
        k = len(cams["cam_f0"]) // 2
        rows.append(("nuPlan (train domain)", [load_rgb(cams[t][k]) for t in TARGET_ORDER], None))
    fig, axes = plt.subplots(len(rows), 4, figsize=(20, 3.1 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (label, imgs, cov) in enumerate(rows):
        for c, (t, img) in enumerate(zip(TARGET_ORDER, imgs)):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            title = t if cov is None else f"{t}  (coverage {cov[t]:.1%})"
            ax.set_title(title, fontsize=10)
            if c == 0:
                ax.set_ylabel(label, fontsize=10)
    fig.tight_layout()
    fig.savefig(a.out, dpi=70)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
