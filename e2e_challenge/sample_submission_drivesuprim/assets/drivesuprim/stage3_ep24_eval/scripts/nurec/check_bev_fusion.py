"""Does the BEV actually fuse all three cameras and all three frames?

The BEV front-end is fed ``bev_imgs`` of shape [B, T, num_cam, 3, H, W] with
T = bev_seq_len = 3 (current frame last) and num_cam = 3 in the order
[CAM_L0, CAM_F0, CAM_R0]. Nothing in the loss checks that each of those nine
views reaches the BEV map: a camera whose calibration is wrong, or a past frame
whose ego warp collapses, contributes nothing and the training curve does not
notice.

Blank one input at a time and measure what changes. For each cell,

    sensitivity = || F_full - F_ablated || / || F_full ||

so zero means the model's BEV is bit-identical without that input -- it is not
being used. A camera that is used leaves a lobe on its own side of the grid; a
past frame that is used leaves a diffuse contribution, weaker than the current
frame but not zero.

  python scripts/nurec/check_bev_fusion.py [N_SCENES]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/nurec"))

from visualize_bev_features import (  # noqa: E402
    bev_of, build_agent, contribution, features_for, load_weights,
    newest_new_ckpt, scenes,
)

CAM_NAMES = ["CAM_L0", "CAM_F0", "CAM_R0"]
OUT = ROOT / "exp/bev_feature_compare"


def ablate(feats: Dict, cam: int = None, frame: int = None) -> Dict:
    out = dict(feats)
    for key in ("bev_imgs", "bev_imgs_teacher"):
        if key not in out:
            continue
        x = out[key].clone()                       # [B, T, cam, 3, H, W]
        if cam is not None:
            x[:, :, cam] = 0.0
        if frame is not None:
            x[:, frame] = 0.0
        out[key] = x
    return out


@torch.no_grad()
def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    device = os.environ.get("NUREC_VIZ_DEVICE",
                            "cuda:0" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)

    agent = build_agent()
    model = agent.model.student.model.to(device).eval()
    ckpt = newest_new_ckpt()
    load_weights(model, ckpt)
    print(f"checkpoint: {ckpt.name}\ndevice: {device}")

    picked = scenes(agent, n)
    seq_len = agent._config.bev_seq_len
    print(f"scenes: {len(picked)}   bev_seq_len: {seq_len}   cameras: {CAM_NAMES}")

    cam_maps: List[List[np.ndarray]] = [[] for _ in CAM_NAMES]
    frame_maps: List[List[np.ndarray]] = [[] for _ in range(seq_len)]
    first: Dict = {}

    for scene in picked:
        feats = features_for(agent, scene)
        full = bev_of(model, feats, device)
        for c in range(len(CAM_NAMES)):
            m = contribution(full, bev_of(model, ablate(feats, cam=c), device))
            cam_maps[c].append(m)
        for t in range(seq_len):
            m = contribution(full, bev_of(model, ablate(feats, frame=t), device))
            frame_maps[t].append(m)
        if not first:
            first = {"token": scene.scene_metadata.initial_token,
                     "cams": [cam_maps[c][0] for c in range(len(CAM_NAMES))],
                     "frames": [frame_maps[t][0] for t in range(seq_len)],
                     "scene": scene}

    def show(maps: List[List[np.ndarray]], names: List[str], kind: str) -> None:
        print(f"\n{kind:24s}{'mean sens.':>12s}{'p90':>8s}{'cells >1%':>11s}{'verdict':>12s}")
        for name, ms in zip(names, maps):
            a = np.stack(ms)
            mean, p90 = a.mean(), np.percentile(a, 90)
            live = (a > 0.01).mean()
            verdict = "unused" if mean < 1e-6 else ("weak" if mean < 0.01 else "used")
            print(f"{name:24s}{mean:>12.4f}{p90:>8.3f}{live:>10.1%}{verdict:>12s}")

    show(cam_maps, CAM_NAMES, "camera ablated")
    show(frame_maps, [f"frame t-{seq_len - 1 - t} ({'current' if t == seq_len - 1 else 'past'})"
                      for t in range(seq_len)], "frame ablated")

    ncol = max(len(CAM_NAMES), seq_len)
    fig, axes = plt.subplots(2, ncol, figsize=(4.4 * ncol, 9.2))
    vmax = max(np.percentile(np.stack(first["cams"]), 99),
               np.percentile(np.stack(first["frames"]), 99), 1e-6)
    for ax, name, m in zip(axes[0], CAM_NAMES, first["cams"]):
        im = ax.imshow(np.flipud(np.rot90(m, k=-1)), cmap="magma", vmin=0, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(f"without {name}\nmean {m.mean():.3f}", fontsize=11)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    labels = [f"without frame t-{seq_len - 1 - t}" + (" (current)" if t == seq_len - 1 else "")
              for t in range(seq_len)]
    for ax, name, m in zip(axes[1], labels, first["frames"]):
        im = ax.imshow(np.flipud(np.rot90(m, k=-1)), cmap="magma", vmin=0, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(f"{name}\nmean {m.mean():.3f}", fontsize=11)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in list(axes[0][len(CAM_NAMES):]) + list(axes[1][seq_len:]):
        ax.axis("off")
    fig.suptitle("BEV sensitivity to each camera and each frame  |  "
                 f"forward up, ego at the bottom  |  token {first['token'][:12]}",
                 fontsize=13)
    fig.tight_layout()
    path = OUT / f"fusion_{first['token'][:12]}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
