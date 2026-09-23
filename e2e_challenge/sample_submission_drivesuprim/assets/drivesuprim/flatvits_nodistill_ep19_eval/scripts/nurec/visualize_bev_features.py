"""What the BEV map looks like in each plan-only run, and how much of it is image.

The plan-only agent carries no segmentation or detection head (``use_aux_heads``
is false), so there is nothing that decodes the BEV grid into road and vehicles.
All that can be rendered is the 56x56x256 tensor itself, projected to three
channels by PCA. That shows whether the map has spatial structure; it cannot
say what the structure means.

The measurement below is the part that does compare the two runs. Every
condition is run twice on the same scene -- once with the real images, once with
the images zeroed -- and the per-cell distance between the two BEV maps is how
much of that cell came from the cameras rather than from the query embedding and
the ego pose. A run whose calibration put the reference points outside the image
has nothing to lose when the images go away.

Three conditions, because the old checkpoint has two interesting ones:

  new / correct     the run started 22:39, its own calibration
  old / correct     the 30-epoch checkpoint fed today's rectified projection
  old / as-trained  the same checkpoint under the projection it actually saw:
                    nuPlan's 1920x1080 pinhole (fx 1545, cx 960, cy 560) applied
                    to a 512x256 image, which is what the miscalibrated logs
                    carried

The third is rebuilt from the correct one rather than from the old logs, which
no longer exist:  lidar2img = K_old @ K_new^-1 @ lidar2img_new, both K in
homogeneous 4x4 form. Extrinsics are untouched, so the only thing that changes
is where each BEV reference point lands in the image -- exactly the bug.

  python scripts/nurec/visualize_bev_features.py [N_SCENES]
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

NEW_CKPT = ROOT / "exp/drivesuprim_bevformer_vov_v2_vits_planonly_scratch_nurec_ckpt"
OLD_CKPT = Path("/home1/irteam/workspace/_archive_2026-08-29/checkpoints"
                "/drivesuprim_bevformer_vov_v2_vits_planonly_scratch_nurec_ckpt"
                "/epoch=29-step=43590.ckpt")
AGENT = "drivesuprim_agent_bevformer_vov_v2_vits_planonly_scratch_nurec"
OUT = ROOT / "exp/bev_feature_compare"

# The pinhole the miscalibrated logs carried: nuPlan's, for a 1920x1080 render.
K_OLD = np.array([[1545.0, 0.0, 960.0], [0.0, 1545.0, 560.0], [0.0, 0.0, 1.0]])


def newest_new_ckpt() -> Path:
    ckpts = sorted(NEW_CKPT.glob("epoch=*.ckpt"))
    if not ckpts:
        raise SystemExit(f"no checkpoint yet under {NEW_CKPT}")
    return ckpts[-1]


def build_agent():
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    with initialize_config_dir(
            config_dir=str(ROOT / "navsim/planning/script/config/pdm_scoring"),
            version_base=None):
        cfg = compose(config_name="default_run_pdm_score_gpu_ssl", overrides=[
            f"agent={AGENT}", "++agent.config.training=false",
            "++agent.config.only_ori_input=true",
            "++agent.config.bevformer_vit_pretrained=false",
            "train_test_split=nurec",
            f"++agent.config.vocab_path={os.environ['NUREC_VOCAB_PATH']}",
            "++agent.config.vocab_size=4096",
            f"++agent.config.ori_vocab_pdm_score_full_path={os.environ['NUREC_ORI_PDM_SCORE']}",
            f"++agent.config.ori_vocab_pdm_score_dir={os.environ['NUREC_ORI_PDM_SCORE_DIR']}"])
    return instantiate(cfg.agent)


# Lightning wraps the agent, the agent wraps an SSL meta-arch, and that holds a
# teacher and a student that are the same architecture. The student is the one
# the optimiser touches, so that is the branch to read.
STUDENT = "agent.model.student.model."


def load_weights(model: torch.nn.Module, ckpt: Path) -> Tuple[int, int]:
    sd = torch.load(ckpt, map_location="cpu")
    sd = sd.get("state_dict", sd)
    student = {k[len(STUDENT):]: v for k, v in sd.items() if k.startswith(STUDENT)}
    if not student:
        raise SystemExit(f"{ckpt}: no keys under {STUDENT}")
    missing, unexpected = model.load_state_dict(student, strict=False)
    if missing:
        raise SystemExit(f"{ckpt.name}: {len(missing)} parameters got no weights, "
                         f"e.g. {missing[:3]}")
    return len(missing), len(unexpected)


def scenes(agent, n: int) -> List:
    from navsim.common.dataclasses import SceneFilter
    from navsim.common.dataloader import SceneLoader
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(ROOT / "exp/nurec_vits_nurec_planonly"
                         / "2026.08.29.22.39.41/code/hydra/config.yaml")
    val_logs = list(cfg.val_logs)
    prep = Path(os.environ["NUREC_PREPARED_ROOT"])
    rng = np.random.default_rng(0)
    picked = list(rng.choice(val_logs, min(n * 3, len(val_logs)), replace=False))
    loader = SceneLoader(
        data_path=prep / "navsim_logs/trainval",
        original_sensor_path=Path(os.environ["NUREC_SENSOR_ROOT"]),
        synthetic_sensor_path=None, synthetic_scenes_path=None,
        scene_filter=SceneFilter(num_history_frames=4, num_future_frames=8,
                                 frame_interval=1, has_route=True, max_scenes=None,
                                 log_names=picked, tokens=None,
                                 include_synthetic_scenes=False),
        # Only CAM_L0/F0/R0 were rectified; asking for all eight opens files
        # that were never written. The agent already names the three it needs.
        sensor_config=agent.get_sensor_config())
    out, seen = [], set()
    for token in loader.tokens:
        scene = loader.get_scene_from_token(token)
        log = scene.scene_metadata.log_name
        if log in seen:
            continue
        seen.add(log)
        out.append(scene)
        if len(out) == n:
            break
    return out


def features_for(agent, scene) -> Dict:
    """One sample, batched exactly the way the dataloader would batch it.

    Several entries are lists of per-frame tensors rather than tensors, so the
    batch dimension cannot just be unsqueezed on; default_collate is what the
    training loop uses and it keeps the nesting intact.
    """
    from torch.utils.data._utils.collate import default_collate
    agent_input = scene.get_agent_input()
    feats: Dict = {}
    for b in agent.get_feature_builders():
        feats.update(b.compute_features(agent_input, scene))
    batched = default_collate([feats])
    # Rotation-augmentation views are consumed by forward_features_list, one
    # entry per rotated projection. A single forward wants the un-rotated view,
    # and the backbone reads bev_aug_yaw as a [bs] tensor rather than the list.
    for key in ("bev_lidar2img_rotated", "bev_aug_yaw"):
        batched.pop(key, None)
    return batched


def broken_lidar2img(l2i: torch.Tensor, k_new: np.ndarray) -> torch.Tensor:
    """Re-aim the projection with the pinhole the miscalibrated logs carried."""
    pad = lambda k: np.block([[k, np.zeros((3, 1))], [np.zeros((1, 3)), np.ones((1, 1))]])
    swap = torch.tensor(pad(K_OLD) @ np.linalg.inv(pad(k_new)), dtype=l2i.dtype)
    return torch.einsum("ij,...jk->...ik", swap, l2i)


@torch.no_grad()
def bev_of(model, feats: Dict[str, torch.Tensor], device: str,
           blind: bool = False) -> np.ndarray:
    grabbed = {}

    def hook(_m, _i, out):
        grabbed["bev"] = out.detach().float().cpu()

    def to_dev(v):
        if torch.is_tensor(v):
            return v.to(device)
        if isinstance(v, (list, tuple)):
            return type(v)(to_dev(e) for e in v)
        return v

    x = {k: to_dev(v) for k, v in feats.items()}
    if blind:
        for key in ("bev_imgs", "bev_imgs_teacher"):
            if key in x:
                x[key] = torch.zeros_like(x[key])
    h = model._backbone.register_forward_hook(hook)
    try:
        model(x)
    finally:
        h.remove()
    return grabbed["bev"][0].numpy()          # [C, H, W]


def pca_rgb(bev: np.ndarray) -> np.ndarray:
    """Three leading principal components of the 256-d cell vectors, as RGB.

    Taken from the eigenvectors of the 256x256 covariance rather than from an
    SVD of the 3136x256 matrix: LAPACK's driver fails to converge on some of
    these scenes, and a symmetric eigenproblem that small is both cheaper and
    stable.
    """
    c, h, w = bev.shape
    x = np.nan_to_num(bev.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    flat = x.reshape(c, h * w).T                          # [HW, C]
    flat = flat - flat.mean(0, keepdims=True)
    cov = (flat.T @ flat) / max(flat.shape[0] - 1, 1)
    vals, vecs = np.linalg.eigh(cov)                      # ascending
    rgb = (flat @ vecs[:, ::-1][:, :3]).reshape(h, w, 3)
    lo = np.percentile(rgb, 2, axis=(0, 1))
    hi = np.percentile(rgb, 98, axis=(0, 1))
    return np.clip((rgb - lo) / np.maximum(hi - lo, 1e-8), 0, 1)


def contribution(real: np.ndarray, blind: np.ndarray) -> np.ndarray:
    """Per cell: ||F_real - F_blind|| / ||F_real||, the camera sensitivity.

    Not a share of anything, and not bounded by 1: blanking the images does not
    subtract a component, it produces a different map, and that map can be as
    large as the original and point elsewhere. Zero is the meaningful value --
    it says the cameras changed nothing at all.
    """
    real = np.nan_to_num(real, nan=0.0, posinf=0.0, neginf=0.0)
    blind = np.nan_to_num(blind, nan=0.0, posinf=0.0, neginf=0.0)
    num = np.linalg.norm(real - blind, axis=0)
    den = np.linalg.norm(real, axis=0) + 1e-8
    return num / den


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    device = os.environ.get("NUREC_VIZ_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
    OUT.mkdir(parents=True, exist_ok=True)

    agent = build_agent()
    model = agent.model.student.model.to(device).eval()
    new_ckpt = newest_new_ckpt()
    print(f"new checkpoint: {new_ckpt.name}")
    print(f"old checkpoint: {OLD_CKPT.name}")
    print(f"device: {device}")

    picked = scenes(agent, n)
    print(f"scenes: {len(picked)}")

    conditions = [("new / correct", new_ckpt, False),
                  ("old / correct", OLD_CKPT, False),
                  ("old / as-trained", OLD_CKPT, True)]

    rows: List[Dict] = []
    per_scene: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for label, ckpt, break_calib in conditions:
        miss, unexp = load_weights(model, ckpt)
        print(f"  {label:18s} missing={miss} unexpected={unexp}")
        for scene in picked:
            token = scene.scene_metadata.initial_token
            feats = features_for(agent, scene)
            if break_calib:
                k_new = np.array([[377.0, 0, 256.0], [0, 377.0, 128.0], [0, 0, 1.0]])
                feats = dict(feats)
                feats["lidar2img"] = broken_lidar2img(feats["lidar2img"], k_new)
            real = bev_of(model, feats, device, blind=False)
            blind = bev_of(model, feats, device, blind=True)
            finite = float(np.isfinite(real).mean())
            if finite < 1.0:
                print(f"    {label} {token[:8]}: only {finite:.1%} of cells finite")
            frac = contribution(real, blind)
            per_scene.setdefault(token, {})[label] = {
                "rgb": pca_rgb(real), "frac": frac, "scene": scene, "bev": real}
            rows.append({"condition": label, "token": token,
                         "finite": finite,
                         "feat_std": float(np.nanstd(np.nan_to_num(real, posinf=0, neginf=0))),
                         "mean_frac": float(frac.mean()),
                         "p90_frac": float(np.percentile(frac, 90)),
                         "cells_above_10pct": float((frac > 0.10).mean())})

    # How much of the map is about THIS scene. The leading components of a
    # single BEV are dominated by which cells see a camera at all, and that
    # pattern is the same in every scene -- so it says nothing about the road.
    # Subtracting the across-scene mean leaves only what varies scene to scene.
    print(f"\n{'condition':20s}{'scene-dependent variance':>26s}")
    scene_share = {}
    for label, _, _ in conditions:
        stack = np.stack([per_scene[t][label]["bev"] for t in per_scene])  # [S,C,H,W]
        stack = np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0)
        total = stack.var(axis=0).sum() + ((stack.mean(0) - stack.mean()) ** 2).sum()
        across = stack.var(axis=0).sum()
        scene_share[label] = across / max(total, 1e-12)
        print(f"{label:20s}{scene_share[label]:>25.1%}")

    print(f"\n{'condition':20s}{'camera sens.':>14s}{'p90':>8s}{'cells >10%':>12s}"
          f"{'feat std':>10s}{'finite':>9s}")
    for label, _, _ in conditions:
        r = [x for x in rows if x["condition"] == label]
        print(f"{label:20s}{np.mean([x['mean_frac'] for x in r]):>12.1%}"
              f"{np.mean([x['p90_frac'] for x in r]):>8.2f}"
              f"{np.mean([x['cells_above_10pct'] for x in r]):>11.1%}"
              f"{np.mean([x['feat_std'] for x in r]):>10.3f}"
              f"{np.mean([x['finite'] for x in r]):>9.1%}")

    for token, by_cond in per_scene.items():
        fig, axes = plt.subplots(2, 4, figsize=(17, 8.4))
        scene = next(iter(by_cond.values()))["scene"]
        cams = scene.frames[scene.scene_metadata.num_history_frames - 1].cameras
        for ax, name in zip(axes[0], ("cam_l0", "cam_f0", "cam_r0")):
            img = getattr(cams, name).image
            ax.imshow(img)
            ax.set_title(name.upper().replace("_", " "), fontsize=10)
            ax.axis("off")
        axes[0, 3].axis("off")
        axes[0, 3].text(0.0, 0.5,
                        "BEV 56x56, 1 m per cell\n0-56 m ahead, +-28 m across\n"
                        "ego at the bottom edge\n\nrow 2 left to right:\n"
                        "  PCA of the BEV feature\n  for each condition\n\nforward is up, ego at the\nbottom edge, vehicle-left\non the image left",
                        fontsize=10, va="center")
        for ax, (label, _, _) in zip(axes[1], conditions):
            d = by_cond[label]
            ax.imshow(np.flipud(np.rot90(d["rgb"], k=-1)), interpolation="nearest")
            ax.set_title(f"{label}\ncamera sensitivity {d['frac'].mean():.2f}", fontsize=10)
            ax.axis("off")
        d = by_cond["new / correct"]
        im = axes[1, 3].imshow(np.flipud(np.rot90(d["frac"], k=-1)),
                               cmap="magma", vmin=0, vmax=1.5, interpolation="nearest")
        axes[1, 3].set_title("new: per-cell camera sensitivity", fontsize=10)
        axes[1, 3].axis("off")
        fig.colorbar(im, ax=axes[1, 3], fraction=0.046)
        fig.suptitle(f"plan-only BEV features  |  token {token}", fontsize=12)
        fig.tight_layout()
        path = OUT / f"bev_{token[:12]}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
