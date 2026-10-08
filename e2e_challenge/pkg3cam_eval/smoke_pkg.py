# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""GPU smoke for a mounted three-camera AXEv1.0-NuRec bundle.

Runs inside the axe-v9 image with the same mounts and environment as the
evaluation drivers. Loads the policy exactly as a driver does (strict
checkpoint, axe-v9 MSDA setting checked at construction), then runs one real
forward at the model's own image size with a per-session ego_geom, and checks
that every output is finite and the plan has the expected shape.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/app")
sys.path.insert(0, "/checks")
from bench_trt import build_policy, trajectory_of  # noqa: E402


def pinhole(yaw_deg, width, height):
    """The driver's virtual pinhole (K377 at 512x256) scaled to width x height."""
    sx, sy = width / 512.0, height / 256.0
    K = np.array([[377.0 * sx, 0, 256.0 * sx], [0, 377.0 * sy, 128.0 * sy], [0, 0, 1.0]])
    y = np.radians(yaw_deg)
    R = np.stack([[np.sin(y), -np.cos(y), 0.0], [0.0, 0.0, -1.0], [np.cos(y), np.sin(y), 0.0]])
    E = np.eye(4)
    E[:3, :3] = R
    E[:3, 3] = -R @ np.array([1.5, 0.0, 1.6])
    P = np.eye(4)
    P[:3, :3] = K
    return (P @ E).astype(np.float32)


def main():
    torch.set_num_threads(1)
    policy = build_policy()
    cfg = policy._config
    W, H, T = int(cfg.bev_img_width), int(cfg.bev_img_height), int(cfg.bev_seq_len)
    dev = torch.device("cuda")
    g = torch.Generator().manual_seed(0)
    imgs = torch.rand(1, T, 3, 3, H, W, generator=g)
    cams = np.stack([pinhole(y, W, H) for y in (60.0, 0.0, -60.0)])
    s = np.linspace(42.0, 80.0, 20, dtype=np.float32)
    feats = {
        "bev_imgs": imgs,
        "bev_imgs_teacher": imgs.clone(),
        "lidar2img": torch.from_numpy(np.stack([cams] * T))[None],
        "bev_ego_pose": torch.zeros(1, T, 3),
        "status_feature": [torch.tensor([[0, 0, 0, 0, 5.0, 0, 0, 0]])] * 2,
        "route_feature": torch.from_numpy(np.stack([s, np.zeros_like(s)], -1) / 80.0)[None],
        "route_mask": torch.ones(1, 20),
        "ego_geom": torch.tensor([[4.6 * 0.98, 1.9 * 0.98, 1.467]]),
    }
    feats = {k: ([t.float().to(dev) for t in v] if isinstance(v, list) else v.float().to(dev))
             for k, v in feats.items()}
    torch.cuda.reset_peak_memory_stats()
    prediction = policy._run_agent(feats)
    trajectory = trajectory_of(prediction)
    assert trajectory.shape in ((1, 40, 3), (40, 3)), trajectory.shape
    bad = [k for k, v in prediction.items() if isinstance(v, torch.Tensor) and not torch.isfinite(v).all()]
    assert not bad, f"non-finite outputs: {bad}"
    report = {
        "passed": True,
        "strict_checkpoint": True,
        "image_size": [W, H],
        "bev_seq_len": T,
        "camera_count": int(cfg.bev_num_cameras),
        "bev_size": [int(cfg.bev_h), int(cfg.bev_w)],
        "bev_num_encoder_layers": int(cfg.bev_num_encoder_layers),
        "image_backbone": str(getattr(cfg, "bevformer_img_backbone_type", "")),
        "vocab_size": int(cfg.vocab_size),
        "device": torch.cuda.get_device_name(0),
        "trajectory_shape": list(trajectory.shape),
        "trajectory_endpoint": np.asarray(trajectory).reshape(-1, 3)[-1].tolist(),
        "peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20),
        "peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20),
    }
    Path("/report/gpu_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS:", json.dumps(report))


if __name__ == "__main__":
    main()
