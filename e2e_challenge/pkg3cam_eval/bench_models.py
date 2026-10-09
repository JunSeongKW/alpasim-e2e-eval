# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""Model latency of one evaluated policy, measured alone on one GPU.

Runs inside the axe-v9 image with exactly the mounts and environment its 441
drivers had (run_bench_models.py builds them), so the code path timed is the
one that was evaluated: DriveSuprimPolicy._run_agent, i.e. the model forward a
Drive call makes, batch 1, FP32 with the configs' own internal ViT AMP.

Inputs are synthetic but shaped as the driver builds them for this model
(bev_seq_len frames x 3 cameras at bev_img_width x bev_img_height, lidar2img,
ego pose, status, route, ego_geom when the source takes it). Latency depends on
shapes, not values.

Reports per-call latency (mean / p50 / p90 / p99 / min, FPS), peak GPU memory,
parameter counts, and a per-component GPU-time breakdown from CUDA events on the
modules the inference forward actually enters (hooks add a little overhead, so
the breakdown is measured in a separate pass from the latency).
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/app")
sys.path.insert(0, "/checks")
from bench_trt import build_policy  # noqa: E402

COMPONENTS = {
    "image_backbone": "teacher.model._backbone.image_encoder.image_backbone",
    "bev_encoder": "teacher.model._backbone.image_encoder.encoder",
    "bev_frontend_total": "teacher.model._backbone.image_encoder",
    "trajectory_head": "teacher.model._trajectory_head",
    "trajectory_offset_head": "teacher.model._trajectory_offset_head",
    "agent_head": "teacher.model._agent_head",
    "model_total": "teacher.model",
}


def pinhole(yaw_deg, width, height):
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


def features(cfg, dev, ego_geom):
    W, H, T = int(cfg.bev_img_width), int(cfg.bev_img_height), int(cfg.bev_seq_len)
    g = torch.Generator().manual_seed(0)
    imgs = torch.rand(1, T, 3, 3, H, W, generator=g)
    cams = np.stack([pinhole(y, W, H) for y in (60.0, 0.0, -60.0)])
    s = np.linspace(42.0, 80.0, 20, dtype=np.float32)
    f = {
        "bev_imgs": imgs, "bev_imgs_teacher": imgs.clone(),
        "lidar2img": torch.from_numpy(np.stack([cams] * T))[None],
        "bev_ego_pose": torch.zeros(1, T, 3),
        "status_feature": [torch.tensor([[0, 0, 0, 0, 5.0, 0, 0, 0]])] * 2,
        "route_feature": torch.from_numpy(np.stack([s, np.zeros_like(s)], -1) / 80.0)[None],
        "route_mask": torch.ones(1, 20),
    }
    if ego_geom:
        f["ego_geom"] = torch.tensor([[4.6 * 0.98, 1.9 * 0.98, 1.467]])
    return {k: ([t.float().to(dev) for t in v] if isinstance(v, list) else v.float().to(dev))
            for k, v in f.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ego-geom", action="store_true")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    dev = torch.device("cuda")
    policy = build_policy()
    cfg = policy._config
    feats = features(cfg, dev, a.ego_geom)
    agent = policy._agent
    model = getattr(agent, "model", agent)

    def call():
        policy._run_agent(feats)

    for _ in range(a.warmup):
        call()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ms = []
    for _ in range(a.iters):
        t0 = time.perf_counter()
        call()
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000.0)
    s = sorted(ms)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]
    latency = {"mean_ms": statistics.fmean(ms), "p50_ms": q(.5), "p90_ms": q(.9),
               "p99_ms": q(.99), "min_ms": s[0], "std_ms": statistics.pstdev(ms),
               "fps_mean": 1000.0 / statistics.fmean(ms), "n": len(ms)}
    mem = {"peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
           "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}

    # component breakdown: CUDA events on module entry / exit
    events, handles = {}, []
    for key, path in COMPONENTS.items():
        try:
            mod = agent.get_submodule("model." + path) if hasattr(agent, "model") else agent.get_submodule(path)
        except AttributeError:
            continue

        def pre(m, i, k=key):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            events.setdefault(k, []).append([e, None])

        def post(m, i, o, k=key):
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            events[k][-1][1] = e

        handles += [mod.register_forward_pre_hook(pre), mod.register_forward_hook(post)]
    for _ in range(10):
        call()
    torch.cuda.synchronize()
    events.clear()
    reps = 50
    for _ in range(reps):
        call()
    torch.cuda.synchronize()
    for h in handles:
        h.remove()
    breakdown = {}
    for k, pairs in events.items():
        t = [x.elapsed_time(y) for x, y in pairs if y is not None]
        breakdown[k] = {"ms_per_forward": sum(t) / reps, "calls_per_forward": len(t) / reps}

    def params(path):
        try:
            m = agent.get_submodule("model." + path) if hasattr(agent, "model") else agent.get_submodule(path)
            return sum(p.numel() for p in m.parameters()) / 1e6
        except AttributeError:
            return None

    report = {
        "name": a.name, "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
        "config": {"bev_img": [int(cfg.bev_img_width), int(cfg.bev_img_height)],
                   "bev_seq_len": int(cfg.bev_seq_len), "bev_num_cameras": int(cfg.bev_num_cameras),
                   "bev_grid": [int(cfg.bev_h), int(cfg.bev_w)],
                   "bev_num_encoder_layers": int(cfg.bev_num_encoder_layers),
                   "image_backbone": str(getattr(cfg, "bevformer_img_backbone_type", "")),
                   "vocab_size": int(cfg.vocab_size), "autocast_fp16": bool(policy._use_autocast)},
        "params_m": {"inference_model": params("teacher.model"),
                     "image_backbone": params(COMPONENTS["image_backbone"]),
                     "whole_agent_incl_student": sum(p.numel() for p in agent.parameters()) / 1e6},
        "latency": latency, "memory": mem, "breakdown": breakdown,
    }
    Path(a.out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
