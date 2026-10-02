"""How much faster is axe-v10's model with TensorRT?

Runs inside an axe-v10 container on one GPU. It times the one call the driver
makes per Drive request -- `DriveSuprimPolicy._run_agent(features)` -- so the
number is the model work the 0.1 s budget is about, not the simulator around it.

Three configurations, so the speedup can be attributed:

  fp32       what axe-v10 ships (DRIVESUPRIM_USE_FP16=0)
  fp16       the driver's own autocast switch (DRIVESUPRIM_USE_FP16=1), no TensorRT
  trt_fp16   the image backbone replaced by a TensorRT FP16 engine, rest as fp32

Most of a TensorRT speedup on a transformer usually comes from FP16, not from
TensorRT itself. Measuring fp16 on its own separates the two, so the report can
say how much TensorRT adds beyond flipping a switch the driver already has.

Inputs. The feature tensors are synthetic but shaped exactly as the driver builds
them for bevformer_m (read off its own `BEV INPUT` log line and code):
  bev_imgs, bev_imgs_teacher  (1, 3 frames, 3 cams, 3, 256, 512)
  lidar2img                   (1, 3, 3, 4, 4)
  bev_ego_pose                (1, 3, 3)
  status_feature              two tensors of (1, 8)
  route_feature/_mask         (1, 20, 2) / (1, 20)
  rerank_route_feature/_mask  (1, 64, 2) / (1, 64)
Latency is fixed by tensor shapes, not values, so synthetic inputs time the model
faithfully. The camera matrices are a real virtual-pinhole projection rather than
random, so BEV sampling lands inside the images the way it does on the road.

Correctness is checked, not assumed: every variant's selected trajectory is
compared with fp32's on the same inputs. A speedup that changes the plan is not
the same model.

Writes one JSON file to --out.
"""
import argparse
import json
import math
import os
import statistics
import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/app")


def build_policy():
    from drivesuprim_challenge.policy import DriveSuprimPolicy

    a = os.environ.get("DRIVESUPRIM_ASSET_DIR", "/app/assets/drivesuprim")
    return DriveSuprimPolicy(
        checkpoint_path=os.environ.get("DRIVESUPRIM_CHECKPOINT_PATH", f"{a}/axe_nurec_vits.ckpt"),
        vocab_path=os.environ.get("DRIVESUPRIM_VOCAB_PATH", f"{a}/nurec_train_kmeans_4096x40x3.npy"),
        backbone_type=os.environ.get("DRIVESUPRIM_BACKBONE_TYPE", "bevformer_m"),
        device="cuda",
        config_path=os.environ.get("DRIVESUPRIM_CONFIG_PATH", f"{a}/axe_nurec_vits_config.json"),
    )


def pinhole_lidar2img(yaw_deg: float) -> np.ndarray:
    """A 4x4 projection for a camera at the origin looking along yaw_deg.

    Virtual pinhole K is the driver's own (fx=fy=377, cx=256, cy=128 for a
    512x256 image). Ego frame is x forward, y left, z up; camera frame is
    x right, y down, z forward.
    """
    K = np.array([[377.0, 0, 256.0], [0, 377.0, 128.0], [0, 0, 1.0]])
    y = math.radians(yaw_deg)
    fwd = np.array([math.cos(y), math.sin(y), 0.0])
    right = np.array([math.sin(y), -math.cos(y), 0.0])
    down = np.array([0.0, 0.0, -1.0])
    R = np.stack([right, down, fwd])          # ego -> camera
    t = -R @ np.array([1.5, 0.0, 1.6])         # camera 1.5 m ahead, 1.6 m up
    E = np.eye(4); E[:3, :3] = R; E[:3, 3] = t
    P = np.eye(4); P[:3, :3] = K
    return (P @ E).astype(np.float32)


def make_features(device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    imgs = torch.rand(1, 3, 3, 3, 256, 512, generator=g)
    cams = np.stack([pinhole_lidar2img(y) for y in (60.0, 0.0, -60.0)])  # L0, F0, R0
    l2i = torch.from_numpy(np.stack([cams] * 3))[None]                    # (1,3,3,4,4)
    pose = torch.zeros(1, 3, 3)
    pose[0, :, 0] = torch.tensor([-1.0, -0.5, 0.0])                       # last 1 s of motion
    s = np.linspace(42.0, 80.0, 20, dtype=np.float32)
    route = torch.from_numpy(np.stack([s, np.zeros_like(s)], -1) / 80.0)[None]
    s64 = np.linspace(0.0, 80.0, 64, dtype=np.float32)
    rr = torch.from_numpy(np.stack([s64, np.zeros_like(s64)], -1) / 80.0)[None]
    f = {
        "bev_imgs": imgs,
        "bev_imgs_teacher": imgs.clone(),
        "lidar2img": l2i,
        "bev_ego_pose": pose,
        "status_feature": [torch.tensor([[0, 0, 0, 0, 5.0, 0, 0, 0]]),
                           torch.tensor([[0, 0, 0, 0, 5.0, 0, 0, 0]])],
        "route_feature": route,
        "route_mask": torch.ones(1, 20),
        "rerank_route_feature": rr,
        "rerank_route_mask": torch.ones(1, 64),
    }
    out = {}
    for k, v in f.items():
        out[k] = [t.float().to(device) for t in v] if isinstance(v, list) else v.float().to(device)
    return out


def trajectory_of(pred):
    key = "final_traj" if "final_traj" in pred else "trajectory"
    return pred[key].detach().float().cpu().numpy()


def time_calls(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ms.append((time.perf_counter() - t0) * 1000.0)
    ms.sort()
    q = lambda p: ms[min(len(ms) - 1, int(p * len(ms)))]
    return {"mean_ms": statistics.fmean(ms), "p50_ms": q(0.50), "p90_ms": q(0.90),
            "p99_ms": q(0.99), "min_ms": ms[0], "n": len(ms)}


def profile_modules(model, call, top=12):
    """Per-submodule GPU time, to find what is worth converting."""
    events = {}
    handles = []
    for name, mod in model.named_children():
        def pre(m, i, n=name):
            e = torch.cuda.Event(enable_timing=True); e.record(); events.setdefault(n, []).append([e, None])
        def post(m, i, o, n=name):
            e = torch.cuda.Event(enable_timing=True); e.record(); events[n][-1][1] = e
        handles.append(mod.register_forward_pre_hook(pre))
        handles.append(mod.register_forward_hook(post))
    try:
        for _ in range(5):
            call()
        events.clear()
        for _ in range(20):
            call()
        torch.cuda.synchronize()
    finally:
        for h in handles:
            h.remove()
    rows = []
    for n, pairs in events.items():
        t = [a.elapsed_time(b) for a, b in pairs if b is not None]
        if t:
            rows.append((n, statistics.fmean(t), len(t) / 20))
    rows.sort(key=lambda r: -r[1])
    return [{"module": n, "mean_ms": round(t, 3), "calls_per_forward": c} for n, t, c in rows[:top]]


def find_image_backbone(model):
    """The image encoder is the usual TensorRT target: a plain conv/transformer
    stack with no data-dependent control flow. Locate it by name."""
    for name in ("img_backbone", "image_backbone", "backbone", "_backbone", "img_encoder"):
        for full, mod in model.named_modules():
            if full.split(".")[-1] == name and sum(p.numel() for p in mod.parameters()) > 1e6:
                return full, mod
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--skip-trt", action="store_true")
    a = ap.parse_args()

    torch.backends.cudnn.benchmark = False
    dev = torch.device("cuda")
    report = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "cuda": torch.version.cuda, "results": {}}

    policy = build_policy()
    feats = make_features(dev)
    agent = policy._agent
    model = getattr(agent, "_drivesuprim_model", None) or getattr(agent, "model", None) or agent

    # --- fp32: as shipped ---------------------------------------------------
    policy._use_autocast = False
    with torch.inference_mode():
        ref = trajectory_of(policy._run_agent(feats))
    report["module_profile_fp32"] = profile_modules(model, lambda: policy._run_agent(feats))
    report["results"]["fp32"] = time_calls(lambda: policy._run_agent(feats), a.warmup, a.iters)
    print("fp32", report["results"]["fp32"], flush=True)

    # --- fp16 autocast: the driver's existing switch -------------------------
    policy._use_autocast = True
    out16 = trajectory_of(policy._run_agent(feats))
    report["results"]["fp16"] = time_calls(lambda: policy._run_agent(feats), a.warmup, a.iters)
    report["results"]["fp16"]["max_traj_diff_m_vs_fp32"] = float(np.abs(out16 - ref).max())
    print("fp16", report["results"]["fp16"], flush=True)
    policy._use_autocast = False

    # --- TensorRT: image backbone as an FP16 engine --------------------------
    if not a.skip_trt:
        name, backbone = find_image_backbone(model)
        report["trt_target"] = name
        if backbone is None:
            report["trt_error"] = "no image backbone found by name; see module_profile_fp32"
        else:
            try:
                from trt_backbone import TrtModule
                trt_mod = TrtModule.build(backbone, sample=feats, policy=policy,
                                          workdir=os.environ.get("TRT_WORKDIR", "/tmp/trt"))
                parent = model.get_submodule(name.rsplit(".", 1)[0]) if "." in name else model
                original = getattr(parent, name.rsplit(".", 1)[-1])
                setattr(parent, name.rsplit(".", 1)[-1], trt_mod)
                try:
                    outt = trajectory_of(policy._run_agent(feats))
                    r = time_calls(lambda: policy._run_agent(feats), a.warmup, a.iters)
                    r["max_traj_diff_m_vs_fp32"] = float(np.abs(outt - ref).max())
                    r["engine_build_s"] = trt_mod.build_seconds
                    report["results"]["trt_fp16"] = r
                    print("trt_fp16", r, flush=True)
                finally:
                    setattr(parent, name.rsplit(".", 1)[-1], original)
            except Exception as exc:  # report, do not hide
                report["trt_error"] = f"{type(exc).__name__}: {exc}"
                print("TRT FAILED:", report["trt_error"], flush=True)

    base = report["results"]["fp32"]["p50_ms"]
    for k, v in report["results"].items():
        v["speedup_vs_fp32_p50"] = round(base / v["p50_ms"], 3)
    json.dump(report, open(a.out, "w"), indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
