"""Compare DriveSuprim FP32 and native-FP16 trajectory outputs."""

from __future__ import annotations

import argparse
import gc
import os

import numpy as np
import torch

from drivesuprim_challenge.policy import DriveSuprimPolicy


def _load(asset_dir: str, *, fp16_mode: str) -> DriveSuprimPolicy:
    os.environ["DRIVESUPRIM_USE_FP16"] = "1" if fp16_mode != "off" else "0"
    os.environ["DRIVESUPRIM_NATIVE_FP16"] = "1" if fp16_mode == "native" else "0"
    return DriveSuprimPolicy(
        checkpoint_path=f"{asset_dir}/drivesuprim_vov.ckpt",
        backbone_path=f"{asset_dir}/dd3d_det_final.pth",
        vocab_path=f"{asset_dir}/test_8192_kmeans.npy",
        backbone_type="vov",
        device="cuda",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", default="/app/assets/drivesuprim")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument(
        "--fp16-mode", choices=("autocast", "native"), default="autocast"
    )
    args = parser.parse_args()

    rng = np.random.default_rng(20260803)
    requests = []
    for index in range(args.samples):
        images = {
            role: rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
            for role in ("left", "front", "right")
        }
        requests.append(
            {
                "images": images,
                "command": index % 3,
                "velocity_xy": (float(index), float(index) / 2),
                "acceleration_xy": (0.1, -0.1),
            }
        )

    fp32_policy = _load(args.asset_dir, fp16_mode="off")
    fp32 = [fp32_policy.predict_batch([request])[0] for request in requests]
    del fp32_policy
    gc.collect()
    torch.cuda.empty_cache()

    fp16_policy = _load(args.asset_dir, fp16_mode=args.fp16_mode)
    fp16 = [fp16_policy.predict_batch([request])[0] for request in requests]

    xy_errors = np.concatenate(
        [
            np.linalg.norm(a.trajectory_xy - b.trajectory_xy, axis=-1)
            for a, b in zip(fp32, fp16, strict=True)
        ]
    )
    heading_errors = np.concatenate(
        [
            np.abs(a.headings - b.headings)
            for a, b in zip(fp32, fp16, strict=True)
        ]
    )
    print(
        f"samples={args.samples} fp16_mode={args.fp16_mode} "
        f"xy_mean_m={xy_errors.mean():.6f} xy_p95_m={np.quantile(xy_errors, 0.95):.6f} "
        f"xy_max_m={xy_errors.max():.6f} heading_mean_rad={heading_errors.mean():.6f} "
        f"heading_max_rad={heading_errors.max():.6f}"
    )


if __name__ == "__main__":
    main()
