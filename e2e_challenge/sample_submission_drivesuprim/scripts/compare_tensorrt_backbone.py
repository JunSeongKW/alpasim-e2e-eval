#!/usr/bin/env python3
"""Compare eager and TensorRT-backbone final trajectories on identical inputs."""

from __future__ import annotations

import argparse
import os

import numpy as np

from drivesuprim_challenge.policy import DriveSuprimPolicy, _TensorRTBackbone


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", default="/app/assets/drivesuprim")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

    os.environ.pop("DRIVESUPRIM_TENSORRT_BACKBONE_PATH", None)
    policy = DriveSuprimPolicy(
        checkpoint_path=f"{args.asset_dir}/drivesuprim_vov.ckpt",
        backbone_path=f"{args.asset_dir}/dd3d_det_final.pth",
        vocab_path=f"{args.asset_dir}/test_8192_kmeans.npy",
        backbone_type="vov",
        device="cuda",
    )
    rng = np.random.default_rng(20260803)
    requests = []
    eager = []
    for index in range(args.samples):
        request = {
            "images": {
                role: rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
                for role in ("left", "front", "right")
            },
            "command": index % 3,
            "velocity_xy": (float(index), 0.25 * index),
            "acceleration_xy": (0.1 * index, -0.05 * index),
        }
        requests.append(request)
        eager.append(policy.predict_batch([request])[0].trajectory_xy.copy())

    import torch
    import torch_tensorrt  # noqa: F401

    engine = torch.jit.load(args.engine, map_location="cuda")
    policy.model._backbone = _TensorRTBackbone(engine.eval())
    tensorrt_outputs = [
        policy.predict_batch([request])[0].trajectory_xy.copy()
        for request in requests
    ]
    errors = np.abs(np.asarray(eager) - np.asarray(tensorrt_outputs))
    print(
        f"samples={args.samples} xy_mae_m={errors.mean():.9f} "
        f"xy_max_error_m={errors.max():.9f} "
        f"exact_equal={np.array_equal(eager, tensorrt_outputs)}"
    )


if __name__ == "__main__":
    main()
