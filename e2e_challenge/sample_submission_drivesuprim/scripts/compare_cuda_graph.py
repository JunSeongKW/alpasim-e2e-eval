#!/usr/bin/env python3
"""Compare eager and CUDA Graph outputs on identical FP16 inputs."""

from __future__ import annotations

import argparse

import numpy as np

from drivesuprim_challenge.policy import DriveSuprimPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", default="/app/assets/drivesuprim")
    parser.add_argument("--samples", type=int, default=5)
    args = parser.parse_args()

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
        images = {
            role: rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
            for role in ("left", "front", "right")
        }
        request = {
            "images": images,
            "command": index % 3,
            "velocity_xy": (float(index), 0.25 * index),
            "acceleration_xy": (0.1 * index, -0.05 * index),
        }
        requests.append(request)
        eager.append(policy.predict_batch([request])[0].trajectory_xy.copy())

    policy._initialize_cuda_graph()
    graph = [
        policy.predict_batch([request])[0].trajectory_xy.copy()
        for request in requests
    ]
    errors = np.abs(np.asarray(eager) - np.asarray(graph))
    print(
        f"samples={args.samples} xy_mae_m={errors.mean():.9f} "
        f"xy_max_error_m={errors.max():.9f} exact_equal={np.array_equal(eager, graph)}"
    )


if __name__ == "__main__":
    main()
