"""Benchmark the exact online DriveSuprim policy path inside the submission image."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from drivesuprim_challenge.policy import DriveSuprimPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", default="/app/assets/drivesuprim")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--backbone-path")
    parser.add_argument("--vocab-path")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--fp16-autocast", action="store_true")
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    asset_dir = Path(args.asset_dir)

    policy = DriveSuprimPolicy(
        checkpoint_path=args.checkpoint_path or asset_dir / "drivesuprim_vov.ckpt",
        backbone_path=args.backbone_path or asset_dir / "dd3d_det_final.pth",
        vocab_path=args.vocab_path or asset_dir / "test_8192_kmeans.npy",
        backbone_type="vov",
        device="cuda",
    )
    image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    request = {
        "images": {role: image for role in ("left", "front", "right")},
        "command": 2,
        "velocity_xy": (0.0, 0.0),
        "acceleration_xy": (0.0, 0.0),
    }

    def predict(batch_size: int):
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=args.fp16_autocast
        ):
            return policy.predict_batch([request] * batch_size)

    print(
        f"parameters={sum(parameter.numel() for parameter in policy.model.parameters())}"
    )
    for batch_size in args.batch_sizes:
        for _ in range(args.warmup):
            predict(batch_size)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        durations = []
        for _ in range(args.iterations):
            started_at = time.perf_counter()
            predict(batch_size)
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - started_at)
        print(
            f"batch_size={batch_size} mean_s={statistics.mean(durations):.6f} "
            f"p50_s={float(np.quantile(durations, 0.50)):.6f} "
            f"p95_s={float(np.quantile(durations, 0.95)):.6f} "
            f"p99_s={float(np.quantile(durations, 0.99)):.6f} "
            f"pmax_s={max(durations):.6f} "
            f"throughput_fps={batch_size / statistics.mean(durations):.3f} "
            f"peak_allocated_gib={torch.cuda.max_memory_allocated() / 2**30:.3f} "
            f"peak_reserved_gib={torch.cuda.max_memory_reserved() / 2**30:.3f}"
        )
        if args.profile:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            ) as profile:
                predict(batch_size)
                torch.cuda.synchronize()
            print(
                profile.key_averages().table(
                    sort_by="self_cuda_time_total", row_limit=20
                )
            )


if __name__ == "__main__":
    main()
