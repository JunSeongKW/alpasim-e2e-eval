#!/usr/bin/env python3
"""Compile only DriveSuprim's convolutional VoVNet backbone to TensorRT."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch_tensorrt

from drivesuprim_challenge.policy import DriveSuprimPolicy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", default="/app/assets/drivesuprim")
    parser.add_argument("--output", type=Path, default=Path("/output/drivesuprim_vov_backbone_fp16.ts"))
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args()

    policy = DriveSuprimPolicy(
        checkpoint_path=f"{args.asset_dir}/drivesuprim_vov.ckpt",
        backbone_path=f"{args.asset_dir}/dd3d_det_final.pth",
        vocab_path=f"{args.asset_dir}/test_8192_kmeans.npy",
        backbone_type="vov",
        device="cuda",
    )
    backbone = policy.model._backbone.eval()
    camera = torch.zeros(
        (1, 3, policy.camera_height, policy.camera_width),
        device="cuda",
        dtype=torch.float32,
    )
    with torch.inference_mode():
        traced = torch.jit.trace(backbone, camera, strict=False)
        eager_feature = backbone(camera)
        precision = torch.float16 if args.precision == "fp16" else torch.float32
        compiled = torch_tensorrt.compile(
            traced,
            ir="ts",
            inputs=[torch_tensorrt.Input(camera.shape, dtype=torch.float32)],
            enabled_precisions={precision},
            truncate_long_and_double=True,
            require_full_compilation=False,
            min_block_size=3,
            workspace_size=4 << 30,
        )
        trt_feature = compiled(camera)
    error = (eager_feature - trt_feature).abs()
    print(
        f"backbone_mae={error.mean().item():.9f} "
        f"backbone_max_error={error.max().item():.9f}"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(compiled, str(args.output))
    print(f"engine={args.output}")


if __name__ == "__main__":
    main()
