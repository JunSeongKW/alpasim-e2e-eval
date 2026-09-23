#!/usr/bin/env python3
"""Trace DriveSuprim and compile its supported regions with TensorRT FP16."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch_tensorrt

from drivesuprim_challenge.policy import DriveSuprimPolicy


class TensorRTWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self, camera_feature: torch.Tensor, status_feature: torch.Tensor
    ) -> torch.Tensor:
        return self.model(
            {
                "camera_feature": camera_feature,
                "status_feature": [status_feature],
            },
            tokens=["alpasim-online"],
        )["final_traj"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-dir", default="/app/assets/drivesuprim")
    parser.add_argument("--output", type=Path, default=Path("/output/drivesuprim_vov_fp16.ts"))
    parser.add_argument("--trace-only", action="store_true")
    args = parser.parse_args()

    policy = DriveSuprimPolicy(
        checkpoint_path=f"{args.asset_dir}/drivesuprim_vov.ckpt",
        backbone_path=f"{args.asset_dir}/dd3d_det_final.pth",
        vocab_path=f"{args.asset_dir}/test_8192_kmeans.npy",
        backbone_type="vov",
        device="cuda",
    )
    wrapper = TensorRTWrapper(policy.model).eval()
    camera = torch.zeros(
        (1, 3, policy.camera_height, policy.camera_width),
        device="cuda",
        dtype=torch.float32,
    )
    status = torch.zeros((1, 8), device="cuda", dtype=torch.float32)
    with torch.inference_mode():
        traced = torch.jit.trace(wrapper, (camera, status), strict=False)
        eager_output = wrapper(camera, status)
        traced_output = traced(camera, status)
    trace_error = (eager_output - traced_output).abs()
    print(
        f"trace_mae={trace_error.mean().item():.9f} "
        f"trace_max_error={trace_error.max().item():.9f}"
    )
    if args.trace_only:
        return

    compiled = torch_tensorrt.compile(
        traced,
        ir="ts",
        inputs=[
            torch_tensorrt.Input(camera.shape, dtype=torch.float32),
            torch_tensorrt.Input(status.shape, dtype=torch.float32),
        ],
        enabled_precisions={torch.float16},
        truncate_long_and_double=True,
        require_full_compilation=False,
        min_block_size=3,
        workspace_size=4 << 30,
    )
    with torch.inference_mode():
        trt_output = compiled(camera, status)
    trt_error = (eager_output - trt_output).abs()
    print(
        f"tensorrt_mae={trt_error.mean().item():.9f} "
        f"tensorrt_max_error={trt_error.max().item():.9f}"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(compiled, str(args.output))
    print(f"engine={args.output}")


if __name__ == "__main__":
    main()
