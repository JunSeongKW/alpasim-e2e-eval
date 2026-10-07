# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""Run the unchanged AXE-v9 policy and a known-value MSDA CUDA check."""

import importlib.util
import json
import os
from pathlib import Path

import torch


def main():
    spec = importlib.util.spec_from_file_location("bench", "/checks/bench_trt.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    torch.set_num_threads(1)
    policy = bench.build_policy()
    config = policy._config
    assert config.bev_num_cameras == 3 and config.vocab_size == 4096
    assert (config.bev_h, config.bev_w) == (56, 56)
    assert config.use_route and config.inference.model == "teacher"
    assert not policy._use_autocast
    assert os.environ["DRIVESUPRIM_FORCE_PYTORCH_MSDA"] == "1"
    from navsim.agents.backbones.bevformer import spatial_cross_attention as sca

    assert sca._FORCE_PYTORCH_MSDA, "AXE-v9 must use its baked PyTorch fallback"
    value = torch.ones(1, 4, 1, 2, device="cuda")
    shapes = torch.tensor([[2, 2]], dtype=torch.long, device="cuda")
    locations = torch.full((1, 1, 1, 1, 1, 2), 0.5, device="cuda")
    weights = torch.ones(1, 1, 1, 1, 1, device="cuda")
    sample = sca._deform_attn(
        value,
        shapes,
        torch.zeros(1, dtype=torch.long, device="cuda"),
        locations,
        weights,
    )
    torch.testing.assert_close(sample, torch.ones_like(sample))
    prediction = policy._run_agent(bench.make_features(torch.device("cuda")))
    trajectory = bench.trajectory_of(prediction)
    assert trajectory.shape in ((1, 40, 3), (40, 3)), trajectory.shape
    for key, tensor in prediction.items():
        if isinstance(tensor, torch.Tensor):
            assert torch.isfinite(tensor).all(), key
    report = {
        "passed": True,
        "strict_checkpoint": True,
        "camera_count": config.bev_num_cameras,
        "bev_size": [config.bev_h, config.bev_w],
        "vocab_size": config.vocab_size,
        "device": torch.cuda.get_device_name(0),
        "msda_constant_sample": sample.detach().cpu().tolist(),
        "trajectory_shape": list(trajectory.shape),
        "trajectory": trajectory.tolist(),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }
    Path("/report/gpu_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("PASS: strict checkpoint, known-value CUDA operator, actual 3-camera forward")


if __name__ == "__main__":
    main()
