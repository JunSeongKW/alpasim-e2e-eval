#!/usr/bin/env python3
"""Compose the calibration-fixed NuRec PDMS ViT-S deployment config."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


CONFIG_NAME = "drivesuprim_agent_bevformer_vov_v2_vits_planonly_scratch_nurec"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config_dir = (
        args.package_root / "navsim/planning/script/config/common/agent"
    ).resolve()
    if not (config_dir / f"{CONFIG_NAME}.yaml").is_file():
        raise FileNotFoundError(f"missing calibration-fixed config: {config_dir}")

    os.environ.setdefault("NAVSIM_DEVKIT_ROOT", "/app")
    os.environ.setdefault("NAVSIM_EXP_ROOT", "/app")
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        composed = compose(config_name=CONFIG_NAME)

    config = OmegaConf.to_container(composed.config, resolve=False)
    config.update(
        {
            "training": False,
            "only_ori_input": True,
            "use_route": True,
            "route_norm_m": 80.0,
            "route_hidden_dim": 64,
            "route_separate_attention": False,
            "route_gate_init": 0.0,
            "use_traffic_light_compliance": False,
            "vocab_size": 4096,
            "vocab_path": "/app/assets/drivesuprim/nurec_train_kmeans_4096x40x3.npy",
            "bevformer_vit_pretrained": False,
            "bevformer_vit_ckpt": "",
            "bev_use_grad_checkpoint": False,
        }
    )

    required = {
        "backbone_type": "bevformer_m",
        "bevformer_img_backbone_type": "vits",
        "bevformer_vit_name": "vit_small_patch16_dinov3",
        "bev_img_width": 512,
        "bev_img_height": 256,
        "bev_seq_len": 3,
        "bev_num_cameras": 3,
        "vocab_size": 4096,
        "use_route": True,
        "only_ori_input": True,
        "use_traffic_light_compliance": False,
        "use_aux_heads": False,
        "feasibility_enabled": False,
        "pdm_heads": [
            "no_at_fault_collisions",
            "drivable_area_compliance",
            "gt_compliance",
            "ego_progress",
        ],
        "pdm_imi_rank_weight": 0.3,
        "pdm_score_log": {
            "no_at_fault_collisions": 8.0,
            "drivable_area_compliance": 8.0,
            "gt_compliance": 8.0,
        },
        "pdm_score_sum": {"ego_progress": 1.0},
        "pdm_aggregate_head": True,
        "pdm_aggregate_rank_weight": 1.0,
    }
    mismatches = {
        key: (config.get(key), expected)
        for key, expected in required.items()
        if config.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"calibration-fixed config mismatch: {mismatches}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote calibration-fixed deployment config: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
