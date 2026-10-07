# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""Compose the bundle's training YAML inside the pinned inference image."""

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

AGENT = "drivesuprim_agent_bevformer_vov_v2_vits_stage3_nurec_5cam_rear56"

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
config_dir = args.source / "navsim/planning/script/config/common/agent"
with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
    cfg = compose(config_name=AGENT)
values = OmegaConf.to_container(cfg.config, resolve=False)
values.update(
    training=False,
    only_ori_input=True,
    bevformer_vit_pretrained=False,
    bevformer_cnn_pretrained=False,
    ckpt_path="/app/assets/drivesuprim/stage3_5cam_ep05.ckpt",
    vocab_path="/app/assets/drivesuprim/nurec_train_kmeans_4096x40x3.npy",
    use_route=True,
    n_camera=5,
)
args.output.write_text(json.dumps(values, indent=2) + "\n")
print("Composed", AGENT, "->", args.output)
