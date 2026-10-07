# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""Compose a bundle's stage-3 agent YAML inside the pinned inference image.

Same rule as 5cam_eval/compose_config.py: the model's own values (gate
thresholds, BEV size, backbone, image size) are kept as trained. The overrides
only switch to inference, stop pretrained downloads, point at the deployed
checkpoint/vocab, and enable the route input the checkpoint was trained with.
"""

import argparse
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--agent", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
config_dir = args.source / "navsim/planning/script/config/common/agent"
with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
    cfg = compose(config_name=args.agent)
values = OmegaConf.to_container(cfg.config, resolve=False)
values.update(
    training=False,
    only_ori_input=True,
    bevformer_vit_pretrained=False,
    bevformer_cnn_pretrained=False,
    ckpt_path=args.checkpoint,
    vocab_path="/app/assets/drivesuprim/nurec_train_kmeans_4096x40x3.npy",
    use_route=True,
)
# The ViT-L configs name a pretrained file (${NAVSIM_EXP_ROOT}/models/
# da_vitl16.pth) that DinoV2ImageBackbone loads whenever the value is
# non-empty. At inference every weight comes from the strictly loaded stage-3
# checkpoint, so clear it -- the same intent as bevformer_vit_pretrained=False.
# The ViT-S bundles already carry "" here.
if values.get("bevformer_vit_ckpt"):
    values["bevformer_vit_ckpt"] = ""
args.output.write_text(json.dumps(values, indent=2) + "\n")
print("Composed", args.agent, "->", args.output)
