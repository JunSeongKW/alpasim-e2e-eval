"""Check the handoff bundle before trusting anything it produces.

Builds the plan-only agent out of this tree alone -- no dataset, no checkpoint,
no GPU -- and prints the constants the handoff document quotes. If this passes,
the code and configs arrived intact and the environment can run them; what is
left to go wrong is the data, which is what section 6 of the README covers.

    python scripts/nurec/verify_bundle.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# Hydra resolves these out of the config even for a build-only run, so give them
# somewhere harmless to point rather than requiring a prepared dataset.
os.environ.setdefault("NAVSIM_EXP_ROOT", str(ROOT / "exp"))
os.environ.setdefault("OPENSCENE_DATA_ROOT", str(ROOT / "data"))
os.environ.setdefault("NUPLAN_MAPS_ROOT", str(ROOT / "data" / "maps"))
os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")

EXPECTED = {
    "pdm_heads": ["no_at_fault_collisions", "drivable_area_compliance",
                  "gt_compliance", "ego_progress"],
    "bev_h": 56, "bev_w": 56, "bev_num_cameras": 3, "bev_seq_len": 3,
    "bev_img_width": 512, "bev_img_height": 256,
}
AGENT = "drivesuprim_agent_bevformer_vov_v2_vits_planonly_scratch_nurec"
VOCAB = ROOT / "assets/nurec/vocab/nurec_train_kmeans_4096x40x3.npy"


def main() -> int:
    fail = []

    import navsim
    here = str(Path(navsim.__file__).resolve())
    if not here.startswith(str(ROOT)):
        fail.append(f"navsim resolved to {here}, not this bundle -- "
                    "another copy is ahead on PYTHONPATH")
    print(f"navsim      {here}")

    import numpy as np
    if not VOCAB.exists():
        print(f"FAIL        vocabulary missing: {VOCAB}")
        return 1
    shape = np.load(VOCAB, mmap_mode="r").shape
    print(f"vocabulary  {shape}  {VOCAB.name}")
    if shape[1:] != (40, 3):
        fail.append(f"vocabulary shape {shape}, expected (K, 40, 3)")

    from navsim.planning.data.nurec_rectify.target import TARGET_K, TARGET_RESOLUTION_WH
    print(f"rectify     fx={TARGET_K[0, 0]:.0f} fy={TARGET_K[1, 1]:.0f} "
          f"cx={TARGET_K[0, 2]:.0f} cy={TARGET_K[1, 2]:.0f}  @ {TARGET_RESOLUTION_WH}")
    w, h = TARGET_RESOLUTION_WH
    if not (0 < TARGET_K[0, 2] < w and 0 < TARGET_K[1, 2] < h):
        fail.append("the rectification target's own principal point is outside its image")

    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    with initialize_config_dir(
            config_dir=str(ROOT / "navsim/planning/script/config/pdm_scoring"),
            version_base=None):
        cfg = compose(config_name="default_run_pdm_score_gpu_ssl", overrides=[
            f"agent={AGENT}", "++agent.config.training=false",
            "++agent.config.only_ori_input=true",
            "++agent.config.bevformer_vit_pretrained=false",
            "train_test_split=nurec", "++agent.config.use_route=true",
            f"++agent.config.vocab_path={VOCAB}", "++agent.config.vocab_size=4096",
            "++agent.config.ori_vocab_pdm_score_full_path=/dev/null",
            "++agent.config.ori_vocab_pdm_score_dir=/dev/null"])
    agent = instantiate(cfg.agent)
    model = agent.model.student.model
    params = sum(p.numel() for p in model.parameters())
    c = agent._config
    print(f"model       {params / 1e6:.1f}M parameters")
    print(f"geometry    bev {c.bev_h}x{c.bev_w}   cameras {c.bev_num_cameras}   "
          f"frames {c.bev_seq_len}   image {c.bev_img_width}x{c.bev_img_height}")
    print(f"heads       {list(c.pdm_heads)}")

    for key, want in EXPECTED.items():
        got = list(getattr(c, key)) if key == "pdm_heads" else getattr(c, key)
        if got != want:
            fail.append(f"{key} is {got}, expected {want}")

    from navsim.agents.drivesuprim.drivesuprim_features import CalibrationMismatch  # noqa: F401
    from navsim.planning.simulation.planner.nurec_controller.nurec_simulator import (  # noqa: F401
        NuRecSimulator)
    print("imports     CalibrationMismatch guard, NuRecSimulator (MPC)")

    if fail:
        print("\nFAILED")
        for f in fail:
            print(f"  - {f}")
        return 1
    print("\nOK -- the bundle is intact. Next: section 6 of README.md, which "
          "checks the data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
