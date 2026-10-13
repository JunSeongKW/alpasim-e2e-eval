"""Register the built TensorRT image as run_pkg3cam model vits512_g112_kv28_trt.

    python3 e2e_challenge/g112_trt/register_trt_model.py [IMAGE]

Writes runs/prepare-stage3-vits512-g112-kv28-trt-20261013/model.json: the g112
bundle's model.json with the image ID as base_image and "baked": true, so every
runner (run_pkg3cam, run_ec2_timing, run_bench_models) starts the image exactly
as it would be submitted -- nothing mounted, no DRIVESUPRIM_* passed.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "runs/prepare-stage3-vits512-g112-kv28-20261010/model.json"
DST = ROOT / "runs/prepare-stage3-vits512-g112-kv28-trt-20261013/model.json"


def main():
    image = sys.argv[1] if len(sys.argv) > 1 else "alpasim-e2e-drivesuprim-stage3:g112-kv28-trt"
    image_id = subprocess.check_output(["docker", "image", "inspect", image, "--format", "{{.Id}}"],
                                       text=True).strip()
    m = json.loads(SRC.read_text())
    m.update({"model": "vits512_g112_kv28_trt", "subject": "stage3-vits512-g112-kv28-trt",
              "run_name": "leaderboard-stage3-vits512-g112-kv28-trt-20261013",
              "base_image": image_id, "image_tag": image, "baked": True,
              "port": 9300, "wizard_port": 34000})
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(m, indent=2) + "\n")
    print(f"{DST}: {image_id} ({image})")


if __name__ == "__main__":
    main()
