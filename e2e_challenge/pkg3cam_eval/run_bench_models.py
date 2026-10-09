# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""Measure the model latency of axe-v9 and the three 10-08 bundles side by side.

    .venv/bin/python e2e_challenge/pkg3cam_eval/run_bench_models.py [--gpus 4,5,6,7]

One container per model, each alone on its own GPU (default 4-7), all started
together so they share the same host conditions. axe-v9 runs its immutable
image as evaluated; the bundles run that image with the evaluation's read-only
mounts and per-model environment (run_pkg3cam.Model), so each times the code
its 441 drivers ran. Results: runs/latency-bench-<stamp>/<model>.json.
"""

import argparse
import concurrent.futures
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from run_pkg3cam import Model  # noqa: E402

BASE = ("696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe@sha256:"
        "85134c1ea9f09d140be610ca063c50ec60e99980c392e5bf81fe6563af503765")
MODELS = ["axe-v9", "vits512_6layer", "vits768_3layer", "vitl512_3layer_lr1"]


def command(name, gpu, out):
    mounts, image = [], BASE
    if name != "axe-v9":
        model = Model(name)
        mounts = model.mounts()
        image = model.meta["base_image"]
    return [
        "docker", "run", "--rm", "--name", f"axe-latency-bench-{name.replace('_', '')}",
        "--gpus", f"device={gpu}", "--cap-drop", "ALL", "--security-opt", "no-new-privileges:true",
        "--read-only", "--memory", "32g", "--cpus", "8",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=2g", "-e", "HOME=/tmp",
        *mounts,
        "-v", f"{ROOT}/e2e_challenge/axe_local_eval/trt_bench:/checks:ro",
        "-v", f"{HERE}/bench_models.py:/benchm/bench_models.py:ro",
        "-v", f"{out}:/out",
        "--entrypoint", "python", image, "/benchm/bench_models.py",
        "--name", name, "--out", f"/out/{name}.json",
        *([] if name == "axe-v9" else ["--ego-geom"]),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gpus", default="4,5,6,7")
    a = ap.parse_args()
    gpus = a.gpus.split(",")
    out = ROOT / "runs" / f"latency-bench-{datetime.now():%Y%m%d_%H%M}"
    out.mkdir(parents=True, exist_ok=True)
    out.chmod(0o777)

    def run(pair):
        name, gpu = pair
        with (out / f"{name}.log").open("w") as stream:
            rc = subprocess.run(command(name, gpu, out), stdout=stream,
                                stderr=subprocess.STDOUT).returncode
        return name, gpu, rc

    with concurrent.futures.ThreadPoolExecutor(len(MODELS)) as pool:
        for name, gpu, rc in pool.map(run, zip(MODELS, gpus)):
            print(f"{name} on GPU {gpu}: exit {rc}", flush=True)
    rows = [json.loads((out / f"{m}.json").read_text()) for m in MODELS if (out / f"{m}.json").exists()]
    (out / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("results:", out)


if __name__ == "__main__":
    main()
