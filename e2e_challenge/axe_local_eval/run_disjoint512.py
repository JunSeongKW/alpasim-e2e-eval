# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""Evaluate the supplied disjoint 3-camera weights in the exact AXE-v9 runtime.

The user authorizes GPU 0–7 and 48 workers for throughput. Each inference is
still batch one; scoring, MPC and inference code are inherited unchanged.
"""

import argparse
import concurrent.futures
import fcntl
import json
import os
import shutil
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN_NAME = "leaderboard-vits512-disjoint-baseline-ep04-20261007"
RUN = ROOT / "runs" / RUN_NAME
PREP = ROOT / "runs/prepare-vits512-disjoint-baseline-ep04-20261007"
IMAGE = "alpasim-e2e-drivesuprim-stage3:vits512-disjoint-baseline-ep04-20261007"
BASE_ID = "sha256:85134c1ea9f09d140be610ca063c50ec60e99980c392e5bf81fe6563af503765"
SHA = "e8f87989206f15586218f70b2a6977ec83fe6e82ce983f144b66a6605650f9f1"
PREFIX = "axe-disjoint512-ep04-drv"
GPUS = list(range(8))
REPLICAS = 6
WORKERS = 48
PORT = 7500
WIZARD_PORT = 23000
KST = ZoneInfo("Asia/Seoul")


def log(message):
    print(f"[{datetime.now(KST):%F %T}] {message}", flush=True)


def output(command):
    return subprocess.check_output(command, text=True).strip()


def owned_containers():
    names = output(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"label=com.docker.compose.project={RUN_NAME}",
            "--format",
            "{{.Names}}",
        ]
    ).splitlines()
    names += output(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"name=^{PREFIX}-g[0-7]-[0-5]$",
            "--format",
            "{{.Names}}",
        ]
    ).splitlines()
    return sorted(set(names))


def environment(run=RUN, *, dry=False):
    env = {
        **os.environ,
        "UV_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "RUN_NAME": RUN_NAME,
        "RUN_DIR": str(run),
        "PRESET": "dev",
        "CONTESTANT_IMAGE": IMAGE,
        "DRIVER_ADDRESSES": "["
        + ",".join(f'"localhost:{p}"' for p in range(PORT, PORT + WORKERS))
        + "]",
        "N_ROLLOUTS": "1",
        "ROLLOUT_WORKERS": str(WORKERS),
        "RENDER_GPUS_CSV": "0,1,2,3,4,5,6,7",
        "RENDERER_REPLICAS_PER_GPU": str(REPLICAS),
        "NRE_CACHE_SIZE": "1",
        "SCENE_LIMIT": "0",
        "SCENE_IDS_FILE": "",
        "RENDER_VIDEO": "false",
        "KEEP_ROLLOUTS": "1",
        "ENABLE_AUTORESUME": "true",
        "SERVICE_STARTUP_TIMEOUT_SEC": "1800",
        "FAST_STARTUP": "1",
        "DRIVER_CONCURRENT_ROLLOUTS": "1",
        "ALPASIM_IMAGE": "nvcr.io/nvidia/nre/nre-ga:26.04",
        "NRE_IMAGE": "nvcr.io/nvidia/nre/nre-ga:26.04",
        "MPC_OVERRIDES": "controller.gains.long_position_weight=0.25 controller.gains.lat_position_weight=1.0 controller.gains.idx_start_penalty=3",
        "EXTRA_OVERRIDES": f"wizard.baseport={WIZARD_PORT} runtime.simulation_config.render_bundling=NONE"
        + (" wizard.dry_run=true" if dry else ""),
    }
    return env


def check():
    base, image = json.loads(output(["docker", "image", "inspect", BASE_ID, IMAGE]))
    assert base["Id"] == BASE_ID
    assert image["Id"] == (PREP / "image-id.txt").read_text().strip()
    assert image["Config"]["Labels"]["org.alpasim.checkpoint.sha256"] == SHA
    layers = base["RootFS"]["Layers"]
    assert image["RootFS"]["Layers"] == layers + [image["RootFS"]["Layers"][-1]]
    for key in ("Env", "Entrypoint", "Cmd", "WorkingDir", "User"):
        assert image["Config"].get(key) == base["Config"].get(key), key
    assert json.loads((PREP / "gpu_validation.json").read_text())["passed"]
    assert json.loads((PREP / "deployment_validation.json").read_text())["passed"]
    assert shutil.disk_usage(ROOT).free >= 250 * 1024**3
    assert not owned_containers(), "Existing containers: refusing a duplicate launch"
    assert not (RUN / "wizard-config.yaml").exists(), "Run already deployed"
    subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(HERE / "disjoint512_leaderboard.py"),
            "--check-existing",
        ],
        check=True,
    )
    stats = output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]
    )
    assert len(stats.splitlines()) == 8
    assert all(int(row.split(",")[1]) < 4000 for row in stats.splitlines()), stats
    for port in [*range(PORT, PORT + WORKERS), *range(WIZARD_PORT, WIZARD_PORT + 130)]:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    log(
        "Validated weights-only overlay, CUDA forward, 441 scenes and all eight free GPUs"
    )
    return image["Id"]


def start_gpu(gpu):
    for replica in range(REPLICAS):
        index = gpu * REPLICAS + replica
        subprocess.run(
            [
                "docker",
                "run",
                "-d",
                "--name",
                f"{PREFIX}-g{gpu}-{replica}",
                "--init",
                "--gpus",
                f"device={gpu}",
                "--cap-drop",
                "ALL",
                "--security-opt",
                "no-new-privileges:true",
                "--read-only",
                "--pids-limit",
                "1024",
                "--memory",
                "32g",
                "--cpus",
                "8",
                "--tmpfs",
                "/tmp:rw,nosuid,nodev,size=2g",
                "--tmpfs",
                "/run:rw,nosuid,nodev,size=64m",
                "-p",
                f"127.0.0.1:{PORT + index}:6789",
                "-e",
                "ALPASIM_DRIVER_HOST=0.0.0.0",
                "-e",
                "ALPASIM_DRIVER_PORT=6789",
                "-e",
                f"ALPASIM_CONTESTANT_REPLICA_INDEX={index}",
                "-e",
                f"ALPASIM_CONTESTANT_REPLICAS={WORKERS}",
                IMAGE,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    log(f"Started {REPLICAS} driver replicas on GPU {gpu}")


def stop_owned():
    names = owned_containers()
    if not names:
        return
    log("Stopping only this evaluation's containers: " + ", ".join(names))
    archive = RUN / "driver-logs"
    archive.mkdir(exist_ok=True)
    for name in names:
        if name.startswith(PREFIX):
            with (archive / f"{name}.log").open("w") as stream:
                subprocess.run(
                    ["docker", "logs", name],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
    subprocess.run(["docker", "stop", "-t", "10", *names], check=True)
    subprocess.run(["docker", "rm", *names], check=True)


def status(stage, **fields):
    (RUN / "status.json").write_text(
        json.dumps(
            {
                "stage": stage,
                "updated_at": datetime.now(KST).isoformat(),
                "launcher_pid": os.getpid(),
                **fields,
            },
            indent=2,
        )
        + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.dry_run:
        subprocess.run(
            ["bash", str(HERE / "run_curated_val.sh")],
            env=environment(PREP / "deployment-dryrun", dry=True),
            check=True,
        )
        return
    image_id = check()
    if args.check:
        return
    lock = (ROOT / f".cache/{RUN_NAME}.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    RUN.mkdir(parents=True, exist_ok=True)
    (ROOT / f".cache/{RUN_NAME}.launcher.pid").write_text(str(os.getpid()) + "\n")
    (RUN / "evaluation_provenance.json").write_text(
        json.dumps(
            {
                "subject": "vits512-disjoint-baseline-ep04",
                "image_id": image_id,
                "base_image_id": BASE_ID,
                "checkpoint_sha256": SHA,
                "checkpoint_path": str(
                    ROOT.parent
                    / "models/20261007_vits512_disjoint_baseline_stage3_epoch04-step4075.ckpt"
                ),
                "changed_model_files": ["/app/assets/drivesuprim/axe_nurec_vits.ckpt"],
                "camera_order": ["CAM_L0", "CAM_F0", "CAM_R0"],
                "preset": "dev",
                "scenes": 441,
                "rollouts_per_scene": 1,
                "gpus": GPUS,
                "drivers": WORKERS,
                "workers": WORKERS,
                "renderers": WORKERS,
                "model_batch_size": 1,
                "driver_grpc_workers": 8,
                "render_bundling": "NONE",
                "route_reranker": False,
                "ego_footprint_from_api": True,
                "ego_center_offset": True,
                "mpc": {
                    "lat_position_weight": 1.0,
                    "long_position_weight": 0.25,
                    "idx_start_penalty": 3,
                },
                "throughput_topology_user_authorized": True,
                "keep_rollouts": True,
                "git_commit": output(["git", "rev-parse", "HEAD"]),
                "started_at": datetime.now(KST).isoformat(),
            },
            indent=2,
        )
        + "\n"
    )
    try:
        status("starting_drivers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(start_gpu, GPUS))
        pending = {f"{PREFIX}-g{g}-{r}" for g in GPUS for r in range(REPLICAS)}
        deadline = time.monotonic() + 1800
        while pending:
            for name in sorted(pending):
                assert (
                    output(
                        ["docker", "inspect", name, "--format", "{{.State.Running}}"]
                    )
                    == "true"
                ), name
                logs = subprocess.check_output(
                    ["docker", "logs", name], text=True, stderr=subprocess.STDOUT
                )
                if "policy load complete" in logs:
                    assert "exact checkpoint load: missing=0 unexpected=0" in logs, name
                    pending.remove(name)
                    log(f"Ready: {name}; remaining {len(pending)}")
            assert time.monotonic() < deadline, f"Drivers not ready: {pending}"
            if pending:
                time.sleep(3)
        status("simulation")
        log("Starting 48-worker curated441 evaluation; original gRPC8, batch1, NONE")
        with (ROOT / "runs" / f"{RUN_NAME}.wizard.log").open("a") as stream:
            proc = subprocess.Popen(
                ["bash", str(HERE / "run_curated_val.sh")],
                env=environment(),
                stdout=stream,
                stderr=subprocess.STDOUT,
            )
        peaks = {str(g): 0 for g in GPUS}
        with (RUN / "vram.csv").open("w") as stream:
            stream.write(
                "timestamp,gpu,memory_used_mib,memory_free_mib,utilization_percent\n"
            )
            while proc.poll() is None:
                stats = output(
                    [
                        "nvidia-smi",
                        "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ]
                )
                for row in stats.splitlines():
                    gpu, used, _, _ = (v.strip() for v in row.split(","))
                    peaks[gpu] = max(peaks[gpu], int(used))
                    stream.write(f"{datetime.now(KST).isoformat()},{row}\n")
                stream.flush()
                (RUN / "vram-peaks.json").write_text(json.dumps(peaks, indent=2) + "\n")
                time.sleep(10)
        assert (
            proc.returncode == 0
        ), f"Simulation exit={proc.returncode}; inspect wizard log"
        assert (RUN / "aggregate/results-summary.json").exists()
        stop_owned()
        status("fitting_leaderboard")
        subprocess.run(
            [str(ROOT / ".venv/bin/python"), str(HERE / "disjoint512_leaderboard.py")],
            env={**environment(), "OMP_NUM_THREADS": "12", "MKL_NUM_THREADS": "12"},
            check=True,
        )
        status("complete")
        log(
            "Complete: runs/leaderboard-36-with-vits512-disjoint-ep04/local_leaderboard.csv"
        )
    except Exception as exc:
        status("failed", error=str(exc))
        # Keep failed rollouts and containers available for diagnosis.
        raise


if __name__ == "__main__":
    main()
