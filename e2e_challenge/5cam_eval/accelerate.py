# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""Resume the supplied model evaluation on all eight GPUs with 48 workers.

The user explicitly prioritizes throughput over the historical 16-worker
topology. Model weights, inference code, precision, MPC and scoring stay fixed.
"""

import argparse
import concurrent.futures
import fcntl
import json
import os
import shutil
import signal
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN_NAME = "leaderboard-stage3-5cam-ep05-20261007"
RUN = ROOT / "runs" / RUN_NAME
PREP = ROOT / "runs/prepare-stage3-5cam-ep05"
IMAGE = "alpasim-e2e-drivesuprim-stage3:5cam-ep05-20261007"
PREFIX = "axe-5cam-ep05-drv"
GPUS = list(range(8))
REPLICAS = 6
WORKERS = len(GPUS) * REPLICAS
PORT = 7160
SHA = "79c625f3a29c49da6b9605b1062de5e8e6ec5da8391362b25507e35725bb8b8e"
KST = ZoneInfo("Asia/Seoul")


def log(message):
    print(f"[{datetime.now(KST):%F %T}] {message}", flush=True)


def output(command):
    return subprocess.check_output(command, text=True).strip()


def check():
    image_id = output(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"])
    assert image_id == (PREP / "image-id.txt").read_text().strip()
    p = json.loads((RUN / "evaluation_provenance.json").read_text())
    assert p["image_id"] == image_id and p["checkpoint_sha256"] == SHA
    assert p["preset"] == "dev" and p["scenes"] == 441
    assert p["mpc"] == {
        "lat_position_weight": 1.0,
        "long_position_weight": 0.25,
        "idx_start_penalty": 3,
    }
    assert json.loads((PREP / "gpu_validation.json").read_text())["passed"]
    assert shutil.disk_usage(RUN).free >= 250 * 1024**3, "Need 250 GiB free"
    subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(HERE / "leaderboard.py"),
            "--check-existing",
        ],
        check=True,
    )
    log(f"Validated original image; target GPUs 0–7, {WORKERS} workers")
    return p


def acquire_lock():
    handle = (ROOT / ".cache/stage3-5cam-441.lock").open("w")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle
    except BlockingIOError:
        old_pid = int((ROOT / ".cache/stage3-5cam-441.launcher.pid").read_text())
        command = Path(f"/proc/{old_pid}/cmdline").read_bytes()
        original = b"e2e_challenge/5cam_eval/run.sh" in command
        assert original or b"e2e_challenge/5cam_eval/accelerate.py" in command
        assert os.getpgid(old_pid) == old_pid
        log(f"Stopping previous evaluation launcher process group {old_pid}")
        if original:
            os.killpg(old_pid, signal.SIGTERM)
            time.sleep(3)
        try:
            os.killpg(old_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for _ in range(10):
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return handle
            except BlockingIOError:
                time.sleep(1)
        raise RuntimeError("Original launcher lock did not release") from None


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


def stop_owned(archive, *, keep_renderers=False):
    names = owned_containers()
    if keep_renderers:
        names = [name for name in names if f"{RUN_NAME}-renderer-" not in name]
    if not names:
        return
    log("Stopping only this evaluation's containers: " + ", ".join(names))
    for name in names:
        if "-runtime-" in name:
            paused = output(
                ["docker", "inspect", name, "--format", "{{.State.Paused}}"]
            )
            if paused == "true":
                subprocess.run(
                    ["docker", "unpause", name], check=True, stdout=subprocess.DEVNULL
                )
    archive.mkdir(parents=True, exist_ok=True)
    for name in names:
        if name.startswith(PREFIX):
            with (archive / f"{name}.log").open("w") as stream:
                subprocess.run(
                    ["docker", "logs", "--tail", "300", name],
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
    for command in (["docker", "stop", "-t", "10", *names], ["docker", "rm", *names]):
        result = subprocess.run(command, text=True, capture_output=True, check=False)
        if result.returncode:
            errors = result.stderr.splitlines()
            # The original launcher's EXIT trap may have removed a driver.
            if any("No such container" not in line for line in errors):
                raise RuntimeError(result.stderr)


def preserve_and_record(previous, *, keep_renderers):
    stamp = datetime.now(KST).strftime("%Y%m%d-%H%M%S")
    archive = RUN / "topology-history" / stamp
    archive.mkdir(parents=True)
    for p in RUN.iterdir():
        if p.is_file() and p.suffix in (".json", ".yaml"):
            shutil.copy2(p, archive / p.name)
    stop_owned(archive / "driver-logs", keep_renderers=keep_renderers)
    completed = sorted(
        p.parent.parent.name for p in (RUN / "rollouts").glob("*/*/_complete")
    )
    assert len(completed) == len(set(completed))
    for directory in list((RUN / "rollouts").glob("*/*")):
        if directory.is_dir() and not (directory / "_complete").exists():
            target = (
                archive
                / "interrupted-rollouts"
                / directory.parent.name
                / directory.name
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            directory.rename(target)
    history = previous.get("execution_phases", [])
    history.append(
        {
            "gpus": previous["gpus"],
            "workers": previous["workers"],
            "render_bundling": previous.get("render_bundling", "NONE"),
            "driver_grpc_workers": previous.get("driver_grpc_workers", 8),
            "completed_clip_ids_at_switch": completed,
            "archive": str(archive),
        }
    )
    previous.update(
        gpus=GPUS,
        drivers=WORKERS,
        workers=WORKERS,
        renderers=WORKERS,
        render_bundling="NONE",
        model_batch_size=1,
        driver_grpc_workers=8,
        official_env_only=True,
        deployment_extra_env=[],
        gpu_memory_limit_mib=81559,
        renderer_camera_rpc_batch=1,
        execution_phases=history,
        throughput_priority_user_authorized=True,
        historical_16_worker_topology_match=False,
        acceleration_commit=output(["git", "rev-parse", "HEAD"]),
    )
    (RUN / "evaluation_provenance.json").write_text(
        json.dumps(previous, indent=2) + "\n"
    )
    log(
        f"Preserved {len(completed)} completed clips; interrupted files moved without deletion"
    )


def start_gpu(gpu):
    for replica in range(REPLICAS):
        index = gpu * REPLICAS + replica
        name = f"{PREFIX}-g{gpu}-{replica}"
        command = [
            "docker",
            "run",
            "-d",
            "--name",
            name,
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
        ]
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    log(f"Started {REPLICAS} driver replicas on GPU {gpu}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--keep-renderers", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    previous = check()
    if args.check:
        return
    lock = acquire_lock()
    assert lock is not None
    (ROOT / ".cache/stage3-5cam-441.launcher.pid").write_text(str(os.getpid()) + "\n")
    preserve_and_record(previous, keep_renderers=args.keep_renderers)
    stats = output(
        [
            "nvidia-smi",
            "-i",
            "0,1,2,3,4,5,6,7",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    assert len(stats.splitlines()) == 8
    if args.keep_renderers:
        assert all(int(row.split(",")[0]) < 65000 for row in stats.splitlines()), stats
        log(
            "Reusing warm renderers; original eight driver gRPC workers and single-camera RPCs"
        )
    else:
        assert all(
            int(row.split(",")[0]) < 4000 and int(row.split(",")[1]) <= 5
            for row in stats.splitlines()
        ), stats
    for port in range(PORT, PORT + WORKERS):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    env = {
        **os.environ,
        "UV_OFFLINE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    tag = f"run/{RUN_NAME}-48workers"
    if subprocess.run(
        ["git", "rev-parse", f"refs/tags/{tag}"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode:
        subprocess.run(["git", "tag", tag], check=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(start_gpu, GPUS))
    deadline = time.monotonic() + 1800
    pending = {f"{PREFIX}-g{g}-{r}" for g in GPUS for r in range(REPLICAS)}
    while pending:
        for name in sorted(pending):
            state = output(
                ["docker", "inspect", name, "--format", "{{.State.Running}}"]
            )
            assert state == "true", f"{name} exited"
            logs = subprocess.check_output(
                ["docker", "logs", name], text=True, stderr=subprocess.STDOUT
            )
            if "policy load complete" in logs:
                pending.remove(name)
                log(f"Ready: {name}; remaining {len(pending)}")
        assert time.monotonic() < deadline, f"Drivers not ready: {pending}"
        if pending:
            time.sleep(3)
    addresses = (
        "[" + ",".join(f'"localhost:{p}"' for p in range(PORT, PORT + WORKERS)) + "]"
    )
    env.update(
        RUN_NAME=RUN_NAME,
        RUN_DIR=str(RUN),
        PRESET="dev",
        CONTESTANT_IMAGE=IMAGE,
        DRIVER_ADDRESSES=addresses,
        N_ROLLOUTS="1",
        ROLLOUT_WORKERS=str(WORKERS),
        RENDER_GPUS_CSV="0,1,2,3,4,5,6,7",
        RENDERER_REPLICAS_PER_GPU=str(REPLICAS),
        NRE_CACHE_SIZE="1",
        SCENE_LIMIT="0",
        SCENE_IDS_FILE="",
        RENDER_VIDEO="false",
        KEEP_ROLLOUTS="1",
        ENABLE_AUTORESUME="true",
        SERVICE_STARTUP_TIMEOUT_SEC="1800",
        FAST_STARTUP="1",
        DRIVER_CONCURRENT_ROLLOUTS="1",
        ALPASIM_IMAGE="nvcr.io/nvidia/nre/nre-ga:26.04",
        NRE_IMAGE="nvcr.io/nvidia/nre/nre-ga:26.04",
        MPC_OVERRIDES="controller.gains.long_position_weight=0.25 controller.gains.lat_position_weight=1.0 controller.gains.idx_start_penalty=3",
        EXTRA_OVERRIDES="wizard.baseport=19900 runtime.simulation_config.render_bundling=NONE",
    )
    log(
        "Starting 48-worker resume with original eight gRPC workers and single-camera RGB requests"
    )
    wizard_log = ROOT / "runs" / f"{RUN_NAME}.speed48.wizard.log"
    with wizard_log.open("a") as stream:
        proc = subprocess.Popen(
            ["bash", str(ROOT / "e2e_challenge/axe_local_eval/run_curated_val.sh")],
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
        )
    peaks = {str(g): 0 for g in GPUS}
    with (RUN / "vram-speed48.csv").open("a") as stream:
        stream.write(
            "timestamp,gpu,memory_used_mib,memory_free_mib,utilization_percent\n"
        )
        while proc.poll() is None:
            stats = output(
                [
                    "nvidia-smi",
                    "-i",
                    "0,1,2,3,4,5,6,7",
                    "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ]
            )
            for row in stats.splitlines():
                gpu, used, _, _ = (v.strip() for v in row.split(","))
                peaks[gpu] = max(peaks[gpu], int(used))
                stream.write(f"{datetime.now(KST).isoformat()},{row}\n")
            stream.flush()
            (RUN / "vram-speed48-peaks.json").write_text(
                json.dumps(peaks, indent=2) + "\n"
            )
            time.sleep(10)
    assert proc.returncode == 0, f"Simulation exit={proc.returncode}; see {wizard_log}"
    log(
        "Simulation complete; releasing evaluation containers before joint CPU leaderboard fit"
    )
    stop_owned(RUN / "driver-logs-speed48")
    fit_env = {**env, "OMP_NUM_THREADS": "12", "MKL_NUM_THREADS": "12"}
    subprocess.run(
        [
            str(ROOT / ".venv/bin/python"),
            str(HERE / "leaderboard.py"),
            "--run-dir",
            str(RUN),
        ],
        env=fit_env,
        check=True,
    )
    log("Complete: runs/leaderboard-35-with-stage3-5cam-ep05/local_leaderboard.csv")


if __name__ == "__main__":
    main()
