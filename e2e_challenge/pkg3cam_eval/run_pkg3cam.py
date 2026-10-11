# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""441 evaluation of a three-camera AXEv1.0-NuRec bundle on GPU 0-7, 48 workers.

    .venv/bin/python e2e_challenge/pkg3cam_eval/run_pkg3cam.py --model vits512_6layer --smoke 4
    .venv/bin/python e2e_challenge/pkg3cam_eval/run_pkg3cam.py --model vits512_6layer [--wait]

Same runtime as run_disjoint512.py (2026-10-07): the immutable axe-v9 image,
48 driver replicas (6 per GPU) and 48 renderers, dev preset, 1 rollout per
clip, MPC lat/lon/idx = 1/0.25/3, batch 1, no reranker, rollouts kept. The
user authorizes GPU 0-7 and the 48-worker topology for throughput.

What differs from axe-v9 is only what the bundle carries: its navsim source,
its stage-3 config and checkpoint (bind-mounted read-only, see prepare.sh) and,
because that source takes the ego box as a per-session input, the 5cam_eval
adapter changes run with a 3-camera layout.

--wait queues the run until all eight GPUs are back under 4 GB used (another
researcher keeps ~3 GB on each), so the next model can start the moment the
previous one releases them.
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
AXE = ROOT / "e2e_challenge/axe_local_eval"
GPUS = list(range(8))
REPLICAS = 6
WORKERS = 48
KST = ZoneInfo("Asia/Seoul")
# model -> (subject, bundle date); the date names the prepare and run directories
MODELS = {
    "vits512_6layer": ("stage3-vits512-6layer", "20261008"),
    "vits768_3layer": ("stage3-vits768-3layer", "20261008"),
    "vitl512_3layer_lr1": ("stage3-vitl512-3layer-lr1", "20261008"),
    "vits512_g112_kv28": ("stage3-vits512-g112-kv28", "20261010"),
    "vits512_vidarrender": ("stage3-vits512-vidarrender", "20261010"),
}


def log(message):
    print(f"[{datetime.now(KST):%F %T}] {message}", flush=True)


def output(command):
    return subprocess.check_output(command, text=True).strip()


class Model:
    def __init__(self, key):
        self.prep = ROOT / "runs" / "prepare-{}-{}".format(*MODELS[key])
        self.meta = json.loads((self.prep / "model.json").read_text())
        self.run_name = self.meta["run_name"]
        self.run = ROOT / "runs" / self.run_name
        self.prefix = f"axe-pkg3-{key.replace('_', '')}-drv"
        self.port = int(self.meta["port"])
        self.wizard_port = int(self.meta["wizard_port"])

    def mounts(self):
        src = self.meta["source"]
        return [
            "-v", f"{src}/navsim:/app/navsim:ro",
            "-v", f"{src}/nurec_pf:/app/nurec_pf:ro",
            "-v", f"{self.prep}/drivesuprim_challenge:/app/drivesuprim_challenge:ro",
            "-v", f"{self.meta['checkpoint']}:/app/assets/drivesuprim/pkg_stage3_best.ckpt:ro",
            "-v", f"{self.prep}/assets/config.json:/app/assets/drivesuprim/pkg_config.json:ro",
            "-e", "DRIVESUPRIM_CHECKPOINT_PATH=/app/assets/drivesuprim/pkg_stage3_best.ckpt",
            "-e", "DRIVESUPRIM_CONFIG_PATH=/app/assets/drivesuprim/pkg_config.json",
            "-e", f"DRIVESUPRIM_PINHOLE_PROFILE={self.meta['pinhole_profile']}",
            "-e", "HF_HUB_OFFLINE=1",
            "-e", "TRANSFORMERS_OFFLINE=1",
            "-e", "PYTHONDONTWRITEBYTECODE=1",
            # Per-model switches recorded in model.json. vitl512 needs
            # XFORMERS_DISABLED=1: the image's xformers has no sm90 build of the
            # fmha kernel its DINOv2 attention calls, and on H100 that kernel
            # prints FATAL and returns garbage instead of raising, so the
            # bundle's own torch fallback (MemEffAttention -> Attention) never
            # triggers. The env var is the bundle's switch for that fallback.
            *[arg for k, v in self.meta.get("extra_env", {}).items() for arg in ("-e", f"{k}={v}")],
        ]

    def owned_containers(self):
        names = output(["docker", "ps", "-a", "--filter",
                        f"label=com.docker.compose.project={self.run_name}",
                        "--format", "{{.Names}}"]).splitlines()
        names += output(["docker", "ps", "-a", "--filter",
                         f"name=^{self.prefix}-g[0-7]-[0-5]$",
                         "--format", "{{.Names}}"]).splitlines()
        return sorted(set(n for n in names if n))

    def environment(self):
        return {
            **os.environ,
            "UV_OFFLINE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "RUN_NAME": self.run_name,
            "RUN_DIR": str(self.run),
            "PRESET": "dev",
            "CONTESTANT_IMAGE": self.meta["base_image"],
            "DRIVER_ADDRESSES": "["
            + ",".join(f'"localhost:{p}"' for p in range(self.port, self.port + WORKERS))
            + "]",
            "N_ROLLOUTS": "1",
            "ROLLOUT_WORKERS": str(WORKERS),
            "RENDER_GPUS_CSV": ",".join(map(str, GPUS)),
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
            "MPC_OVERRIDES": "controller.gains.long_position_weight=0.25 "
            "controller.gains.lat_position_weight=1.0 controller.gains.idx_start_penalty=3",
            "EXTRA_OVERRIDES": f"wizard.baseport={self.wizard_port} "
            "runtime.simulation_config.render_bundling=NONE",
        }

    def docker_driver(self, name, gpu, port, index, detach=True, extra=()):
        return [
            "docker", "run", *(["-d"] if detach else ["--rm"]), "--name", name, "--init",
            "--gpus", f"device={gpu}", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges:true", "--read-only",
            "--pids-limit", "1024", "--memory", "32g", "--cpus", "8",
            "--tmpfs", "/tmp:rw,nosuid,nodev,size=2g",
            "--tmpfs", "/run:rw,nosuid,nodev,size=64m",
            *([] if port is None else ["-p", f"127.0.0.1:{port}:6789"]),
            "-e", "ALPASIM_DRIVER_HOST=0.0.0.0", "-e", "ALPASIM_DRIVER_PORT=6789",
            "-e", f"ALPASIM_CONTESTANT_REPLICA_INDEX={index}",
            "-e", f"ALPASIM_CONTESTANT_REPLICAS={WORKERS}",
            *self.mounts(), *extra, self.meta["base_image"],
        ]


def gpus_free():
    stats = output(["nvidia-smi", "--query-gpu=index,memory.used",
                    "--format=csv,noheader,nounits"]).splitlines()
    return len(stats) == 8 and all(int(r.split(",")[1]) < 4000 for r in stats), stats


def smoke(model, gpu):
    out = model.prep / "smoke"
    out.mkdir(exist_ok=True)
    os.chmod(out, 0o777)
    cmd = model.docker_driver(
        f"{model.prefix}-smoke", gpu, None, 0, detach=False,
        extra=["-v", f"{ROOT}/e2e_challenge/axe_local_eval/trt_bench:/checks:ro",
               "-v", f"{HERE}/smoke_pkg.py:/checks_pkg/smoke_pkg.py:ro",
               "-v", f"{out}:/report", "--entrypoint", "python"],
    ) + ["/checks_pkg/smoke_pkg.py"]
    with (out / "smoke.log").open("w") as stream:
        rc = subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT).returncode
    text = (out / "smoke.log").read_text()
    assert rc == 0, f"smoke failed (exit {rc}); see {out/'smoke.log'}"
    # The policy constructor raises if the known-value MSDA check fails, so a
    # zero exit already covers it; the strict-load line is printed, not logged.
    assert "exact checkpoint load: missing=0 unexpected=0" in text, "checkpoint not strict"
    # A kernel that prints FATAL and carries on returns garbage that can still be
    # finite (10-08 vitl512 passed a finite-output check this way), so any
    # FATAL line fails the smoke outright.
    assert "FATAL" not in text, f"FATAL kernel messages in {out/'smoke.log'}"
    log("Smoke PASS: " + (out / "gpu_validation.json").read_text().replace("\n", " "))


def check(model):
    assert model.meta["checkpoint_sha256"] == output(
        ["sha256sum", model.meta["checkpoint"]]).split()[0], "checkpoint changed"
    assert json.loads((model.prep / "smoke/gpu_validation.json").read_text())["passed"]
    need = 40 if PRUNE_ASL else 190
    assert shutil.disk_usage(ROOT).free >= need * 1024**3, f"need {need} GiB free for rollouts"
    assert not model.owned_containers(), "Existing containers: refusing a duplicate launch"
    if not RESUME:
        assert not (model.run / "wizard-config.yaml").exists(), "Run already deployed"
    for port in [*range(model.port, model.port + WORKERS),
                 *range(model.wizard_port, model.wizard_port + 130)]:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))


def status(model, stage, **fields):
    (model.run / "status.json").write_text(json.dumps(
        {"stage": stage, "updated_at": datetime.now(KST).isoformat(),
         "launcher_pid": os.getpid(), **fields}, indent=2) + "\n")


def stop_owned(model):
    names = model.owned_containers()
    if not names:
        return
    log("Stopping only this evaluation's containers: " + ", ".join(names))
    archive = model.run / "driver-logs"
    archive.mkdir(exist_ok=True)
    for name in names:
        if name.startswith(model.prefix):
            with (archive / f"{name}.log").open("w") as stream:
                subprocess.run(["docker", "logs", name], stdout=stream,
                               stderr=subprocess.STDOUT, check=False)
    subprocess.run(["docker", "stop", "-t", "10", *names], check=True)
    subprocess.run(["docker", "rm", *names], check=True)


PRUNE_ASL = False
# --resume: reuse the run directory; ENABLE_AUTORESUME skips rollouts with _complete.
RESUME = False


def prune_finished_asl(model):
    """Delete rollout.asl of rollouts that are already scored.

    The runtime evaluator writes metrics.parquet when a rollout ends and the
    final aggregation reads only those files (eval/aggregation/main.py); the
    ASL is needed only for video or re-evaluation. With the disk near full a
    441 run cannot hold all ASLs (~350 MB each, ~172 GB). An ASL is removed only
    once _complete and metrics.parquet both exist and _complete is a minute old.
    Every removal is listed in asl_pruned.tsv.
    """
    freed = 0
    now = time.time()
    with (model.run / "asl_pruned.tsv").open("a") as stream:
        for done in (model.run / "rollouts").glob("*/*/_complete"):
            asl = done.parent / "rollout.asl"
            if (asl.exists() and (done.parent / "metrics.parquet").exists()
                    and now - done.stat().st_mtime > 60):
                size = asl.stat().st_size
                asl.unlink()
                freed += size
                stream.write(f"{datetime.now(KST).isoformat()}\t{done.parent.relative_to(model.run)}\t{size}\n")
    return freed


def evaluate(model):
    lock = (ROOT / f".cache/{model.run_name}.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    free, stats = gpus_free()
    assert free, f"GPUs busy: {stats}"
    check(model)
    model.run.mkdir(parents=True, exist_ok=True)
    (model.run / "evaluation_provenance.json").write_text(json.dumps({
        **{k: model.meta[k] for k in ("model", "subject", "bundle", "agent", "checkpoint",
                                      "checkpoint_sha256", "bundle_manifest_sha256",
                                      "pinhole_profile", "base_image", "config_summary")},
        "camera_order": ["CAM_L0", "CAM_F0", "CAM_R0"],
        "mounted_read_only": ["/app/navsim", "/app/nurec_pf", "/app/drivesuprim_challenge",
                              "/app/assets/drivesuprim/pkg_stage3_best.ckpt",
                              "/app/assets/drivesuprim/pkg_config.json"],
        "preset": "dev", "scenes": 441, "rollouts_per_scene": 1, "gpus": GPUS,
        "drivers": WORKERS, "workers": WORKERS, "renderers": WORKERS,
        "model_batch_size": 1, "render_bundling": "NONE", "route_reranker": False,
        "ego_geom_from_api": True,
        "mpc": {"lat_position_weight": 1.0, "long_position_weight": 0.25, "idx_start_penalty": 3},
        "throughput_topology_user_authorized": True, "keep_rollouts": True,
        "asl_pruned_after_scoring": PRUNE_ASL,
        "extra_env": model.meta.get("extra_env", {}),
        "git_commit": output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "started_at": datetime.now(KST).isoformat(),
    }, indent=2) + "\n")
    try:
        status(model, "starting_drivers")

        def start_gpu(gpu):
            for replica in range(REPLICAS):
                index = gpu * REPLICAS + replica
                subprocess.run(model.docker_driver(f"{model.prefix}-g{gpu}-{replica}", gpu,
                                                   model.port + index, index),
                               check=True, stdout=subprocess.DEVNULL)
            log(f"Started {REPLICAS} drivers on GPU {gpu}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(start_gpu, GPUS))
        pending = {f"{model.prefix}-g{g}-{r}" for g in GPUS for r in range(REPLICAS)}
        deadline = time.monotonic() + 1800
        while pending:
            for name in sorted(pending):
                assert output(["docker", "inspect", name, "--format",
                               "{{.State.Running}}"]) == "true", f"{name} exited"
                logs = subprocess.check_output(["docker", "logs", name], text=True,
                                               stderr=subprocess.STDOUT)
                if "policy load complete" in logs:
                    assert "exact checkpoint load: missing=0 unexpected=0" in logs, name
                    pending.remove(name)
            assert time.monotonic() < deadline, f"Drivers not ready: {pending}"
            if pending:
                time.sleep(3)
        log(f"All {WORKERS} drivers ready (strict checkpoint)")
        status(model, "simulation")
        with (ROOT / "runs" / f"{model.run_name}.wizard.log").open("a") as stream:
            proc = subprocess.Popen(["bash", str(AXE / "run_curated_val.sh")],
                                    env=model.environment(), stdout=stream,
                                    stderr=subprocess.STDOUT, cwd=ROOT)
        peaks = {str(g): 0 for g in GPUS}
        with (model.run / "vram.csv").open("w") as stream:
            stream.write("timestamp,gpu,memory_used_mib,memory_free_mib,utilization_percent\n")
            while proc.poll() is None:
                stats = output(["nvidia-smi",
                                "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                                "--format=csv,noheader,nounits"])
                for row in stats.splitlines():
                    gpu, used, _, _ = (v.strip() for v in row.split(","))
                    peaks[gpu] = max(peaks[gpu], int(used))
                    stream.write(f"{datetime.now(KST).isoformat()},{row}\n")
                stream.flush()
                (model.run / "vram-peaks.json").write_text(json.dumps(peaks, indent=2) + "\n")
                done = sum(1 for _ in (model.run / "rollouts").glob("**/_complete")) \
                    if (model.run / "rollouts").exists() else 0
                if PRUNE_ASL and (model.run / "rollouts").exists():
                    prune_finished_asl(model)
                status(model, "simulation", completed_rollouts=done)
                time.sleep(10)
        assert proc.returncode == 0, f"Simulation exit={proc.returncode}; see wizard log"
        assert (model.run / "aggregate/results-summary.json").exists()
        stop_owned(model)
        status(model, "complete")
        log(f"Complete: {model.run / 'aggregate/results-summary.json'}")
    except Exception as exc:
        status(model, "failed", error=str(exc))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=sorted(MODELS))
    parser.add_argument("--smoke", type=int, metavar="GPU")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="continue a stopped run; completed rollouts are skipped")
    parser.add_argument("--prune-asl", action="store_true",
                        help="delete each scored rollout.asl during the run (disk)")
    parser.add_argument("--replicas", type=int, default=6,
                        help="drivers and renderers per GPU (6 = 48 workers)")
    args = parser.parse_args()
    global REPLICAS, WORKERS, PRUNE_ASL, RESUME
    REPLICAS, WORKERS = args.replicas, args.replicas * len(GPUS)
    PRUNE_ASL = args.prune_asl
    RESUME = args.resume
    os.chdir(ROOT)
    model = Model(args.model)
    if args.smoke is not None:
        smoke(model, args.smoke)
        return
    if args.check:
        check(model)
        log("check OK")
        return
    if args.wait:
        log("Queued: waiting for all eight GPUs under 4 GB used (no job is stopped)")
        stable = 0
        while stable < 2:
            stable = stable + 1 if gpus_free()[0] else 0
            if stable < 2:
                time.sleep(30)
    evaluate(model)


if __name__ == "__main__":
    main()
