# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""Time one model's 441 run under the official ec2 contract (16 replicas x 2 rollouts).

    .venv/bin/python e2e_challenge/pkg3cam_eval/run_ec2_timing.py --model axe-v9 --date 20261012 [--wait]
    .venv/bin/python e2e_challenge/pkg3cam_eval/run_ec2_timing.py --table --date 20261012 \
        --bench runs/latency-bench-20261012_0839

The official evaluation is src/wizard/configs/e2e_challenge/ec2.yaml with the
8gpu_32rollouts topology: 16 replicas of the submitted image round-robin on
GPUs 4-7 with 2 concurrent rollouts each (32 in flight), 32 renderer replicas
on GPUs 0-3 (4 containers x 8, harmonizer off, BATCH_RENDER_RGB), physics on
4-7, and no driver call during the 1.7 s forced-GT start. The preset is used as
is; only what a local run has to change is overridden: deploy=local, the
external driver addresses, the 441 curated_val scenes, 1 rollout per scene and
ports. Controller gains are the axe submission's (lat 1.0 / lon 0.25 / idx 3),
which every model here was evaluated with. Startup-only settings (seeded
renderer cache, no .pyc writes) as in run_curated_val.sh, and
ALPASIM_FINE_RPC_BUCKETS=1 so the runtime's RPC histograms resolve percentiles.

axe-v9 runs its submitted image; the three-camera bundles run that image with
their read-only mounts (run_pkg3cam.Model), which on EC2 would be baked in.

runs/ec2-timing-<model>-<date>/timing.json holds: driver load time, wall time
from wizard start to the aggregate, the simulation span (first rollout start to
last finish), rollouts per hour, latency of Drive / image submission / render
RPCs as the simulator saw them (mean exact; percentiles interpolated in the
histogram), rollout and step durations, GPU memory peaks. rollout.asl is
deleted per rollout once scored (the disk is near full).
"""

import argparse
import concurrent.futures
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run_pkg3cam  # noqa: E402

KST = ZoneInfo("Asia/Seoul")
AXE_V9 = "696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v9"
AXE_V9_SHA = "364c801e397e9d6e36ea1f9027dc4ca804ad2e429bb588cf4e84185ca851de3d"
MODELS = ["axe-v9", *run_pkg3cam.MODELS]
REPLICAS = 16
DRIVER_GPUS = [4, 5, 6, 7]
PORT, WIZARD_PORT = 8300, 29000
GAINS = ["controller.gains.long_position_weight=0.25", "controller.gains.lat_position_weight=1.0",
         "controller.gains.idx_start_penalty=3"]
RENDER_CACHE = ROOT / ".cache/renderer-shared"


def log(message):
    print(f"[{datetime.now(KST):%F %T}] {message}", flush=True)


def output(command):
    return subprocess.check_output(command, text=True).strip()


class Driver:
    def __init__(self, key):
        self.key = key
        self.slug = key.replace("_", "")
        self.prefix = f"axe-ec2t-{self.slug}"
        if key == "axe-v9":
            self.image, self.mounts, self.strict = AXE_V9, [], False
            label = output(["docker", "image", "inspect", AXE_V9, "--format",
                            '{{index .Config.Labels "org.alpasim.checkpoint.sha256"}}'])
            assert label == AXE_V9_SHA, f"axe-v9 image carries {label}"
            self.checkpoint_sha256 = AXE_V9_SHA
        else:
            model = run_pkg3cam.Model(key)
            self.image, self.mounts, self.strict = model.meta["base_image"], model.mounts(), True
            self.checkpoint_sha256 = model.meta["checkpoint_sha256"]
            assert output(["sha256sum", model.meta["checkpoint"]]).split()[0] == self.checkpoint_sha256

    def names(self):
        return [f"{self.prefix}-r{i:02d}" for i in range(REPLICAS)]

    def owned(self, run_name):
        names = output(["docker", "ps", "-a", "--filter", f"name=^{self.prefix}-r[0-9][0-9]$",
                        "--format", "{{.Names}}"]).splitlines()
        names += output(["docker", "ps", "-a", "--filter", f"label=com.docker.compose.project={run_name}",
                         "--format", "{{.Names}}"]).splitlines()
        return sorted(set(n for n in names if n))

    def start(self):
        def one(i):
            gpu = DRIVER_GPUS[i % len(DRIVER_GPUS)]          # round-robin, as on EC2
            subprocess.run([
                "docker", "run", "-d", "--name", self.names()[i], "--init",
                "--gpus", f"device={gpu}", "--cap-drop", "ALL",
                "--security-opt", "no-new-privileges:true", "--read-only",
                "--pids-limit", "1024", "--memory", "32g", "--cpus", "8",
                "--tmpfs", "/tmp:rw,nosuid,nodev,size=2g", "--tmpfs", "/run:rw,nosuid,nodev,size=64m",
                "-p", f"127.0.0.1:{PORT + i}:6789",
                "-e", "ALPASIM_DRIVER_HOST=0.0.0.0", "-e", "ALPASIM_DRIVER_PORT=6789",
                "-e", f"ALPASIM_CONTESTANT_REPLICA_INDEX={i}", "-e", f"ALPASIM_CONTESTANT_REPLICAS={REPLICAS}",
                *self.mounts, self.image,
            ], check=True, stdout=subprocess.DEVNULL)

        with concurrent.futures.ThreadPoolExecutor(8) as pool:
            list(pool.map(one, range(REPLICAS)))
        pending, deadline = set(self.names()), time.monotonic() + 1800
        while pending:
            for name in sorted(pending):
                assert output(["docker", "inspect", name, "--format", "{{.State.Running}}"]) == "true", f"{name} exited"
                logs = subprocess.check_output(["docker", "logs", name], text=True, stderr=subprocess.STDOUT)
                if "policy load complete" in logs:
                    if self.strict:
                        assert "exact checkpoint load: missing=0 unexpected=0" in logs, name
                    pending.remove(name)
            assert time.monotonic() < deadline, f"drivers not ready: {sorted(pending)}"
            if pending:
                time.sleep(3)

    def stop(self, run_name, archive: Path):
        names = self.owned(run_name)
        if not names:
            return
        archive.mkdir(exist_ok=True)
        for name in names:
            if name.startswith(self.prefix):
                with (archive / f"{name}.log").open("w") as stream:
                    subprocess.run(["docker", "logs", name], stdout=stream, stderr=subprocess.STDOUT, check=False)
        log("Stopping only this run's containers: " + ", ".join(names))
        subprocess.run(["docker", "stop", "-t", "10", *names], check=False, stdout=subprocess.DEVNULL)
        subprocess.run(["docker", "rm", *names], check=False, stdout=subprocess.DEVNULL)


def gpu_used():
    return {int(r.split(",")[0]): int(r.split(",")[1]) for r in output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"]).splitlines()}


def overrides(run: Path, addresses: str):
    renderer_env = ["OMP_NUM_THREADS=1", "PYTHONDONTWRITEBYTECODE=1", "TORCHINDUCTOR_COMPILE_THREADS=2",
                    "TORCHINDUCTOR_CACHE_DIR=/home/.cache/torchinductor", "TRITON_CACHE_DIR=/home/.cache/triton"]
    env = lambda items: "[" + ",".join(f'"{x}"' for x in items) + "]"  # noqa: E731
    return [
        "+e2e_challenge=ec2", "+nurec_scenes=curated_val", "deploy=local", "wizard.validate_mount_points=false",
        f"wizard.log_dir={run}", f"wizard.external_services.driver={addresses}", f"wizard.baseport={WIZARD_PORT}",
        "runtime.simulation_config.n_rollouts=1", "runtime.enable_autoresume=true",
        "+runtime.endpoints.startup_timeout_s=1800", "eval.enabled=true", *GAINS,
        'services.renderer.volumes=["${scenes.scene_cache}:/mnt/nre-data",'
        f'"${{defines.sensordata}}/ego-hoods:/mnt/ego-hoods","{RENDER_CACHE}:/home/.cache"]',
        f"services.renderer.environments={env(renderer_env)}",
        f"services.physics.environments={env(['WARP_CACHE_PATH=/mnt/warp', 'PYTHONDONTWRITEBYTECODE=1'])}",
        f"services.controller.environments={env(['PYTHONDONTWRITEBYTECODE=1'])}",
        f"services.runtime.environments={env(['PYTHONDONTWRITEBYTECODE=1', 'ALPASIM_FINE_RPC_BUCKETS=1'])}",
    ]


def prune_asl(run: Path):
    now = time.time()
    for done in (run / "rollouts").glob("*/*/_complete"):
        asl = done.parent / "rollout.asl"
        if asl.exists() and (done.parent / "metrics.parquet").exists() and now - done.stat().st_mtime > 60:
            asl.unlink()


def histograms(prom: Path):
    """{(metric, service, method): {"buckets": {le: count}, "sum": s, "count": n}} summed over workers."""
    out = defaultdict(lambda: {"buckets": defaultdict(float), "sum": 0.0, "count": 0.0})
    pat = re.compile(r'^(\w+)_(bucket|sum|count)\{([^}]*)\} ([0-9.e+-]+)$')
    for line in prom.read_text().splitlines():
        m = pat.match(line)
        if not m:
            continue
        name, part, labels, value = m.groups()
        lab = dict(re.findall(r'(\w+)="([^"]*)"', labels))
        key = (name, lab.get("service", ""), lab.get("method", ""))
        if part == "bucket":
            out[key]["buckets"][float("inf") if lab["le"] == "+Inf" else float(lab["le"])] += float(value)
        else:
            out[key][part] += float(value)
    return out


def percentile(h, q):
    edges = sorted(h["buckets"])
    total = h["buckets"][edges[-1]]
    target, prev_edge, prev_count = q * total, 0.0, 0.0
    for edge in edges:
        count = h["buckets"][edge]
        if count >= target:
            if edge == float("inf"):
                return prev_edge
            frac = (target - prev_count) / (count - prev_count) if count > prev_count else 0
            return prev_edge + frac * (edge - prev_edge)
        prev_edge, prev_count = edge, count
    return edges[-1]


def latency(h):
    if not h["count"]:
        return {"n": 0, "mean_ms": None, "p50_ms": None, "p90_ms": None, "p99_ms": None}
    return {"n": int(h["count"]), "mean_ms": 1000 * h["sum"] / h["count"] if h["count"] else None,
            **{f"p{int(q * 100)}_ms": 1000 * percentile(h, q) for q in (0.5, 0.9, 0.99)}}


def evaluate(key, date, wait, smoke=0):
    drv = Driver(key)
    run_name = f"ec2-timing-{drv.slug}-{date}" + (f"-smoke{smoke}" if smoke else "")
    run = ROOT / "runs" / run_name
    lock = (ROOT / f".cache/{run_name}.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (run / "aggregate/results-summary.json").exists(), f"already done: {run}"
    assert not drv.owned(run_name), "containers of this run exist"
    if wait:
        log(f"{key}: waiting for all eight GPUs under 4 GB used")
    stable = 0
    while stable < (2 if wait else 1):     # with --wait, free on two looks 30 s apart
        stable = stable + 1 if all(v < 4000 for v in gpu_used().values()) else 0
        if not stable:
            assert wait, f"GPUs busy: {gpu_used()}"
        if stable < (2 if wait else 1):
            time.sleep(30)
    subprocess.run([str(ROOT / "e2e_challenge/axe_local_eval/warm_renderer_cache.sh")], check=True,
                   env={**os.environ, "RENDER_CACHE": str(RENDER_CACHE), "WARM_GPU": "0"})
    run.mkdir(parents=True, exist_ok=True)
    timing = {"model": key, "image": drv.image, "checkpoint_sha256": drv.checkpoint_sha256,
              "contract": "ec2 preset, 8gpu_32rollouts: 16 driver replicas x 2 rollouts on GPUs 4-7, "
                          "32 renderer replicas on 0-3, harmonizer off, BATCH_RENDER_RGB, no driver call in forced GT",
              "scenes": smoke or 441, "rollouts_per_scene": 1, "gains": GAINS,
              "git_commit": output(["git", "-C", str(ROOT), "rev-parse", "HEAD"])}
    write = lambda: (run / "timing.json").write_text(json.dumps(timing, indent=2) + "\n")  # noqa: E731
    peaks = {g: 0 for g in range(8)}
    try:
        t0 = time.time()
        timing["drivers_started_at"] = datetime.now(KST).isoformat()
        log(f"{key}: starting {REPLICAS} drivers round-robin on GPUs {DRIVER_GPUS}")
        drv.start()
        timing["driver_load_s"] = round(time.time() - t0, 1)
        log(f"{key}: drivers ready in {timing['driver_load_s']} s")
        addresses = "[" + ",".join(f'"localhost:{PORT + i}"' for i in range(REPLICAS)) + "]"
        env = {**os.environ, "UV_OFFLINE": "1", "ALPASIM_DATA_DIR": str(ROOT / "data"),
               "ALPASIM_IMAGE": "nvcr.io/nvidia/nre/nre-ga:26.04", "CONTESTANT_IMAGE": drv.image}
        t1 = time.time()
        timing["wizard_started_at"] = datetime.now(KST).isoformat()
        write()
        with (ROOT / "runs" / f"{run_name}.wizard.log").open("a") as stream:
            extra = [f"scenes.limit_to_first_n={smoke}"] if smoke else []
            proc = subprocess.Popen(["uv", "run", "alpasim_wizard", *overrides(run, addresses), *extra],
                                    cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        while proc.poll() is None:
            for g, v in gpu_used().items():
                peaks[g] = max(peaks[g], v)
            if (run / "rollouts").exists():
                prune_asl(run)
                done = sum(1 for _ in (run / "rollouts").glob("*/*/_complete"))
                timing["progress"] = {"completed": done, "at": datetime.now(KST).isoformat()}
                timing["gpu_peak_mib"] = peaks
                write()
            time.sleep(15)
        timing["wizard_s"] = round(time.time() - t1, 1)
        assert proc.returncode == 0, f"wizard exit {proc.returncode}"
        summary = json.loads((run / "aggregate/results-summary.json").read_text())
        stamps = sorted(p.stat().st_mtime for p in (run / "rollouts").glob("*/*/_complete"))
        rollouts = summary["rollouts"]
        timing.update({
            "rollouts": len(rollouts),
            "infra_failures": sum("AioRpcError" in (r.get("failure_reason") or "") for r in rollouts),
            "first_rollout_done_s": round(stamps[0] - t1, 1), "last_rollout_done_s": round(stamps[-1] - t1, 1),
            "rollouts_per_hour": round(len(stamps) / ((stamps[-1] - t1) / 3600), 1),
            "mean_scene_score": sum(r["score"] for r in rollouts) / len(rollouts),
            "gpu_peak_mib": peaks,
        })
        h = histograms(run / "telemetry/metrics.prom")
        rpc = lambda svc, method: latency(h[("rpc_duration_seconds", svc, method)])  # noqa: E731
        timing["latency"] = {
            "drive": rpc("driver", "drive"),
            "submit_image_observation": rpc("driver", "submit_image_observation"),
            "render": rpc("sensorsim", "batch_render_rgb") if h[("rpc_duration_seconds", "sensorsim", "batch_render_rgb")]["count"]
            else rpc("sensorsim", "render_rgb"),
            "controller": rpc("controller", "run_controller_and_vehicle"),
            "rollout_s_mean": h[("rollout_duration_seconds", "", "")]["sum"] / max(h[("rollout_duration_seconds", "", "")]["count"], 1),
        }
        timing["finished_at"] = datetime.now(KST).isoformat()
        write()
        log(f"{key}: done in {timing['wizard_s'] / 3600:.2f} h; drive mean {timing['latency']['drive']['mean_ms']:.0f} ms")
    finally:
        timing["gpu_peak_mib"] = peaks
        write()
        drv.stop(run_name, run / "driver-logs")


def table(date, bench):
    rows = []
    for key in MODELS:
        p = ROOT / "runs" / f"ec2-timing-{key.replace('_', '')}-{date}" / "timing.json"
        t = json.loads(p.read_text()) if p.exists() else None
        b = json.loads((Path(bench) / f"{key}.json").read_text()) if bench and (Path(bench) / f"{key}.json").exists() else None
        rows.append((key, t, b))
    md = ["| 모델 | 모델 단독 지연 평균/p99 (ms) | Drive RPC 평균 / p50 / p90 / p99 (ms) | 이미지 전달 평균 (ms) | 렌더 평균 (ms) | "
          "드라이버 기동 | 441 전체 시간 | 마지막 롤아웃까지 | 롤아웃/시간 | 장면 점수 | GPU 최대 (0-3 / 4-7, GB) |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    fmt = lambda x: "–" if x is None else f"{x:.0f}"  # noqa: E731
    for key, t, b in rows:
        bl = f"{b['latency']['mean_ms']:.1f} / {b['latency']['p99_ms']:.1f}" if b else "–"
        if not t or "latency" not in t:
            md.append(f"| {key} | {bl} | (진행 전) | | | | | | | | |")
            continue
        d, lat = t["latency"]["drive"], t["latency"]
        pk = t["gpu_peak_mib"]
        md.append(f"| {key} | {bl} | {fmt(d['mean_ms'])} / {fmt(d['p50_ms'])} / {fmt(d['p90_ms'])} / {fmt(d['p99_ms'])} | "
                  f"{fmt(lat['submit_image_observation']['mean_ms'])} | {fmt(lat['render']['mean_ms'])} | "
                  f"{t['driver_load_s'] / 60:.1f} 분 | {t['wizard_s'] / 3600:.2f} h | {t['last_rollout_done_s'] / 3600:.2f} h | "
                  f"{t['rollouts_per_hour']:.0f} | {t['mean_scene_score']:.4f} | "
                  f"{max(pk[str(g)] if str(g) in pk else pk[g] for g in range(4)) / 1024:.0f} / "
                  f"{max(pk[str(g)] if str(g) in pk else pk[g] for g in range(4, 8)) / 1024:.0f} |")
    out = ROOT / "runs" / f"ec2-timing-summary-{date}.md"
    out.write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=MODELS)
    ap.add_argument("--date", required=True)
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--table", action="store_true")
    ap.add_argument("--bench")
    ap.add_argument("--smoke", type=int, default=0, help="first N scenes only, separate run dir")
    a = ap.parse_args()
    os.chdir(ROOT)
    if a.table:
        table(a.date, a.bench)
    else:
        evaluate(a.model, a.date, a.wait, a.smoke)


if __name__ == "__main__":
    main()
