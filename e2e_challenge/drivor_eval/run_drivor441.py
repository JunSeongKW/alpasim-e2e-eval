# SPDX-License-Identifier: Apache-2.0
"""DrivoR (mingyu's handoff bundle) on curated_val 441, reference-matched protocol.

    .venv/bin/python e2e_challenge/drivor_eval/run_drivor441.py --model ours_hugsim_final --smoke 2
    .venv/bin/python e2e_challenge/drivor_eval/run_drivor441.py --model ours_hugsim_final [--wait] [--prune-asl]

Protocol (protocol/PROTOCOL_DIFFERENCES.md of the bundle): +e2e_challenge=dev
+nurec_scenes=curated_val, 3 rollouts per scene (the 8 reference subjects' 441x3),
the stock nonlinear MPC gains (MPC_OVERRIDES="" -- NOT the axe tuned 1/0.25/3),
nothing tuned on these scenes. The scoring contract comes from the preset via
e2e_challenge/axe_local_eval/run_curated_val.sh; what this script sets is
placement only (GPUs, replicas, ports) plus the driver.

Drivers are host processes of drivor_driver.py in the drivor-eval conda env,
run through the env link kjs-DrivoR-alpasim441 so nvidia-smi names them; one
model per process, one rollout at a time per driver, as many drivers as
rollout workers. The adapter settings below are frozen for both models and
recorded in evaluation_provenance.json.

--smoke N runs the first N scenes with one rollout under a separate run name.
--wait queues until the chosen GPUs are under 4 GB used.
"""

import argparse
import concurrent.futures
import fcntl
import hashlib
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
KIT = ROOT.parent / "mingyu/DRIVOR_ALPASIM_NUREC441_HANDOFF_20261008_2005"
ENV_PY = Path("/home/kaist5/miniconda3/envs/kjs-DrivoR-alpasim441/bin/python")
KST = ZoneInfo("Asia/Seoul")
MODELS = {"drivor_original": 0, "ours_hugsim_final": 1}
# Frozen adapter settings, identical for both models (see drivor_driver.py).
# FP32 rather than the bundle's historical FP16 autocast: in FP16 the scorer's
# sigmoid underflows to 0 once every candidate's collision logit is below
# about -16.6, all scores become -inf and the policy (correctly) refuses to
# choose -- seen at the simulator's first step, whose acceleration carries an
# initialisation spike (-21.6 m/s^2). FP32 has the same weights and inputs and
# picks trajectories 0.01-0.02 m apart at the median (replay of 5 rollouts,
# runs/drivor-conformance-20261009). Ego inputs stay as the simulator reports
# them: clamping to NAVSIM's range touched 28% of steps and moved the 4 s point
# by up to 10-13 m at p95.
PRECISION = "fp32"
EGO_CLAMP = "none"


def log(message):
    print(f"[{datetime.now(KST):%F %T}] {message}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Run:
    def __init__(self, a):
        self.a = a
        self.gpus = [int(g) for g in a.gpus.split(",")]
        self.workers = a.replicas * len(self.gpus)
        stamp = a.date
        kind = f"smoke{a.smoke}" if a.smoke else f"441x{a.rollouts}"
        self.name = f"drivor-{a.model.replace('_', '-')}-{kind}-{stamp}"
        self.dir = ROOT / "runs" / self.name
        # Distinct port blocks per model so both can be prepared side by side.
        self.port = a.port_base + 100 * MODELS[a.model] + (50 if a.smoke else 0)
        self.wizard_port = a.wizard_port_base + 1000 * MODELS[a.model] + (500 if a.smoke else 0)
        self.pid_file = self.dir / "drivers.pids"

    def addresses(self):
        return "[" + ",".join(f'"localhost:{p}"' for p in range(self.port, self.port + self.workers)) + "]"

    def environment(self):
        return {
            **os.environ,
            "UV_OFFLINE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "RUN_NAME": self.name, "RUN_DIR": str(self.dir), "PRESET": "dev",
            "DRIVER_ADDRESSES": self.addresses(),
            "N_ROLLOUTS": "1" if self.a.smoke else str(self.a.rollouts),
            "ROLLOUT_WORKERS": str(self.workers),
            "RENDER_GPUS_CSV": ",".join(map(str, self.gpus)),
            "RENDERER_REPLICAS_PER_GPU": str(self.a.replicas),
            "NRE_CACHE_SIZE": "1",
            "SCENE_LIMIT": str(self.a.smoke or 0), "SCENE_IDS_FILE": "",
            "RENDER_VIDEO": "false", "KEEP_ROLLOUTS": "1", "ENABLE_AUTORESUME": "true",
            "SERVICE_STARTUP_TIMEOUT_SEC": "1800", "FAST_STARTUP": "1", "DRIVER_CONCURRENT_ROLLOUTS": "1",
            "ALPASIM_IMAGE": "nvcr.io/nvidia/nre/nre-ga:26.04", "NRE_IMAGE": "nvcr.io/nvidia/nre/nre-ga:26.04",
            # Stock controller: the reference subjects were run with it.
            "MPC_OVERRIDES": "",
            "EXTRA_OVERRIDES": f"wizard.baseport={self.wizard_port}",
        }

    # ---- drivers -------------------------------------------------------------------------
    def driver_pids(self):
        if not self.pid_file.exists():
            return {}
        pids = {}
        for line in self.pid_file.read_text().splitlines():
            index, pid = line.split()
            pids[int(index)] = int(pid)
        return pids

    def alive(self, pid):
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
        except FileNotFoundError:
            return False
        return "drivor_driver.py" in cmd and self.name in cmd

    def start_drivers(self):
        assert not any(self.alive(p) for p in self.driver_pids().values()), "drivers already running"
        (self.dir / "drivers").mkdir(parents=True, exist_ok=True)
        lines = []
        for index in range(self.workers):
            gpu = self.gpus[index // self.a.replicas]
            out = self.dir / "drivers" / f"drv-{index:02d}"
            out.mkdir(exist_ok=True)
            cmd = [str(ENV_PY), str(HERE / "drivor_driver.py"), "--policy", self.a.model,
                   "--port", str(self.port + index), "--device", "cuda:0", "--precision", PRECISION,
                   "--ego-clamp", EGO_CLAMP, "--log-dir", str(out), "--threads", "2"]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "2"}
            with (out / "stdout.log").open("w") as stream:
                proc = subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT, env=env,
                                        start_new_session=True, cwd=str(self.dir))
            lines.append(f"{index} {proc.pid}")
        self.pid_file.write_text("\n".join(lines) + "\n")
        deadline = time.monotonic() + 900
        pending = set(range(self.workers))
        while pending:
            for index in sorted(pending):
                text = (self.dir / "drivers" / f"drv-{index:02d}" / "stdout.log").read_text(errors="replace")
                assert "Traceback" not in text, f"driver {index} failed; see its stdout.log"
                assert self.alive(self.driver_pids()[index]), f"driver {index} exited"
                if "serving drivor-" in text:
                    pending.remove(index)
            assert time.monotonic() < deadline, f"drivers not ready: {sorted(pending)}"
            if pending:
                time.sleep(3)
        log(f"All {self.workers} drivers serving ({self.a.model}, {PRECISION}, ego clamp {EGO_CLAMP})")

    def stop_drivers(self):
        pids = {i: p for i, p in self.driver_pids().items() if self.alive(p)}
        if pids:
            log(f"Stopping this run's {len(pids)} driver processes: {sorted(pids.values())}")
        for pid in pids.values():
            os.kill(pid, signal.SIGTERM)
        for _ in range(30):
            if not any(self.alive(p) for p in pids.values()):
                break
            time.sleep(1)
        for pid in pids.values():
            if self.alive(pid):
                os.kill(pid, signal.SIGKILL)

    def receipts(self):
        """Sum of all sessions' counters; a full run must show zero errors."""
        total, sessions = {}, 0
        for path in (self.dir / "drivers").glob("drv-*/sessions.jsonl"):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                sessions += 1
                for k, v in row.items():
                    if isinstance(v, (int, float)) and k not in ("wall_s",):
                        total[k] = max(total.get(k, 0), v) if k.startswith("max_") else total.get(k, 0) + v
        return {"sessions": sessions, **total}

    # ---- run -----------------------------------------------------------------------------
    def gpus_free(self):
        rows = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used",
                                        "--format=csv,noheader,nounits"], text=True).splitlines()
        used = {int(r.split(",")[0]): int(r.split(",")[1]) for r in rows}
        return all(used[g] < 4000 for g in self.gpus), used

    def check(self):
        ckpts = json.loads((KIT / "CHECKPOINTS.json").read_text())
        for name, info in ckpts.items():
            assert sha256(KIT / info["path"]) == info["sha256"], f"checkpoint changed: {name}"
        need = 40 if self.a.prune_asl else 200 * (self.a.rollouts if not self.a.smoke else 0) + 20
        assert shutil.disk_usage(ROOT).free >= need * 1024**3, f"need {need} GiB free"
        for port in [*range(self.port, self.port + self.workers),
                     *range(self.wizard_port, self.wizard_port + 130)]:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", port))
        if not self.a.resume:
            assert not (self.dir / "wizard-config.yaml").exists(), "run already deployed (use --resume)"
        return ckpts

    def provenance(self, ckpts):
        git = lambda *args: subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()  # noqa: E731
        return {
            "model": self.a.model, "bundle": str(KIT),
            "checkpoints": {k: v["sha256"] for k, v in ckpts.items()},
            "adapter": {
                "precision": PRECISION, "ego_clamp": EGO_CLAMP,
                "camera_map": "pai_views.py: F0/L0/R0 rotation-only from front_wide/cross_left/cross_right; "
                              "B0 proxy stitch of rear_left+rear_right (ground + 15 m cylinder); "
                              "unseen pixels edge-filled; nuPlan 1920x1080 distorted pinhole",
                "nuplan_rig_sha256": sha256(HERE / "nuplan_rig.json"),
                "command": "official determine_command_from_route, threshold 3.0 m, lookahead 20 m",
                "output": "8 poses 0.5..4.0 s + current pose, no interpolation",
                "files_sha256": {f: sha256(HERE / f) for f in ("drivor_driver.py", "pai_views.py")},
            },
            "protocol": {"preset": "dev", "suite": "curated_val", "scenes": self.a.smoke or 441,
                         "rollouts_per_scene": 1 if self.a.smoke else self.a.rollouts,
                         "mpc": "stock (no overrides)", "render_bundling": "NONE (preset default)"},
            "placement": {"gpus": self.gpus, "drivers": self.workers, "workers": self.workers,
                          "renderers": self.workers, "driver_port_base": self.port,
                          "wizard_port_base": self.wizard_port},
            "asl_pruned_after_scoring": self.a.prune_asl,
            "git_commit": git("rev-parse", "HEAD"), "git_dirty": git("status", "--short", "--", "e2e_challenge/drivor_eval"),
            "started_at": datetime.now(KST).isoformat(),
        }

    def status(self, stage, **fields):
        (self.dir / "status.json").write_text(json.dumps(
            {"stage": stage, "updated_at": datetime.now(KST).isoformat(), "launcher_pid": os.getpid(),
             **fields}, indent=2) + "\n")

    def prune_finished_asl(self):
        freed = 0
        now = time.time()
        with (self.dir / "asl_pruned.tsv").open("a") as stream:
            for done in (self.dir / "rollouts").glob("*/*/_complete"):
                asl = done.parent / "rollout.asl"
                if asl.exists() and (done.parent / "metrics.parquet").exists() and now - done.stat().st_mtime > 60:
                    size = asl.stat().st_size
                    asl.unlink()
                    freed += size
                    stream.write(f"{datetime.now(KST).isoformat()}\t{done.parent.relative_to(self.dir)}\t{size}\n")
        return freed

    def evaluate(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        lock = (self.dir / ".launcher.lock").open("w")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        free, used = self.gpus_free()
        assert free, f"GPUs busy: {used}"
        ckpts = self.check()
        (self.dir / "evaluation_provenance.json").write_text(json.dumps(self.provenance(ckpts), indent=2) + "\n")
        try:
            self.status("starting_drivers")
            self.start_drivers()
            self.status("simulation")
            with (ROOT / "runs" / f"{self.name}.wizard.log").open("a") as stream:
                proc = subprocess.Popen(["bash", str(ROOT / "e2e_challenge/axe_local_eval/run_curated_val.sh")],
                                        env=self.environment(), stdout=stream, stderr=subprocess.STDOUT, cwd=ROOT)
            peaks = {str(g): 0 for g in self.gpus}
            with (self.dir / "vram.csv").open("a") as stream:
                while proc.poll() is None:
                    rows = subprocess.check_output(
                        ["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu",
                         "--format=csv,noheader,nounits"], text=True)
                    for row in rows.splitlines():
                        gpu, mem = row.split(",")[0].strip(), int(row.split(",")[1])
                        if gpu in peaks:
                            peaks[gpu] = max(peaks[gpu], mem)
                        stream.write(f"{datetime.now(KST).isoformat()},{row}\n")
                    stream.flush()
                    dead = [i for i, p in self.driver_pids().items() if not self.alive(p)]
                    if dead:
                        log(f"driver processes died: {dead}; stopping the simulation")
                        proc.terminate()
                        raise RuntimeError(f"drivers died: {dead}")
                    done = sum(1 for _ in (self.dir / "rollouts").glob("**/_complete")) \
                        if (self.dir / "rollouts").exists() else 0
                    if self.a.prune_asl and (self.dir / "rollouts").exists():
                        self.prune_finished_asl()
                    self.status("simulation", completed_rollouts=done, vram_peak_mib=peaks)
                    time.sleep(15)
            assert proc.returncode == 0, f"simulation exit {proc.returncode}; see the wizard log"
            assert (self.dir / "aggregate/results-summary.json").exists()
            receipt = self.receipts()
            (self.dir / "driver_receipts.json").write_text(json.dumps(receipt, indent=2) + "\n")
            log(f"driver receipts: {receipt}")
            assert receipt.get("errors", 0) == 0, "driver errors during the run"
            self.status("complete", receipts=receipt)
            log(f"Complete: {self.dir / 'aggregate/results-summary.json'}")
        except Exception as exc:
            self.status("failed", error=str(exc))
            raise
        finally:
            self.stop_drivers()
            self.remove_containers()

    def remove_containers(self):
        """The wizard stops its compose services but leaves them; remove only this run's."""
        names = subprocess.run(["docker", "ps", "-a", "--filter", f"label=com.docker.compose.project={self.name}",
                                "--format", "{{.Names}}"], capture_output=True, text=True).stdout.split()
        if names:
            log(f"Removing this run's containers: {', '.join(sorted(names))}")
            subprocess.run(["docker", "stop", "-t", "10", *names], check=False, stdout=subprocess.DEVNULL)
            subprocess.run(["docker", "rm", *names], check=False, stdout=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=sorted(MODELS))
    ap.add_argument("--gpus", default="4,5,6,7")
    ap.add_argument("--replicas", type=int, default=6, help="drivers and renderers per GPU")
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--smoke", type=int, default=0, metavar="N_SCENES")
    ap.add_argument("--date", default=datetime.now(KST).strftime("%Y%m%d"))
    ap.add_argument("--port-base", type=int, default=7000)
    ap.add_argument("--wizard-port-base", type=int, default=31000)
    ap.add_argument("--prune-asl", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--wait", action="store_true")
    a = ap.parse_args()
    os.chdir(ROOT)
    run = Run(a)
    if a.wait:
        log(f"Queued: waiting for GPUs {run.gpus} under 4 GB used (no job is stopped)")
        stable = 0
        while stable < 2:
            stable = stable + 1 if run.gpus_free()[0] else 0
            if stable < 2:
                time.sleep(30)
    run.evaluate()


if __name__ == "__main__":
    main()
