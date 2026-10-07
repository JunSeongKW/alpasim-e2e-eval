# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""Retry only infrastructure failures, preserving the 439 valid model results."""

import concurrent.futures
import fcntl
import json
import math
import os
import shutil
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path

import run_disjoint512 as base

ROOT = base.ROOT
HERE = base.HERE
ORIGINAL = base.RUN
ORIGINAL_NAME = base.RUN_NAME
RETRY_NAME = ORIGINAL_NAME + "-infra-retry"
RETRY = ROOT / "runs" / RETRY_NAME
OUT = ROOT / "runs/leaderboard-36-with-vits512-disjoint-ep04"
HISTORY = ORIGINAL / "attempt-history/initial-two-rpc-failures"


def is_infrastructure_failure(row):
    return "AioRpcError" in (row.get("failure_reason") or "")


def write_status(stage, **fields):
    value = {
        "stage": stage,
        "updated_at": datetime.now(base.KST).isoformat(),
        "launcher_pid": os.getpid(),
        "retry_run": RETRY_NAME,
        **fields,
    }
    (ORIGINAL / "status.json").write_text(json.dumps(value, indent=2) + "\n")
    (RETRY / "status.json").write_text(json.dumps(value, indent=2) + "\n")


def validate_replacement(before, after, failed_ids):
    old = {r["clipgt_id"]: r for r in before["rollouts"]}
    new = {r["clipgt_id"]: r for r in after["rollouts"]}
    assert len(old) == len(new) == len(after["rollouts"]) == 441
    assert old.keys() == new.keys()
    assert not any(is_infrastructure_failure(r) for r in new.values())
    for clip_id in old.keys() - failed_ids:
        assert old[clip_id]["rollout_id"] == new[clip_id]["rollout_id"], clip_id
        assert abs(old[clip_id]["score"] - new[clip_id]["score"]) < 1e-12, clip_id
        left, right = old[clip_id]["score_metrics"], new[clip_id]["score_metrics"]
        assert left.keys() == right.keys(), clip_id
        for metric, value in left.items():
            actual = right[metric]
            if isinstance(value, (int, float)) and isinstance(actual, (int, float)):
                assert math.isclose(value, actual, rel_tol=0, abs_tol=1e-12), (
                    clip_id,
                    metric,
                )
            else:
                assert value == actual, (clip_id, metric)
    for clip_id in failed_ids:
        assert old[clip_id]["rollout_id"] != new[clip_id]["rollout_id"]


def main():
    os.chdir(ROOT)
    old_pid = json.loads((ORIGINAL / "status.json").read_text())["launcher_pid"]
    assert not Path(f"/proc/{old_pid}").exists(), "Original launcher must finish first"
    lock = (ROOT / f".cache/{RETRY_NAME}.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not RETRY.exists(), "Refusing duplicate retry deployment"
    before = json.loads((ORIGINAL / "aggregate/results-summary.json").read_text())
    failed = [r for r in before["rollouts"] if is_infrastructure_failure(r)]
    assert len(failed) == 2
    failed_ids = {r["clipgt_id"] for r in failed}
    assert len(list((ORIGINAL / "rollouts").rglob("_complete"))) == 439
    assert len(list((ORIGINAL / "rollouts").rglob("metrics.parquet"))) == 439
    base.RUN_NAME, base.RUN = RETRY_NAME, RETRY
    base.PREFIX = "axe-disjoint512-ep04-retry"
    base.GPUS, base.REPLICAS, base.WORKERS = [0, 1], 1, 2
    base.PORT, base.WIZARD_PORT = 7600, 24000
    assert not base.owned_containers()
    image_id = base.output(
        ["docker", "image", "inspect", base.IMAGE, "--format", "{{.Id}}"]
    )
    assert image_id == (base.PREP / "image-id.txt").read_text().strip()
    stats = base.output(
        [
            "nvidia-smi",
            "-i",
            "0,1",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
    )
    assert all(int(v) < 4000 for v in stats.splitlines()), stats
    for port in [7600, 7601, *range(24000, 24020)]:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    RETRY.mkdir(parents=True)
    (ROOT / f".cache/{RETRY_NAME}.launcher.pid").write_text(str(os.getpid()) + "\n")
    clip_file = RETRY / "clip_ids.txt"
    clip_file.write_text("".join(c + "\n" for c in sorted(failed_ids)))
    provenance = {
        "image_id": image_id,
        "checkpoint_sha256": base.SHA,
        "clip_ids": sorted(failed_ids),
        "original_valid_results_preserved": 439,
        "retry_reason": "renderer connection UNAVAILABLE before HTTP/2 SETTINGS",
        "gpus": [0, 1],
        "workers": 2,
        "model_batch_size": 1,
        "driver_grpc_workers": 8,
        "preset": "dev",
        "mpc": [1.0, 0.25, 3],
        "rollouts_per_scene": 1,
        "score_selection": "first infrastructure-complete retry; physical failures retained",
        "git_commit": base.output(["git", "rev-parse", "HEAD"]),
    }
    (RETRY / "evaluation_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )
    try:
        write_status("infrastructure_retry_starting", valid_clips=439, pending_clips=2)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(base.start_gpu, [0, 1]))
        pending = {f"{base.PREFIX}-g{g}-0" for g in [0, 1]}
        deadline = time.monotonic() + 900
        while pending:
            for name in sorted(pending):
                assert (
                    base.output(
                        ["docker", "inspect", name, "--format", "{{.State.Running}}"]
                    )
                    == "true"
                )
                logs = subprocess.check_output(
                    ["docker", "logs", name], text=True, stderr=subprocess.STDOUT
                )
                if "policy load complete" in logs:
                    assert "exact checkpoint load: missing=0 unexpected=0" in logs
                    pending.remove(name)
            assert time.monotonic() < deadline
            if pending:
                time.sleep(2)
        env = base.environment(RETRY)
        env.update(RENDER_GPUS_CSV="0,1", SCENE_IDS_FILE=str(clip_file))
        write_status(
            "infrastructure_retry_simulation", valid_clips=439, pending_clips=2
        )
        with (ROOT / "runs" / f"{RETRY_NAME}.wizard.log").open("w") as stream:
            subprocess.run(
                ["bash", str(HERE / "run_curated_val.sh")],
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
        retry_summary = json.loads(
            (RETRY / "aggregate/results-summary.json").read_text()
        )
        assert len(retry_summary["rollouts"]) == 2
        assert {r["clipgt_id"] for r in retry_summary["rollouts"]} == failed_ids
        assert not any(is_infrastructure_failure(r) for r in retry_summary["rollouts"])
        assert len(list((RETRY / "rollouts").rglob("_complete"))) == 2
        # Check every scoring setting before moving any completed data.
        from omegaconf import OmegaConf

        for name in ["eval-config.yaml", "controller-config.yaml"]:
            assert OmegaConf.load(ORIGINAL / name) == OmegaConf.load(RETRY / name)
        assert (
            OmegaConf.load(ORIGINAL / "generated-user-config-0.yaml").simulation_config
            == OmegaConf.load(RETRY / "generated-user-config-0.yaml").simulation_config
        )
        base.stop_owned()
        HISTORY.mkdir(parents=True)
        (HISTORY / "infrastructure_failures.json").write_text(
            json.dumps(failed, indent=2) + "\n"
        )
        shutil.copyfile(ORIGINAL / "status.json", HISTORY / "retry_status.json")
        (ORIGINAL / "aggregate").rename(HISTORY / "aggregate")
        OUT.rename(HISTORY / "preliminary_leaderboard_36")
        for row in failed:
            source = ORIGINAL / "rollouts" / row["clipgt_id"] / row["rollout_id"]
            target = HISTORY / "rollouts" / row["clipgt_id"] / row["rollout_id"]
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
        for row in retry_summary["rollouts"]:
            source = RETRY / "rollouts" / row["clipgt_id"] / row["rollout_id"]
            target = ORIGINAL / "rollouts" / row["clipgt_id"] / row["rollout_id"]
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
        write_status(
            "reaggregating_after_infrastructure_retry", valid_clips=441, pending_clips=0
        )
        from eval.aggregation.main import _run_aggregation_core
        from eval.schema import EvalConfig

        cfg = OmegaConf.merge(
            OmegaConf.structured(EvalConfig),
            OmegaConf.load(ORIGINAL / "eval-config.yaml"),
        )
        _run_aggregation_core([ORIGINAL], ORIGINAL / "aggregate", cfg)
        after = json.loads((ORIGINAL / "aggregate/results-summary.json").read_text())
        validate_replacement(before, after, failed_ids)
        assert len(list((ORIGINAL / "rollouts").rglob("_complete"))) == 441
        p = json.loads((ORIGINAL / "evaluation_provenance.json").read_text())
        p["infrastructure_retry"] = {
            **provenance,
            "archived_initial_attempt": str(HISTORY),
            "retry_run": str(RETRY),
        }
        (ORIGINAL / "evaluation_provenance.json").write_text(
            json.dumps(p, indent=2) + "\n"
        )
        (ORIGINAL / "infrastructure_retry_audit.json").write_text(
            json.dumps(
                {
                    "passed": True,
                    "clips": 441,
                    "one_completed_rollout_per_clip": True,
                    "valid_original_clips_unchanged": 439,
                    "retried_infrastructure_clips": sorted(failed_ids),
                    "final_infrastructure_failures": 0,
                },
                indent=2,
            )
            + "\n"
        )
        write_status(
            "fitting_leaderboard_after_retry", valid_clips=441, pending_clips=0
        )
        subprocess.run(
            [str(ROOT / ".venv/bin/python"), str(HERE / "disjoint512_leaderboard.py")],
            env={**env, "OMP_NUM_THREADS": "12", "MKL_NUM_THREADS": "12"},
            check=True,
        )
        write_status(
            "complete", valid_clips=441, pending_clips=0, infrastructure_failures=0
        )
        base.log("Complete: valid 441 clips and corrected 36-subject leaderboard")
    except Exception as exc:
        write_status("infrastructure_retry_failed", error=str(exc))
        raise


if __name__ == "__main__":
    main()
