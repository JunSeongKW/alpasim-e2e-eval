# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""Retry only the infrastructure (RPC) failures of a finished pkg3cam 441 run.

    .venv/bin/python e2e_challenge/pkg3cam_eval/retry_infra.py --model vits512_vidarrender

Same procedure as axe_local_eval/retry_disjoint512_infrastructure.py (10-07),
for any run_pkg3cam.py model: the failed clips run again in a separate
<run>-infra-retry directory on GPU 0-1 with the same image, mounts, preset and
gains; the scoring configs of both runs are compared before anything moves; the
failed attempts go to attempt-history/, the retried rollouts take their place,
and the run is re-aggregated. Every other clip's score must come out bit-identical.
A physical failure (collision, offroad) in the retry is kept as the result.
"""

import argparse
import fcntl
import json
import math
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path

import run_pkg3cam as base

ROOT = base.ROOT


def is_infrastructure_failure(row):
    return "AioRpcError" in (row.get("failure_reason") or "")


def validate_replacement(before, after, failed_ids):
    old = {r["clipgt_id"]: r for r in before["rollouts"]}
    new = {r["clipgt_id"]: r for r in after["rollouts"]}
    assert len(old) == len(new) == len(after["rollouts"]) == 441
    assert old.keys() == new.keys()
    assert not any(is_infrastructure_failure(r) for r in new.values())
    for clip_id in old.keys() - failed_ids:
        assert old[clip_id]["rollout_id"] == new[clip_id]["rollout_id"], clip_id
        assert old[clip_id]["score"] == new[clip_id]["score"], clip_id
        for metric, value in old[clip_id]["score_metrics"].items():
            actual = new[clip_id]["score_metrics"][metric]
            if isinstance(value, float):
                assert math.isclose(value, actual, rel_tol=0, abs_tol=1e-12), (clip_id, metric)
            else:
                assert value == actual, (clip_id, metric)
    for clip_id in failed_ids:
        assert old[clip_id]["rollout_id"] != new[clip_id]["rollout_id"], clip_id


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", required=True, choices=sorted(base.MODELS))
    a = ap.parse_args()
    os.chdir(ROOT)
    model = base.Model(a.model)
    original, name = model.run, model.run_name
    status = json.loads((original / "status.json").read_text())
    assert status["stage"] == "complete", status
    assert not Path(f"/proc/{status['launcher_pid']}").exists(), "original launcher still running"
    before = json.loads((original / "aggregate/results-summary.json").read_text())
    failed = [r for r in before["rollouts"] if is_infrastructure_failure(r)]
    if not failed:
        base.log(f"{name}: no infrastructure failures, nothing to retry")
        return
    failed_ids = {r["clipgt_id"] for r in failed}
    valid = 441 - len(failed_ids)
    assert len(list((original / "rollouts").rglob("_complete"))) == valid

    retry_name = name + "-infra-retry"
    retry = ROOT / "runs" / retry_name
    lock = (ROOT / f".cache/{retry_name}.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not retry.exists(), f"refusing a duplicate retry: {retry}"
    base.GPUS, base.REPLICAS = [0, 1], 1
    base.WORKERS = min(2, len(failed_ids))
    base.GPUS = base.GPUS[:base.WORKERS]
    model.run, model.run_name = retry, retry_name
    model.prefix += "-retry"
    model.port += 60
    model.wizard_port += 500
    assert not model.owned_containers()
    free, stats = base.gpus_free()
    assert all(int(r.split(",")[1]) < 4000 for r in stats if int(r.split(",")[0]) in base.GPUS), stats
    for port in [*range(model.port, model.port + base.WORKERS),
                 *range(model.wizard_port, model.wizard_port + 130)]:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))

    retry.mkdir(parents=True)
    clip_file = retry / "clip_ids.txt"
    clip_file.write_text("".join(c + "\n" for c in sorted(failed_ids)))
    provenance = {
        "model": a.model,
        "checkpoint_sha256": model.meta["checkpoint_sha256"],
        "base_image": model.meta["base_image"],
        "clip_ids": sorted(failed_ids),
        "original_valid_results_preserved": valid,
        "retry_reason": [r["failure_reason"].splitlines()[2].strip() if r["failure_reason"].count("\n") > 2
                         else r["failure_reason"][:200] for r in failed],
        "gpus": base.GPUS, "workers": base.WORKERS, "preset": "dev",
        "mpc": {"lat_position_weight": 1.0, "long_position_weight": 0.25, "idx_start_penalty": 3},
        "rollouts_per_scene": 1,
        "score_selection": "first infrastructure-complete retry; physical failures retained",
        "git_commit": base.output(["git", "rev-parse", "HEAD"]),
    }
    (retry / "evaluation_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    try:
        for index, gpu in enumerate(base.GPUS):
            subprocess.run(model.docker_driver(f"{model.prefix}-g{gpu}-0", gpu, model.port + index, index),
                           check=True, stdout=subprocess.DEVNULL)
        pending = {f"{model.prefix}-g{g}-0" for g in base.GPUS}
        deadline = time.monotonic() + 900
        while pending:
            for cname in sorted(pending):
                assert base.output(["docker", "inspect", cname, "--format", "{{.State.Running}}"]) == "true"
                logs = subprocess.check_output(["docker", "logs", cname], text=True, stderr=subprocess.STDOUT)
                if "policy load complete" in logs:
                    assert "exact checkpoint load: missing=0 unexpected=0" in logs, cname
                    pending.remove(cname)
            assert time.monotonic() < deadline, f"drivers not ready: {pending}"
            if pending:
                time.sleep(3)
        base.log(f"retry drivers ready; simulating {sorted(failed_ids)}")
        env = model.environment()
        env.update(SCENE_IDS_FILE=str(clip_file))
        with (ROOT / "runs" / f"{retry_name}.wizard.log").open("w") as stream:
            subprocess.run(["bash", str(base.AXE / "run_curated_val.sh")], env=env,
                           stdout=stream, stderr=subprocess.STDOUT, check=True, cwd=ROOT)
        summary = json.loads((retry / "aggregate/results-summary.json").read_text())
        assert {r["clipgt_id"] for r in summary["rollouts"]} == failed_ids
        assert not any(is_infrastructure_failure(r) for r in summary["rollouts"]), "retry failed again"

        from omegaconf import OmegaConf
        for cfg_name in ["eval-config.yaml", "controller-config.yaml"]:
            assert OmegaConf.load(original / cfg_name) == OmegaConf.load(retry / cfg_name), cfg_name
        assert (OmegaConf.load(original / "generated-user-config-0.yaml").simulation_config
                == OmegaConf.load(retry / "generated-user-config-0.yaml").simulation_config)
    finally:
        base.stop_owned(model)

    history = original / "attempt-history/infrastructure-failures"
    history.mkdir(parents=True)
    (history / "infrastructure_failures.json").write_text(json.dumps(failed, indent=2) + "\n")
    (original / "aggregate").rename(history / "aggregate")
    for row in failed:
        source = original / "rollouts" / row["clipgt_id"] / row["rollout_id"]
        if source.exists():
            target = history / "rollouts" / row["clipgt_id"] / row["rollout_id"]
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
    for row in summary["rollouts"]:
        source = retry / "rollouts" / row["clipgt_id"] / row["rollout_id"]
        (source / "rollout.asl").unlink(missing_ok=True)
        target = original / "rollouts" / row["clipgt_id"] / row["rollout_id"]
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)

    from eval.aggregation.main import _run_aggregation_core
    from eval.schema import EvalConfig
    cfg = OmegaConf.merge(OmegaConf.structured(EvalConfig), OmegaConf.load(original / "eval-config.yaml"))
    _run_aggregation_core([original], original / "aggregate", cfg)
    after = json.loads((original / "aggregate/results-summary.json").read_text())
    validate_replacement(before, after, failed_ids)
    assert len(list((original / "rollouts").rglob("_complete"))) == 441
    p = json.loads((original / "evaluation_provenance.json").read_text())
    p["infrastructure_retry"] = {**provenance, "archived_initial_attempt": str(history), "retry_run": str(retry)}
    (original / "evaluation_provenance.json").write_text(json.dumps(p, indent=2) + "\n")
    (original / "infrastructure_retry_audit.json").write_text(json.dumps({
        "passed": True, "clips": 441, "one_completed_rollout_per_clip": True,
        "valid_original_clips_unchanged": valid,
        "retried_infrastructure_clips": sorted(failed_ids),
        "final_infrastructure_failures": 0,
    }, indent=2) + "\n")
    base.log(f"{name}: retried {len(failed_ids)} clip(s); 441 clips, 0 infrastructure failures")


if __name__ == "__main__":
    main()
