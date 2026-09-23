"""Evaluate the merged checkpoint, release GPUs, then fit the expanded local board.

Run with uv run --no-sync python -u e2e_challenge/axe_local_eval/run_merged_ep29.py.
The previous leaderboard stays intact; the expanded fit includes all its subjects.
"""

import collections
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
AXE = ROOT / "e2e_challenge/axe_local_eval"
RUN = ROOT / "runs/leaderboard-merged-ep29-bestmpc"
BOARD = ROOT / "runs/leaderboard-merged-ep29-integrated"
OLD_BOARD = ROOT / "runs/leaderboard-ep29-integrated"
STATE = AXE / "merged_ep29.status.json"
CHECKPOINT = Path("/home/kaist5/data/junseong/models/stage3_merged_epoch29.ckpt")
CHECKPOINT_SHA = "2cf6ce8d9caf1f3a3c8bf4e21b943df6f4e94d9ff1618d08177fba7f4b833d93"
IMAGE = "alpasim-e2e-drivesuprim-stage3:merged-ep29-bestmpc"
PREFIX = "axe-lb-merged-ep29"
SUBJECT = "merged-ep29-bestmpc"
NAMES = [f"{PREFIX}-g{gpu}-{replica}" for gpu in range(4, 8) for replica in range(4)]
GAINS = {
    "lat_position_weight": 1.0,
    "long_position_weight": 0.25,
    "idx_start_penalty": 3,
    "heading_weight": 1.0,
    "rel_front_steering_angle_weight": 5.0,
    "rel_acceleration_weight": 1.0,
    "acceleration_weight": 0.1,
}
EXISTING = {
    "stage3-ep30": "leaderboard-stage3-ep30",
    "axe-v8-ep24": "axe-ep24-curatedval-egobox",
    "flatvits-ep25": "leaderboard-flatvits-ep25",
    "epzero-ep29-tuned": "leaderboard-epzero-ep29",
    "epzero-ep29-base": "leaderboard-epzero-ep29-basegains",
    "axe-v5": "leaderboard-axe-v5",
    "axe-v4": "leaderboard-axe-v4",
}
child = None


def log(message):
    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)


def status(phase, **extra):
    data = {"phase": phase, "updated_at": datetime.now().astimezone().isoformat(),
            "pid": os.getpid(), "run": str(RUN), "leaderboard": str(BOARD), **extra}
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(STATE)
    log(phase)


def capture(args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def execute(args, logfile, env=None):
    global child
    with logfile.open("a") as out:
        child = subprocess.Popen(args, cwd=ROOT, env=env, stdout=out,
                                 stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = child.wait()
        finally:
            child = None
    if rc:
        raise RuntimeError(f"Command exited {rc}: {args[0]}; see {logfile}")


def interrupted(signum, frame):
    if child is not None:
        os.killpg(child.pid, signal.SIGTERM)
    raise InterruptedError(f"Received signal {signum}")


def summary(path, canonical=None, exactly_one=False):
    data = json.loads((path / "aggregate/results-summary.json").read_text())
    rows = data.get("rollouts", [])
    counts = collections.Counter(r["clipgt_id"] for r in rows)
    if len(counts) != 441 or (canonical is not None and set(counts) != canonical):
        raise ValueError(f"Incomplete or different 441-scene set: {path}")
    if exactly_one and (len(rows) != 441 or set(counts.values()) != {1}):
        raise ValueError(f"Expected one rollout for each of 441 clips: {path}")
    if any(isinstance(r.get("score"), bool) or not isinstance(r.get("score"), (int, float))
           or not math.isfinite(r["score"]) or not 0 <= r["score"] <= 1 for r in rows):
        raise ValueError(f"Invalid or missing scene score: {path}")
    return data, set(counts)


def cleanup():
    errors = []
    compose = RUN / "docker-compose.yaml"
    if compose.exists():
        result = subprocess.run(["docker", "compose", "-f", str(compose), "down",
                                 "--timeout", "10", "--remove-orphans"],
                                capture_output=True, text=True)
        if result.returncode:
            errors.append(result.stderr)
    # Only the names reserved by this run, never other GPU users' containers.
    present = set(capture(["docker", "ps", "-a", "--format", "{{.Names}}"]).splitlines())
    targets = [name for name in NAMES if name in present]
    if targets:
        result = subprocess.run(["docker", "rm", "-f", *targets], capture_output=True, text=True)
        if result.returncode:
            errors.append(result.stderr)
    if errors:
        raise RuntimeError("Container cleanup failed: " + "\n".join(errors))
    log("Evaluation containers removed; GPUs released")


def main():
    os.chdir(ROOT)
    if RUN.exists() or BOARD.exists():
        raise FileExistsError("Run/output already exists; refusing to overwrite evaluation evidence")
    old = json.loads((OLD_BOARD / "manifest.json").read_text())
    assert set(old["included_subject_ids"]) == set(old["reference_subject_ids"]) | set(EXISTING)
    _, canonical = summary(ROOT / "runs/leaderboard-epzero-ep29")
    for directory in EXISTING.values():
        summary(ROOT / "runs" / directory, canonical)
    refs = ROOT / "e2e_challenge/local_evaluation/data/pai"
    ref_manifest = json.loads((refs / "reference_manifest.json").read_text())
    for ref in ref_manifest["runs"]:
        summary((refs / ref["summary_path"]).parent.parent, canonical)
    digest = hashlib.file_digest(CHECKPOINT.open("rb"), "sha256").hexdigest()
    assert digest == CHECKPOINT_SHA, "Checkpoint changed"
    image = json.loads(capture(["docker", "image", "inspect", IMAGE]))[0]
    assert image["Config"]["Labels"]["org.alpasim.checkpoint.sha256"] == CHECKPOINT_SHA
    present = set(capture(["docker", "ps", "-a", "--format", "{{.Names}}"]).splitlines())
    assert not present.intersection(NAMES), "Reserved driver names are already in use"
    for row in capture(["nvidia-smi", "-i", "4,5,6,7", "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits"]).splitlines():
        assert int(row) < 100, "GPUs 4-7 are no longer free"
    assert os.statvfs(ROOT).f_bavail * os.statvfs(ROOT).f_frsize > 250 * 2**30
    RUN.mkdir()
    provenance = {
        "checkpoint": str(CHECKPOINT), "checkpoint_sha256": digest,
        "image": IMAGE, "image_id": image["Id"], "base_image": "alpasim-e2e-drivesuprim-stage3:epzero-ep29",
        "reference_submission": "696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v9",
        "preset": "dev", "suite": "nurec_curated_val", "clips": 441, "rollouts_per_clip": 1,
        "gpus": [4, 5, 6, 7], "drivers": 16, "workers": 16, "gains": GAINS,
        "gain_source": "ep29 search 015-screen-lat1 (117 clips, 1 repeat, score 0.74921)",
        "comparison": "Model plus MPC configuration; gains and repeat counts differ across historical subjects",
        "selection_note": "MPC selected on 117 clips within this 441-clip set using a different checkpoint; not an untouched test set",
        "git_head": capture(["git", "rev-parse", "HEAD"]),
        "git_status": capture(["git", "status", "--short"]),
        "source_hashes": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in [
            "src/eval/src/eval/data.py", "src/eval/src/eval/video.py",
            "src/wizard/configs/e2e_challenge/dev.yaml", "e2e_challenge/local_evaluation/evaluate.py"]},
        "started_at": datetime.now().astimezone().isoformat(),
    }
    (RUN / "evaluation_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    env = dict(os.environ, IMAGE=IMAGE, EXPECTED_CHECKPOINT_SHA256=CHECKPOINT_SHA,
               OFFICIAL_ENV_ONLY="1", CONTAINER_PREFIX=PREFIX, BASE_PORT="6900",
               GPU_INDICES_CSV="4,5,6,7", REPLICAS_PER_GPU="4")
    try:
        status("starting_drivers")
        driverlog = AXE / "merged_ep29.drivers.log"
        execute([str(AXE / "start_drivers.sh")], driverlog, env)
        addresses = driverlog.read_text().splitlines()[-1]
        assert len(json.loads(addresses)) == 16
        env.update(RUN_DIR=str(RUN), PRESET="dev", CONTESTANT_IMAGE=IMAGE,
                   DRIVER_ADDRESSES=addresses, N_ROLLOUTS="1", ROLLOUT_WORKERS="16",
                   RENDER_GPUS_CSV="4,5,6,7", RENDERER_REPLICAS_PER_GPU="4", NRE_CACHE_SIZE="1",
                   ENABLE_AUTORESUME="false", KEEP_ROLLOUTS="1", SCENE_IDS_FILE="", SCENE_LIMIT="0",
                   RENDER_VIDEO="false", DRIVER_CONCURRENT_ROLLOUTS="1",
                   MPC_OVERRIDES=" ".join(f"controller.gains.{k}={v}" for k, v in GAINS.items()))
        status("evaluating_441_clips")
        execute([str(AXE / "run_curated_val.sh")], Path(str(RUN) + ".wizard.log"), env)
    finally:
        cleanup()
    data, _ = summary(RUN, canonical, exactly_one=True)
    status("fitting_integrated_leaderboard", completed_clips=441)
    args = ["uv", "run", "--no-sync", "python", str(ROOT / "e2e_challenge/local_evaluation/evaluate.py"),
            "--track", "pai", "--algorithm", "zoib", "--device", "cpu"]
    for subject_id, directory in EXISTING.items():
        args.extend(["--run", f"{subject_id}={ROOT / 'runs' / directory}"])
    args.extend(["--run", f"{SUBJECT}={RUN}", "--output-dir", str(BOARD)])
    execute(args, AXE / "merged_ep29.leaderboard.log",
            dict(os.environ, OMP_NUM_THREADS="12", MKL_NUM_THREADS="12"))
    manifest = json.loads((BOARD / "manifest.json").read_text())
    assert manifest["scenario_count"] == 441 and not manifest["exclusions"]
    assert set(manifest["included_subject_ids"]) == set(old["included_subject_ids"]) | {SUBJECT}
    assert manifest["effective_algorithm"] == "zoib"
    with (BOARD / "capability_ranking.csv").open() as f:
        ranking = list(csv.DictReader(f))
    current = next(row for row in ranking if row["subject_id"] == SUBJECT)
    report = ["# Merged epoch29 — local 441-clip evaluation", "",
              f"Completed: {datetime.now().astimezone().isoformat()}", "",
              "441 clips × 1 rollout, dev preset, GPUs 4–7; all original 15 subjects included in a new 16-subject fit.",
              "", f"Checkpoint SHA256: `{digest}`", "", f"MPC gains: `{json.dumps(GAINS)}`", "",
              "Comparison is between model + MPC combinations. Historical subjects use differing MPC gains and repeat counts.",
              "MPC selection used 117 of these 441 clips on the epzero checkpoint; this is not an untouched test set.",
              "", "| Rank | Subject | PCS | Average scene score | Rank interval |",
              "|---:|---|---:|---:|---|"]
    for row in ranking:
        report.append(f"| {row['rank']} | {row['subject_id']} | {float(row['policy_capability_score']):.2f} | "
                      f"{float(row['average_scene_score']):.5f} | {row['rank_lo']}–{row['rank_hi']} |")
    reasons = dict(collections.Counter(r.get("failure_reason") or "pass" for r in data["rollouts"]))
    report.extend(["", "Failure reasons (including infrastructure failures):", "", "```json",
                   json.dumps(reasons, indent=2), "```", ""])
    (BOARD / "REPORT.md").write_text("\n".join(report))
    # Aggregation is complete and validated. Preserve per-rollout metrics and all summaries;
    # raw trajectories are the same disposable large files cleaned at the user's request.
    removed = 0
    for path in (RUN / "rollouts").rglob("rollout.asl"):
        if path.is_file() and not path.is_symlink():
            path.unlink()
            removed += 1
    status("complete", completed_clips=441, result=current, deleted_raw_rollout_files=removed)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    try:
        main()
    except BaseException as exc:
        status("failed", error=repr(exc))
        raise
