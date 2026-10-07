# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""Validate the 441-scene inputs and jointly refit 34 existing subjects + 5cam.

The published reference anchors and ranking policy come from evaluate.py.
No PCS from the previous, separate 34-subject fit is reused.
"""

import argparse
import csv
import json
import math
import os
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SUBJECT = "stage3-5cam-ep05"
RUN_NAME = "leaderboard-stage3-5cam-ep05-20261007"
DEFAULT_RUN = ROOT / "runs" / RUN_NAME
DEFAULT_OUT = ROOT / "runs/leaderboard-35-with-stage3-5cam-ep05"


def inputs():
    old = json.loads((HERE / "existing_subjects.json").read_text())
    if len(old) != 26:
        raise RuntimeError("Requires the original 26 local subjects")
    reference_manifest = (
        ROOT / "e2e_challenge/local_evaluation/data/pai/reference_manifest.json"
    )
    references = json.loads(reference_manifest.read_text())["runs"]
    if len(references) != 8:
        raise RuntimeError("Requires the original eight reference subjects")
    paths = {
        name: ROOT / "runs" / run / "aggregate/results-summary.json"
        for name, run in old.items()
    }
    paths.update(
        {
            r["subject_id"]: reference_manifest.parent / r["summary_path"]
            for r in references
        }
    )
    previous = json.loads((ROOT / "runs/leaderboard-261006/manifest.json").read_text())
    if set(paths) != set(previous["included_subject_ids"]):
        raise RuntimeError("Inputs differ from the existing 34-subject leaderboard")
    return old, paths


def clip_scores(path, *, one_rollout=False):
    summary = json.loads(path.read_text())
    if summary.get("scene_score_enabled") is not True:
        raise RuntimeError(f"{path}: scene score must be enabled")
    clips = Counter()
    for row in summary["rollouts"]:
        score = row.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise TypeError(f"{path}: invalid/missing score for {row.get('clipgt_id')}")
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise RuntimeError(f"{path}: score outside [0,1]")
        clips[row["clipgt_id"]] += 1
    if len(clips) != 441 or (one_rollout and set(clips.values()) != {1}):
        raise RuntimeError(
            f"{path}: requires 441 unique clips"
            + (" and exactly one rollout each" if one_rollout else "")
        )
    return set(clips), summary


def check_inputs(new_run=None):
    old, paths = inputs()
    expected = None
    for name, path in paths.items():
        clips, _ = clip_scores(path)
        if expected is None:
            expected = clips
        if clips != expected:
            raise RuntimeError(f"{name}: different 441-scene set")
    if new_run is not None:
        clips, summary = clip_scores(
            new_run / "aggregate/results-summary.json", one_rollout=True
        )
        if clips != expected:
            raise RuntimeError("5cam clips differ from the original leaderboard")
        # Preserve recorded zero scores and report RPC/infrastructure failures.
        # They must never disappear by restricting to successful clips.
        health = dict(
            Counter((r.get("failure_reason") or "none") for r in summary["rollouts"])
        )
        (new_run / "evaluation_health.json").write_text(
            json.dumps(health, indent=2) + "\n"
        )
        print("5cam rollout failure reasons:", health)
    print("PASS: original 34 subjects have identical, valid 441-scene summaries")
    return old


def fit(new_run, output):
    old = check_inputs(new_run)
    if (output / "manifest.json").exists():
        raise RuntimeError(f"Refusing to overwrite an existing leaderboard: {output}")
    command = [
        "uv",
        "run",
        "--extra",
        "local-evaluation",
        "python",
        str(ROOT / "e2e_challenge/local_evaluation/evaluate.py"),
        "--track",
        "pai",
        "--device",
        "cpu",
        "--epochs",
        "1000",
        "--seed",
        "42",
        "--num-particles",
        "16",
        "--rank-interval-monte-carlo-samples",
        "100000",
        "--rank-interval-seed",
        "0",
    ]
    for name, run in old.items():
        command += ["--run", f"{name}={ROOT / 'runs' / run}"]
    command += ["--run", f"{SUBJECT}={new_run}", "--output-dir", str(output)]
    output.mkdir(parents=True, exist_ok=True)
    with (output / "fit.log").open("w") as log:
        subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            env={**os.environ, "UV_OFFLINE": "1"},
            check=True,
        )
    manifest = json.loads((output / "manifest.json").read_text())
    if (
        manifest["scenario_count"] != 441
        or set(manifest["included_subject_ids"]) != set(inputs()[1]) | {SUBJECT}
        or manifest["exclusions"]
        or manifest["warnings"]
    ):
        raise RuntimeError(
            "Joint fit did not include all 35 subjects and all 441 scenes cleanly"
        )
    with (ROOT / "e2e_challenge/axe_local_eval/data/local_leaderboard_261006.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        notes = {
            r["Subject"].replace(" ⚓", ""): r["참고"] for r in csv.DictReader(handle)
        }
    notes[SUBJECT] = (
        "bevformer_m + ViT-S / 카메라 5개(L1/L0/F0/R0/R1) / 4096 vocab / "
        "BEV 56×112(-56~56m, -28~28m) / mpc(lat/lon/idx = 1/0.25/3) / "
        "ego footprint 적용 / Stage 3 ep05"
    )
    with (output / "capability_ranking.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    destination = output / "local_leaderboard.csv"
    with destination.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "Rank",
                "Subject",
                "PCS",
                "Average Scene Score",
                "At-fault-distance (km)",
                "Rank 구간 (95%)",
                "참고",
            ]
        )
        for row in sorted(rows, key=lambda r: int(r["rank"])):
            name = row["subject_id"]
            writer.writerow(
                [
                    row["rank"],
                    name + (" ⚓" if name in ("alpamayo1", "alternative_2") else ""),
                    round(float(row["policy_capability_score"])),
                    f"{float(row['average_scene_score']):.4f}",
                    f"{float(row['avg_dist_between_incidents_at_fault']):.4f}",
                    f"{row['rank_lo']}-{row['rank_hi']}",
                    notes.get(name, "reference 모델"),
                ]
            )
    # The paired report uses exactly the same recorded clip scores as the fit.
    with (output / "paired_with_axe_v9.txt").open("w") as log:
        subprocess.run(
            [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "e2e_challenge/axe_local_eval/compare_on_clips.py"),
                f"{SUBJECT}={new_run / 'aggregate/results-summary.json'}",
                f"axe-v9={ROOT / 'runs/leaderboard-merged-route-ep30/aggregate/results-summary.json'}",
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print("Joint 35-subject leaderboard:", destination)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-existing", action="store_true")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    if args.check_existing:
        check_inputs()
    else:
        fit(args.run_dir, args.output_dir)
