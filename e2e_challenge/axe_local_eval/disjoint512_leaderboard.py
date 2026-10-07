# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""Jointly fit the existing 35 subjects and the new disjoint ViT-S checkpoint."""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "e2e_challenge/5cam_eval"))
from leaderboard import clip_scores, inputs

SUBJECT = "vits512-disjoint-baseline-ep04"
RUN_NAME = "leaderboard-vits512-disjoint-baseline-ep04-20261007"
RUN = ROOT / "runs" / RUN_NAME
OUT = ROOT / "runs/leaderboard-36-with-vits512-disjoint-ep04"


def check(new_run=None):
    local, paths = inputs()
    local["stage3-5cam-ep05"] = "leaderboard-stage3-5cam-ep05-20261007"
    paths["stage3-5cam-ep05"] = (
        ROOT / "runs" / local["stage3-5cam-ep05"] / "aggregate/results-summary.json"
    )
    previous = json.loads(
        (ROOT / "runs/leaderboard-35-with-stage3-5cam-ep05/manifest.json").read_text()
    )
    assert set(paths) == set(previous["included_subject_ids"])
    expected = None
    for name, path in paths.items():
        clips, _ = clip_scores(path)
        if expected is None:
            expected = clips
        assert clips == expected, f"Different 441 clips: {name}"
    if new_run is not None:
        clips, _ = clip_scores(
            new_run / "aggregate/results-summary.json", one_rollout=True
        )
        assert clips == expected, "New run must have the same 441 clips"
    print("PASS: existing 35 subjects use the same 441 clips", flush=True)
    return local, set(paths)


def fit():
    local, expected = check(RUN)
    assert not (OUT / "manifest.json").exists(), "Refusing to overwrite a leaderboard"
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
    for name, run in local.items():
        command += ["--run", f"{name}={ROOT / 'runs' / run}"]
    command += ["--run", f"{SUBJECT}={RUN}", "--output-dir", str(OUT)]
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "fit.log").open("w") as stream:
        subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "UV_OFFLINE": "1"},
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    manifest = json.loads((OUT / "manifest.json").read_text())
    assert manifest["scenario_count"] == 441
    assert set(manifest["included_subject_ids"]) == expected | {SUBJECT}
    assert not manifest["warnings"] and not manifest["exclusions"]
    with (ROOT / "e2e_challenge/axe_local_eval/data/local_leaderboard_261007.csv").open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        notes = {
            r["Subject"].replace(" ⚓", ""): r["참고"] for r in csv.DictReader(stream)
        }
    notes[SUBJECT] = (
        "bevformer_m + ViT-S / 카메라 3개(L0/F0/R0) / 4096 vocab / "
        "mpc(lat/lon/idx = 1/0.25/3) / ego footprint 적용 / "
        "train·val 분리 학습 / Stage 3 epoch04-step4075 / axe-v9 추론 설정"
    )
    with (OUT / "capability_ranking.csv").open(newline="") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda r: int(r["rank"]))
    assert len(rows) == 36
    with (OUT / "local_leaderboard.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "Rank",
                "Subject",
                "PCS",
                "Average Scene Score",
                "At-fault-distance",
                "Rank 구간",
                "참고",
            ]
        )
        for row in rows:
            name = row["subject_id"]
            writer.writerow(
                [
                    row["rank"],
                    name + (" ⚓" if name in ("alpamayo1", "alternative_2") else ""),
                    round(float(row["policy_capability_score"])),
                    f"{float(row['average_scene_score']):.4f}",
                    f"{float(row['avg_dist_between_incidents_at_fault']):.4f}",
                    f"{int(float(row['rank_lo']))}-{int(float(row['rank_hi']))}",
                    notes.get(name, "reference 모델"),
                ]
            )
    with (OUT / "paired_with_axe_v9_and_5cam.txt").open("w") as stream:
        subprocess.run(
            [
                str(ROOT / ".venv/bin/python"),
                str(ROOT / "e2e_challenge/axe_local_eval/compare_on_clips.py"),
                f"{SUBJECT}={RUN / 'aggregate/results-summary.json'}",
                f"axe-v9={ROOT / 'runs/leaderboard-merged-route-ep30/aggregate/results-summary.json'}",
                f"stage3-5cam-ep05={ROOT / 'runs/leaderboard-stage3-5cam-ep05-20261007/aggregate/results-summary.json'}",
            ],
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print("Complete:", OUT / "local_leaderboard.csv", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-existing", action="store_true")
    args = parser.parse_args()
    if args.check_existing:
        check()
    else:
        fit()
