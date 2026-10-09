# SPDX-License-Identifier: Apache-2.0
"""Drive-IRT fit of the two DrivoR runs with the 8 published PAI reference subjects.

    .venv/bin/python e2e_challenge/drivor_eval/fit_drivor.py \
        --original runs/drivor-drivor-original-441x3-<date> --ours runs/drivor-ours-hugsim-final-441x3-<date>

The bundle's protocol: one fit holding all eight reference subjects (loaded
by evaluate.py from data/pai/reference_manifest.json; alpamayo1 = 2000 and
alternative_2 = 600 anchor the local scale) and both DrivoR rows, nothing else
-- not our 39-subject leaderboard. Same fit settings as every earlier local
fit (ZOIB, CPU, seed 42, 1000 epochs, 16 particles, rank MC 100000 / seed 0).
Each run must cover the same 441 clips with the expected rollouts per clip and
no infrastructure (RPC) failure: a renderer error scored as 0 would rank the
simulator. Repeat rollouts are averaged per scene by evaluate.py.
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


def check_run(run: Path, rollouts: int):
    summary = json.loads((run / "aggregate/results-summary.json").read_text())
    assert summary.get("scene_score_enabled") is True, f"{run}: scene score disabled"
    clips = Counter()
    for row in summary["rollouts"]:
        score = row.get("score")
        assert isinstance(score, (int, float)) and not isinstance(score, bool) and math.isfinite(score) \
            and 0 <= score <= 1, f"{run}: bad score for {row.get('clipgt_id')}"
        clips[row["clipgt_id"]] += 1
    assert len(clips) == 441, f"{run}: {len(clips)} clips, need 441"
    assert set(clips.values()) == {rollouts}, f"{run}: rollouts per clip {Counter(clips.values())}"
    bad = [r["clipgt_id"] for r in summary["rollouts"] if "AioRpcError" in (r.get("failure_reason") or "")]
    assert not bad, f"{run}: infrastructure failures need a retry first: {bad[:5]}"
    receipts = json.loads((run / "driver_receipts.json").read_text())
    assert receipts.get("errors", 0) == 0, f"{run}: driver errors {receipts}"
    return set(clips)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--ours", type=Path, required=True)
    ap.add_argument("--rollouts", type=int, default=3)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    runs = {"drivor_original": a.original.resolve(), "ours_hugsim_final": a.ours.resolve()}
    clips = {name: check_run(run, a.rollouts) for name, run in runs.items()}
    assert clips["drivor_original"] == clips["ours_hugsim_final"], "different 441 clip sets"
    out = (a.out or ROOT / f"runs/leaderboard-drivor-8ref-{a.original.name.rsplit('-', 1)[-1]}").resolve()
    assert not (out / "manifest.json").exists(), f"refusing to overwrite {out}"
    out.mkdir(parents=True, exist_ok=True)
    command = ["uv", "run", "--extra", "local-evaluation", "python",
               str(ROOT / "e2e_challenge/local_evaluation/evaluate.py"),
               "--track", "pai", "--device", "cpu", "--epochs", "1000", "--seed", "42",
               "--num-particles", "16", "--rank-interval-monte-carlo-samples", "100000",
               "--rank-interval-seed", "0", "--output-dir", str(out)]
    for name, run in runs.items():
        command += ["--run", f"{name}={run}"]
    with (out / "fit.log").open("w") as stream:
        subprocess.run(command, cwd=ROOT, env={**os.environ, "UV_OFFLINE": "1", "OMP_NUM_THREADS": "12",
                                               "MKL_NUM_THREADS": "12"},
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["scenario_count"] == 441
    assert len(manifest["included_subject_ids"]) == 10, manifest["included_subject_ids"]
    with (out / "capability_ranking.csv").open(newline="") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda r: int(r["rank"]))
    with (out / "drivor_leaderboard.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Rank", "Subject", "PCS", "Average Scene Score", "At-fault-distance", "Rank interval"])
        for row in rows:
            name = row["subject_id"]
            writer.writerow([row["rank"], name + (" (anchor)" if name in ("alpamayo1", "alternative_2") else ""),
                             round(float(row["policy_capability_score"])),
                             f"{float(row['average_scene_score']):.4f}",
                             f"{float(row['avg_dist_between_incidents_at_fault']):.4f}",
                             f"{int(float(row['rank_lo']))}-{int(float(row['rank_hi']))}"])
    print("warnings:", manifest["warnings"], "exclusions:", manifest["exclusions"])
    print("Complete:", out / "drivor_leaderboard.csv", flush=True)


if __name__ == "__main__":
    main()
