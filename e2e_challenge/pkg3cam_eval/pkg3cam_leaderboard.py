# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""Jointly fit the existing 36 subjects with the 2026-10-08 three-camera bundles.

    .venv/bin/python e2e_challenge/pkg3cam_eval/pkg3cam_leaderboard.py vits512_6layer vits768_3layer ...

Same fit as disjoint512_leaderboard.py (ZOIB, CPU, seed 42, 1000 epochs, 16
particles, rank MC 100000 / seed 0) over the same 441 clips. Every new run must
cover those 441 clips with one rollout each and carry no infrastructure (RPC)
failure, or it is refused: a renderer error scored as a 0 would rank the
simulator, not the model. Writes runs/leaderboard-<N>-with-pkg3cam-20261008/ and
the user-format CSV (7 columns) beside the fit.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "e2e_challenge/5cam_eval"))
from leaderboard import clip_scores, inputs  # noqa: E402

PREVIOUS = ROOT / "runs/leaderboard-36-with-vits512-disjoint-ep04"
PREVIOUS_CSV = PREVIOUS / "local_leaderboard.csv"
COMMON = "4096 vocab / mpc(lat/lon/idx = 1/0.25/3) / ego footprint 적용 / 원본 임계값(drivable/agent = 0.6/0.4)"
NEW = {
    "vits512_6layer": ("stage3-vits512-6layer", "leaderboard-stage3-vits512-6layer-20261008",
                       "bevformer_m + ViT-S / 카메라 3개(L0/F0/R0) 512×256 / BEV encoder 6층 / "
                       + COMMON + " / Stage 3 best(epoch14)"),
    "vits768_3layer": ("stage3-vits768-3layer", "leaderboard-stage3-vits768-3layer-20261008",
                       "bevformer_m + ViT-S / 카메라 3개(L0/F0/R0) 768×384(raw f-theta 가상 pinhole) / "
                       "BEV encoder 3층 / " + COMMON + " / Stage 3 best(epoch09)"),
    "vitl512_3layer_lr1": ("stage3-vitl512-3layer-lr1", "leaderboard-stage3-vitl512-3layer-lr1-20261008",
                           "bevformer_m + ViT-L(DINOv2) / 카메라 3개(L0/F0/R0) 512×256 / BEV encoder 3층 / "
                           "backbone lr ×1.0 / " + COMMON + " / Stage 3 best"),
}


def existing():
    local, paths = inputs()
    for name, run in (("stage3-5cam-ep05", "leaderboard-stage3-5cam-ep05-20261007"),
                      ("vits512-disjoint-baseline-ep04",
                       "leaderboard-vits512-disjoint-baseline-ep04-20261007")):
        local[name] = run
        paths[name] = ROOT / "runs" / run / "aggregate/results-summary.json"
    previous = json.loads((PREVIOUS / "manifest.json").read_text())
    assert set(paths) == set(previous["included_subject_ids"]), "36-subject set changed"
    expected = None
    for name, path in paths.items():
        clips, _ = clip_scores(path)
        expected = clips if expected is None else expected
        assert clips == expected, f"Different 441 clips: {name}"
    return local, set(paths), expected


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("models", nargs="+", choices=sorted(NEW))
    a = ap.parse_args()
    local, subjects, expected = existing()
    new = {NEW[m][0]: (ROOT / "runs" / NEW[m][1], NEW[m][2]) for m in a.models}
    for name, (run, _) in new.items():
        clips, summary = clip_scores(run / "aggregate/results-summary.json", one_rollout=True)
        assert clips == expected, f"{name}: not the same 441 clips"
        bad = [r["clipgt_id"] for r in summary["rollouts"]
               if "AioRpcError" in (r.get("failure_reason") or "")]
        assert not bad, f"{name}: infrastructure failures need a retry first: {bad}"
    total = len(subjects) + len(new)
    out = ROOT / f"runs/leaderboard-{total}-with-pkg3cam-20261008"
    assert not (out / "manifest.json").exists(), f"Refusing to overwrite {out}"
    command = ["uv", "run", "--extra", "local-evaluation", "python",
               str(ROOT / "e2e_challenge/local_evaluation/evaluate.py"),
               "--track", "pai", "--device", "cpu", "--epochs", "1000", "--seed", "42",
               "--num-particles", "16", "--rank-interval-monte-carlo-samples", "100000",
               "--rank-interval-seed", "0"]
    for name, run in local.items():
        command += ["--run", f"{name}={ROOT / 'runs' / run}"]
    for name, (run, _) in new.items():
        command += ["--run", f"{name}={run}"]
    command += ["--output-dir", str(out)]
    out.mkdir(parents=True, exist_ok=True)
    with (out / "fit.log").open("w") as stream:
        subprocess.run(command, cwd=ROOT, env={**os.environ, "UV_OFFLINE": "1",
                                               "OMP_NUM_THREADS": "12", "MKL_NUM_THREADS": "12"},
                       stdout=stream, stderr=subprocess.STDOUT, check=True)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["scenario_count"] == 441
    assert set(manifest["included_subject_ids"]) == subjects | set(new)
    assert not manifest["warnings"] and not manifest["exclusions"]
    with PREVIOUS_CSV.open(encoding="utf-8-sig", newline="") as stream:
        notes = {r["Subject"].replace(" ⚓", ""): r["참고"] for r in csv.DictReader(stream)}
    notes.update({name: note for name, (_, note) in new.items()})
    with (out / "capability_ranking.csv").open(newline="") as stream:
        rows = sorted(csv.DictReader(stream), key=lambda r: int(r["rank"]))
    assert len(rows) == total
    with (out / "local_leaderboard.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Rank", "Subject", "PCS", "Average Scene Score",
                         "At-fault-distance", "Rank 구간", "참고"])
        for row in rows:
            name = row["subject_id"]
            writer.writerow([
                row["rank"], name + (" ⚓" if name in ("alpamayo1", "alternative_2") else ""),
                round(float(row["policy_capability_score"])),
                f"{float(row['average_scene_score']):.4f}",
                f"{float(row['avg_dist_between_incidents_at_fault']):.4f}",
                f"{int(float(row['rank_lo']))}-{int(float(row['rank_hi']))}",
                notes.get(name, "reference 모델")])
    print("Complete:", out / "local_leaderboard.csv", flush=True)


if __name__ == "__main__":
    main()
