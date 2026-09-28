"""Per-clip CSV for one 441-clip run, in the column layout of nurec1606_eval.csv.

Model columns come from the run's results-summary.json; scene columns come from
the pool that scan_scene_attrs.py builds. The scoring is not reimplemented here:
`official_score` and `band` are imported from analyze_1606.py, which produced the
1606-clip file, so the `score` and `turn_band` columns mean exactly the same
thing in both and the two files can be concatenated or joined.

`score` is recomputed from the three hard gates (collision_at_fault, offroad,
left_corridor_laterally) and then saturated progress, rather than read from the
summary's own `score`, so that it means the same thing as in the 1606 file. For
the runs here the two agree on every clip, which confirms these runs already
scored the corridor gate; the recomputation is what makes that checkable instead
of assumed, and it is also what fills `failure_reason` with the gate that fired.

`brightness` is left empty for scenes whose .usdz carries no preview frames --
22 of the 441, all from the 26.01 release. Rendering substitutes would not be
the same quantity as the 1606 file's, so the cell stays blank.

Usage:
    python per_clip_csv.py runs/<run>/aggregate/results-summary.json \
        --pool pool441.json --out out.csv
"""
import argparse
import collections
import csv
import importlib.util
import json
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_ANALYZE = _HERE.parent / "sample_submission_drivesuprim" / "analyze_1606.py"
_spec = importlib.util.spec_from_file_location("analyze_1606", _ANALYZE)
_analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_analyze)
official_score, band, COLS = _analyze.official_score, _analyze.band, _analyze.COLS


def build_rows(summary_path: str, pool: dict) -> list[dict]:
    rollouts = json.load(open(summary_path))["rollouts"]
    rows = []
    for r in rollouts:
        cid = r["clipgt_id"].replace("clipgt-", "")
        sm = r.get("score_metrics") or {}
        m = r.get("metrics") or {}
        at = pool.get(cid, {})
        score, reason = official_score(sm)
        if score is None:  # the rollout never produced metrics
            score, reason = 0.0, "scene_error"
        turn = at.get("turn")
        rows.append(
            {
                "clip_id": cid,
                "score": round(score, 6),
                "failure_reason": reason,
                "collision_at_fault": sm.get("collision_at_fault"),
                "offroad": sm.get("offroad"),
                "progress_clipped_rel": sm.get("progress_clipped_rel"),
                "gt_dist_traveled_m": sm.get("gt_dist_traveled_m"),
                "lateral_dist_to_gt_trajectory": sm.get("lateral_dist_to_gt_trajectory"),
                "min_distance_to_obstacle_m": m.get("min_distance_to_obstacle_m"),
                "turn_deg": turn,
                "turn_band": band(turn),
                "vmed": at.get("vmed"),
                "near_mean": at.get("near_mean"),
                "mindist": at.get("mindist"),
                "brightness": at.get("brightness"),
            }
        )
    rows.sort(key=lambda x: x["clip_id"])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--label", default="")
    a = ap.parse_args()

    pool = json.load(open(a.pool))
    rows = build_rows(a.summary, pool)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)

    name = a.label or pathlib.Path(a.summary).parts[-3]
    s = np.array([r["score"] for r in rows])
    local = np.array(
        [r["score"] for r in json.load(open(a.summary))["rollouts"] if r.get("score") is not None]
    )
    print(f"CSV: {a.out}  ({len(rows)}행 x {len(COLS)}열)")
    print(f"\n[{name}]")
    print(f"  공식 규정 (3게이트)  평균 {s.mean():.4f}   득점 {int((s>0).sum())}/{len(s)} ({100*(s>0).mean():.1f}%)")
    print(f"  로컬 규정 (2게이트)  평균 {local.mean():.4f}   득점 {int((local>0).sum())}/{len(local)}")

    c = collections.Counter(r["failure_reason"] for r in rows)
    fails = len(rows) - c["pass"]
    print(f"\n  실패 {fails}건")
    for k, v in c.most_common():
        if k != "pass":
            print(f"    {k:<22}{v:>5}   전체 {100*v/len(rows):>5.1f}%   실패내 {100*v/fails:>5.1f}%")

    missing = sum(1 for r in rows if r["brightness"] is None)
    if missing:
        print(f"\n  brightness 빈칸 {missing}건 (usdz 에 미리보기 프레임 없음)")


if __name__ == "__main__":
    main()
