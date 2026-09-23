"""Compare runs on exactly the clips they all have, not on whatever each finished.

A run still in flight has scored only part of the scene set, and the clips it has
are the first ones in processing order rather than a random sample. Putting its
partial mean beside another run's full-set mean compares two different scene
sets, and the gap that shows is mostly the scene difficulty, not the policy.

So the intersection is taken first and every run is re-aggregated over it.
Metrics that are rates average directly; distance per at-fault incident is a
ratio of sums and is recomputed as one.
"""

import argparse
import json
import pathlib
import statistics

RATES = [
    "collision_at_fault",
    "collision_any",
    "offroad",
    "left_corridor_laterally",
    "wrong_lane",
]
MEANS = ["dist_traveled_m", "dist_to_gt_trajectory", "duration_frac_20s"]


def rollouts_of(path: pathlib.Path) -> dict:
    """clip -> rollout record, for rollouts that carry a score."""
    summary = json.loads(path.read_text())
    out = {}
    for r in summary.get("rollouts", []):
        if r.get("score") is None:
            continue
        out.setdefault(r["clipgt_id"], []).append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="label=path/to/results-summary.json")
    args = ap.parse_args()

    data = {}
    for spec in args.runs:
        label, _, path = spec.partition("=")
        data[label] = rollouts_of(pathlib.Path(path))

    common = set.intersection(*(set(v) for v in data.values()))
    print(f"공통 클립 {len(common)}개로 비교"
          + "".join(f"   {l}:{len(v)}" for l, v in data.items()))
    print()

    rows = {}
    for label, per_clip in data.items():
        recs = [r for c in common for r in per_clip[c]]
        m = [r["metrics"] for r in recs]
        row = {
            "avg_scene_score": statistics.fmean(r["score"] for r in recs),
            "zero_rate": sum(1 for r in recs if r["score"] == 0.0) / len(recs),
        }
        for k in RATES + MEANS:
            vals = [x[k] for x in m if isinstance(x.get(k), (int, float))]
            row[k] = statistics.fmean(vals) if vals else None
        # Distance per at-fault incident is a ratio of sums, not a mean of ratios.
        dist = sum(x["dist_traveled_m"] for x in m if isinstance(x.get("dist_traveled_m"), (int, float)))
        inc = sum(
            1 for x in m
            if x.get("collision_at_fault") or x.get("offroad")
        )
        row["dist_per_incident_km"] = dist / inc / 1000 if inc else None
        rows[label] = row

    keys = ["avg_scene_score", "zero_rate", "dist_per_incident_km"] + RATES + MEANS
    w = max(len(k) for k in keys) + 2
    print(f"{'지표':<{w}}" + "".join(f"{l[:13]:>15}" for l in rows))
    print("-" * (w + 15 * len(rows)))
    for k in keys:
        line = f"{k:<{w}}"
        for label in rows:
            v = rows[label][k]
            line += f"{v:>15.4f}" if isinstance(v, float) else f"{'-':>15}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
