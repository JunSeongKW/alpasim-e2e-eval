"""Per-clip results for the 1606-clip run, scored by the challenge rules.

The challenge scores three hard gates, not two: `collision_at_fault`,
`offroad`, and `left_corridor_laterally`. The third is upstream NVIDIA's,
added in `f012862` (PR #163) on the `e2e_challenge` branch this checkout
tracks, and the organizer-published reference runs under
`../local_evaluation/data/pai/` list it in their own `score_criteria` and
attribute rollout failures to it. An earlier version of this file treated it
as a local addition and scored on two gates; that was wrong and understated
every failure count.

Scores are recomputed here from the raw gate values rather than read from
`score`, so the per-clip breakdown can be joined to the scenario features.
"""
import argparse, collections, csv, io, json, os, zipfile
import numpy as np

# Challenge gates (NVlabs/alpasim@e2e_challenge, eval/aggregation/scene_score.py).
# Any non-zero value zeroes the scene score; lateral corridor exit is measured
# against max_dist_to_gt_trajectory = 4.0 m in both the dev and ec2 presets.
OFFICIAL_GATES = ("collision_at_fault", "offroad", "left_corridor_laterally")
PROGRESS_SATURATION = 0.8      # progress at/above this scores full marks
MIN_GT_DIST_M = 5.0            # shorter recordings score full progress

COLS = [
    "clip_id", "score", "failure_reason",
    "collision_at_fault", "offroad",
    "progress_clipped_rel", "gt_dist_traveled_m",
    "lateral_dist_to_gt_trajectory", "min_distance_to_obstacle_m",
    "turn_deg", "turn_band", "vmed", "near_mean", "mindist", "brightness",
]


def official_score(sm):
    """Score under the challenge rules: three gates, then saturated progress."""
    gt = sm.get("gt_dist_traveled_m")
    prog = sm.get("progress_clipped_rel")
    if gt is None or prog is None:
        return None, None
    progress_score = 1.0 if gt < MIN_GT_DIST_M else min(
        min(max(prog, 0.0), 1.0) / PROGRESS_SATURATION, 1.0)
    for g in OFFICIAL_GATES:
        if sm.get(g):
            return 0.0, g
    return progress_score, "pass"


def band(t):
    if t is None or t < 0:
        return "unknown"
    return "90+" if t >= 90 else "60-90" if t >= 60 else "45-60" if t >= 45 else "20-45" if t >= 20 else "<20"


def brightness(usdz_dir, clip_id):
    path = os.path.join(usdz_dir, f"{clip_id}.usdz")
    if not usdz_dir or not os.path.exists(path):
        return None
    try:
        from PIL import Image
        with zipfile.ZipFile(path) as z:
            frames = [n for n in z.namelist()
                      if n.startswith("frames/") and n.endswith((".jpeg", ".jpg"))]
            if not frames:
                return None
            vals = [np.asarray(Image.open(io.BytesIO(z.read(f))).convert("L")).mean()
                    for f in frames]
        return round(float(np.mean(vals)), 1)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("summary")
    ap.add_argument("--pool", required=True)
    ap.add_argument("--usdz-dir", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rollouts = json.load(open(a.summary))["rollouts"]
    attrs = {v["id"]: v for v in json.load(open(a.pool)).values()}

    rows = []
    for r in rollouts:
        cid = r["clipgt_id"].replace("clipgt-", "")
        sm, m, at = r.get("score_metrics", {}), r.get("metrics", {}), attrs.get(cid, {})
        score, reason = official_score(sm)
        if score is None:                       # rollout never produced metrics
            score, reason = 0.0, "scene_error"
        turn = at.get("turn")
        rows.append({
            "clip_id": cid,
            "score": round(score, 6),
            "failure_reason": reason,
            "collision_at_fault": sm.get("collision_at_fault"),
            "offroad": sm.get("offroad"),
            "progress_clipped_rel": sm.get("progress_clipped_rel"),
            "gt_dist_traveled_m": sm.get("gt_dist_traveled_m"),
            "lateral_dist_to_gt_trajectory": sm.get("lateral_dist_to_gt_trajectory"),
            "min_distance_to_obstacle_m": m.get("min_distance_to_obstacle_m"),
            "turn_deg": turn, "turn_band": band(turn),
            "vmed": at.get("vmed"), "near_mean": at.get("near_mean"),
            "mindist": at.get("mindist"),
            "brightness": brightness(a.usdz_dir, cid),
        })

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(rows)
    print(f"CSV: {a.out}  ({len(rows)}행 x {len(COLS)}열)")

    s = np.array([r["score"] for r in rows])
    l = np.array([r["score"] for r in rollouts])
    print(f"\n■ 공식 규정  평균 {s.mean():.4f}  완주 {int((s>0).sum())}/{len(s)} ({100*(s>0).mean():.1f}%)")
    print(f"■ 로컬 규정  평균 {l.mean():.4f}  완주 {int((l>0).sum())}/{len(l)} ({100*(l>0).mean():.1f}%)"
          f"   (회랑 이탈을 0점 처리)")

    c = collections.Counter(r["failure_reason"] for r in rows)
    fails = len(rows) - c["pass"]
    print(f"\n■ 실패 원인 ({fails}건, 공식 규정)")
    for k, v in c.most_common():
        if k != "pass":
            print(f"  {k:22}{v:>6}   전체 {100*v/len(rows):>5.1f}%   실패내 {100*v/fails:>5.1f}%")

    print(f"\n■ 회전각 구간별 (공식 규정)")
    print(f"  {'구간':10}{'수':>6}{'평균':>9}{'완주율':>9}   주요 실패")
    g = collections.defaultdict(list)
    for r in rows:
        g[r["turn_band"]].append(r)
    for b in ("<20", "20-45", "45-60", "60-90", "90+", "unknown"):
        if b not in g:
            continue
        rr = g[b]
        sc = np.array([x["score"] for x in rr])
        fc = collections.Counter(x["failure_reason"] for x in rr if x["failure_reason"] != "pass")
        print(f"  {b:10}{len(rr):>6}{sc.mean():>9.3f}{100*(sc>0).mean():>8.1f}%   "
              + "  ".join(f"{k} {v}" for k, v in fc.most_common(3)))


if __name__ == "__main__":
    main()
