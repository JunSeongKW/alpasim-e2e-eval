"""Does a policy slow to a near-stop at curves where the logged human did not?

    .venv/bin/python e2e_challenge/axe_local_eval/curve_stop_vs_human.py \
        --run axe-v9=runs/curve-axe-v9-441-20261010 --out runs/curve-stop-vs-human-20261010

Each clip is compared with its own human log at the same route position, not at
the same time: a policy that fell behind early is still judged on what it did
when it reached the curve. Route position is distance travelled from the start
of the clip; the sim's per-step dist_traveled_m and the usdz rig trajectory both
start there (same timestamps), so no alignment is needed.

Curve zone: the 15 m window with the largest heading change in the human log
(scenario_analysis feature table) plus 20 m of approach before it. Curve clips
have |yaw| >= 30 deg in that window; straight clips (|yaw| < 15) get the same
zone on their own (nearly straight) window, so both groups are measured on the
same route length by the same rule.

Unwarranted near-stop: the ego is below 1 m/s for at least 1 s, at a position
where the human never dropped below 3 m/s within +-2 m. Steps count from the
first eval_relevant step up to the first collision of any kind -- after that the
motion is no longer only the policy's.
"""

import argparse
import csv
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT.parent / "scenario_analysis/analysis_result_vehicle_country/tables/val441_features_vehicle_country.csv"
USDZ = ROOT / "data/nre-artifacts/all-usdzs"
APPROACH_M = 20.0
STOP_MS, STOP_S = 1.0, 1.0
HUMAN_MOVING_MS, HUMAN_NEAR_M = 3.0, 2.0
CURVE_DEG, STRAIGHT_DEG = 30.0, 15.0
LAG_MAX_S = 3.0
NEEDED = ["dist_traveled_m", "collision_any", "eval_relevant"]


def smooth(v, k=5):
    return np.convolve(np.pad(v, k // 2, mode="edge"), np.ones(k) / k, mode="valid")


def usdz_paths():
    """clip id -> usdz files that hold its log (named by clip id or by scene uuid)."""
    paths = {}
    for csv_path in sorted((ROOT / "data/scenes").glob("sim_scenes*.csv")):
        with csv_path.open() as f:
            for r in csv.DictReader(f):
                clip = r["scene_id"].removeprefix("clipgt-")
                paths.setdefault(clip, [USDZ / f"{clip}.usdz"]).append(USDZ / f"{r['uuid']}.usdz")
    return paths


def human_log(clip, candidates):
    path = next(p for p in candidates if p.exists())
    with zipfile.ZipFile(path) as z:
        seq = json.loads(z.read("rig_trajectories.json"))["rig_trajectories"][0]
    xy = np.asarray(seq["T_rig_worlds"], dtype=np.float64)[:, :2, 3]
    t = np.asarray(seq["T_rig_world_timestamps_us"], dtype=np.int64)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    v = smooth(np.diff(s) / (np.diff(t) / 1e6))
    return t, s, np.r_[v, v[-1]]


def human_slowest_near(hs, hv, s0):
    near = np.abs(hs - s0) <= HUMAN_NEAR_M
    return float(hv[near].min()) if near.any() else float("nan")


def time_in(s, t_s, lo, hi):
    """Seconds a monotone track spends between route positions lo and hi."""
    if hi <= lo:
        return float("nan")
    a, b = np.interp([lo, hi], s, t_s, left=np.nan, right=np.nan)
    return float(b - a)


def rollout_record(clip, feat, human, steps):
    ht, hs, hv = human
    t = (steps.index.values - ht[0]) / 1e6
    s = steps["dist_traveled_m"].values
    relevant = np.flatnonzero(steps["eval_relevant"].values > 0)
    hit = np.flatnonzero(steps["collision_any"].values > 0)
    i0 = relevant[0] if relevant.size else 0
    i1 = hit[0] if hit.size else len(t)
    t, s = t[i0:i1], s[i0:i1]
    if len(t) < 5:
        return None
    v = smooth(np.diff(s) / np.diff(t))
    v = np.r_[v, v[-1]]
    ts_m, te_m = feat["turn_window_start_m"], feat["turn_window_end_m"]
    zone = (max(0.0, ts_m - APPROACH_M), te_m)

    stops, run = [], []
    for k in range(len(t)):
        if v[k] < STOP_MS:
            run.append(k)
            continue
        if run and t[run[-1]] - t[run[0]] >= STOP_S - 1e-6:
            stops.append(run)
        run = []
    if run and t[run[-1]] - t[run[0]] >= STOP_S - 1e-6:
        stops.append(run)
    ht_s = (ht - ht[0]) / 1e6
    unwarranted = []
    for r in stops:
        pos = float(np.median(s[r]))
        human_min = human_slowest_near(hs, hv, pos)
        if human_min >= HUMAN_MOVING_MS:
            # How far behind the human the ego reached this spot. The log's other
            # agents replay on the human's clock, so a late ego meets a different
            # scene there (a light that has turned, crossing traffic).
            lag = float(t[r[0]] - np.interp(pos, np.maximum.accumulate(hs), ht_s))
            unwarranted.append((pos, float(t[r[-1]] - t[r[0]]), human_min, lag))
    in_zone = [u for u in unwarranted if zone[0] <= u[0] <= zone[1]]

    # Speed through the part of the zone the ego actually covered in the window.
    lo, hi = max(zone[0], s[0]), min(zone[1], s[-1])
    ego_t = time_in(np.maximum.accumulate(s), t, lo, hi) if hi - lo >= 5 else float("nan")
    hum_t = time_in(np.maximum.accumulate(hs), ht_s, lo, hi) if hi - lo >= 5 else float("nan")
    return {
        "clip": clip,
        "yaw_deg": feat["yaw"],
        "group": "curve" if feat["yaw"] >= CURVE_DEG else "straight" if feat["yaw"] < STRAIGHT_DEG else "mild",
        "turn": feat["turn_label"],
        "zone_start_m": round(zone[0], 1),
        "zone_end_m": round(zone[1], 1),
        "start_m": round(float(s[0]), 1),
        "end_m": round(float(s[-1]), 1),
        "cut_by_collision": bool(hit.size),
        "reached_zone": bool(s[-1] >= zone[0]),
        "cleared_zone": bool(s[-1] >= zone[1]),
        "near_stop_in_zone": bool(in_zone),
        "near_stop_in_zone_s": round(sum(u[1] for u in in_zone), 1),
        "near_stop_in_zone_at_m": round(in_zone[0][0], 1) if in_zone else None,
        "human_min_ms_there": round(in_zone[0][2], 2) if in_zone else None,
        "lag_behind_human_s": round(in_zone[0][3], 1) if in_zone else None,
        "near_stop_in_zone_on_time": any(u[3] <= LAG_MAX_S for u in in_zone),
        "near_stop_outside_zone": bool(len(unwarranted) > len(in_zone)),
        "zone_speed_ratio": round(hum_t / ego_t, 3) if ego_t and ego_t > 0 and np.isfinite(hum_t) else None,
        "ego_min_ms_in_zone": round(float(v[(s >= zone[0]) & (s <= zone[1])].min()), 2)
        if ((s >= zone[0]) & (s <= zone[1])).any() else None,
        "human_min_ms_in_zone": round(float(hv[(hs >= zone[0]) & (hs <= zone[1])].min()), 2),
    }


def load_run(run_dir):
    path = Path(run_dir) / "aggregate/metrics_unprocessed.parquet"
    d = pd.read_parquet(path, columns=["name", "timestamps_us", "values", "clipgt_id", "rollout_id"],
                        filters=[("name", "in", NEEDED)])
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="LABEL=RUN_DIR")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    feats = {}
    with FEATURES.open() as f:
        for r in csv.DictReader(f):
            feats[r["clip_id"]] = {
                "yaw": abs(float(r["max_windowed_yaw_deg"])),
                "turn_window_start_m": float(r["turn_window_start_m"]),
                "turn_window_end_m": float(r["turn_window_end_m"]),
                "turn_label": r["turn_label"],
            }
    humans, files = {}, usdz_paths()
    rows = []
    for spec in a.run:
        label, run_dir = spec.split("=", 1)
        d = load_run(run_dir)
        for (cgt, rid), g in d.groupby(["clipgt_id", "rollout_id"]):
            clip = cgt.removeprefix("clipgt-")
            if clip not in feats:
                continue
            if clip not in humans:
                humans[clip] = human_log(clip, files.get(clip, [USDZ / f"{clip}.usdz"]))
            steps = g.pivot_table(index="timestamps_us", columns="name", values="values").sort_index()
            rec = rollout_record(clip, feats[clip], humans[clip], steps)
            if rec:
                rows.append({"model": label, "rollout": rid, **rec})
    df = pd.DataFrame(rows)
    df.to_csv(out / "rollouts.csv", index=False)

    lines = ["| model | group | rollouts | near-stop in zone (human moving) | ... reached <= 3 s behind human | "
             "near-stop elsewhere | zone speed / human (median) | zone speed < 50% of human | did not clear zone |",
             "|---|---|---|---|---|---|---|---|---|"]
    summary = []
    for (m, g), x in df[df.group != "mild"].groupby(["model", "group"]):
        reached = x[x.reached_zone]
        ratio = x.zone_speed_ratio.dropna()
        row = {
            "model": m, "group": g, "n": len(x),
            "near_stop_in_zone": int(x.near_stop_in_zone.sum()),
            "near_stop_in_zone_on_time": int(x.near_stop_in_zone_on_time.sum()),
            "near_stop_outside": int(x.near_stop_outside_zone.sum()),
            "speed_ratio_median": float(ratio.median()) if len(ratio) else None,
            "speed_ratio_lt_half": int((ratio < 0.5).sum()), "speed_ratio_n": len(ratio),
            "not_cleared": int((reached.cleared_zone == False).sum()), "reached": len(reached),
        }
        summary.append(row)
        lines.append(
            f"| {m} | {g} | {row['n']} | {row['near_stop_in_zone']} ({row['near_stop_in_zone'] / row['n']:.0%}) | "
            f"{row['near_stop_in_zone_on_time']} ({row['near_stop_in_zone_on_time'] / row['n']:.0%}) | "
            f"{row['near_stop_outside']} ({row['near_stop_outside'] / row['n']:.0%}) | "
            f"{row['speed_ratio_median']:.2f} | {row['speed_ratio_lt_half']}/{row['speed_ratio_n']} | "
            f"{row['not_cleared']}/{row['reached']} |")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
