"""Scenario attributes for the curated_val clips, in the shape analyze_1606.py wants.

The per-clip CSV joins two kinds of column: what the model did (read straight out
of a run's results-summary.json) and what the scene is (computed here from the
scene's .usdz). Only the second kind needs this script, and it is the same for
every model, so it is computed once and reused for both runs.

The definitions are lifted verbatim from the scans that produced the existing
1606-clip pool, so the values join cleanly onto that file:

  turn      max |yaw(t+4s) - yaw(t)| over the GT rig trajectory, in degrees
  vmed      median GT speed, m/s
  near_mean mean number of vehicles within 30 m of the ego, sampled at 40
            evenly spaced GT timestamps
  mindist   closest any vehicle ever came to the ego on the GT recording, m
            (999.0 when the scene has no vehicle tracks)

Two details that matter for joining. A .usdz file name is not always the scene
id, so the file for each clip is resolved from data/scenes/sim_scenes*.csv and
the internal metadata.yaml scene_id is what gets written out. And the vehicle
classes are the same five used before; widening that set would change near_mean
and mindist for every clip and break comparability with the 1606 file.

Usage:
    python scan_scene_attrs.py --summary runs/<run>/aggregate/results-summary.json \
        --out data/pool441.json
"""
import argparse
import csv
import glob
import json
import os
import pathlib
import zipfile

import numpy as np
import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[2]
SCENE_CSVS = ("data/scenes/sim_scenes.csv", "data/scenes/sim_scenes_2604.csv")
HF_SNAPSHOTS = (
    "/home/kaist5/.cache/huggingface/hub/"
    "datasets--nvidia--PhysicalAI-Autonomous-Vehicles-NuRec/snapshots/*"
)
USDZ_LINK_DIR = "/home/kaist5/Dataset/alpasim/data/nre-artifacts/all-usdzs"


def resolve_paths(summary_path: str) -> dict[str, str]:
    """clip_id -> .usdz path, for every clip a run scored.

    A .usdz file name is not always the scene id -- 2 of the 441 are not -- so
    the scene tables are the authority and the flat symlink directory is only a
    fallback. Together they resolve all 441; either alone does not.
    """
    clips = {
        r["clipgt_id"]
        for r in json.load(open(summary_path))["rollouts"]
    }
    table: dict[str, list[dict]] = {}
    for name in SCENE_CSVS:
        for row in csv.DictReader(open(_ROOT / name)):
            table.setdefault(row["scene_id"], []).append(row)

    snapshots = glob.glob(HF_SNAPSHOTS)
    out, missing = {}, []
    for clip in sorted(clips):
        bare = clip.replace("clipgt-", "")
        found = None
        for row in table.get(clip, []):
            for snap in snapshots:
                candidate = os.path.join(snap, row["path"])
                if os.path.exists(candidate):
                    found = candidate
                    break
            if found:
                break
        if not found:
            link = os.path.join(USDZ_LINK_DIR, f"{bare}.usdz")
            if os.path.exists(link):
                found = os.path.realpath(link)
        if found:
            out[bare] = found
        else:
            missing.append(bare)
    if missing:
        print(f"WARNING: no .usdz for {len(missing)} clips: {missing[:5]}")
    return out

VEHICLE_CLASSES = {"automobile", "heavy_truck", "bus", "other_vehicle", "trailer"}
NEAR_RADIUS_M = 30.0
TURN_WINDOW_S = 4.0
SAMPLES = 40
TRACK_MATCH_US = 200_000


def scan(path: str) -> dict | None:
    """Everything the CSV needs from one scene, or None if the file is unusable."""
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if "rig_trajectories.json" not in names:
            return None
        scene = yaml.safe_load(z.read("metadata.yaml"))["scene_id"]
        rig = json.loads(z.read("rig_trajectories.json"))
        tracks = (
            json.loads(z.read("sequence_tracks.json"))["dummy_chunk_id"]["tracks_data"]
            if "sequence_tracks.json" in names
            else None
        )
        frames = [n for n in names if n.startswith("frames/") and n.endswith((".jpeg", ".jpg"))]
        bright = None
        if frames:
            from io import BytesIO

            from PIL import Image

            vals = [
                np.asarray(Image.open(BytesIO(z.read(f))).convert("L")).mean()
                for f in frames
            ]
            bright = round(float(np.mean(vals)), 1)

    seq = rig["rig_trajectories"][0]
    T = np.asarray(seq["T_rig_worlds"], dtype=np.float64)
    ets = np.asarray(seq["T_rig_world_timestamps_us"], dtype=np.int64)
    if len(T) < 10:
        return None
    exy = T[:, :2, 3]
    yaw = np.unwrap(np.arctan2(T[:, 1, 0], T[:, 0, 0]))
    dt = np.diff(ets) / 1e6
    if np.median(dt) <= 0:
        return None
    step = np.linalg.norm(np.diff(exy, axis=0), axis=1)
    speed = step / np.clip(dt, 1e-6, None)
    win = max(2, int(round(TURN_WINDOW_S / np.median(dt))))
    turn = (
        float(np.abs(yaw[win:] - yaw[:-win]).max() * 180 / np.pi)
        if len(yaw) > win
        else 0.0
    )

    near_mean, mindist, nveh = 0.0, 999.0, 0
    if tracks is not None:
        labels = tracks["tracks_label_class"]
        veh = [j for j, L in enumerate(labels) if str(L) in VEHICLE_CLASSES]
        nveh = len(veh)
        if veh:
            poses, ts = tracks["tracks_poses"], tracks["tracks_timestamps_us"]
            counts = []
            for gi in np.linspace(0, len(ets) - 1, min(SAMPLES, len(ets))).astype(int):
                t0, e, n = ets[gi], exy[gi], 0
                for j in veh:
                    tj = np.asarray(ts[j], dtype=np.int64)
                    if len(tj) == 0:
                        continue
                    k = int(np.argmin(np.abs(tj - t0)))
                    if abs(tj[k] - t0) > TRACK_MATCH_US:
                        continue
                    p = np.asarray(poses[j][k], dtype=np.float64)
                    pxy = p[:2, 3] if p.ndim == 2 and p.shape[0] >= 3 else np.asarray(p[:2])
                    d = float(np.linalg.norm(pxy - e))
                    mindist = min(mindist, d)
                    if d <= NEAR_RADIUS_M:
                        n += 1
                counts.append(n)
            near_mean = float(np.mean(counts))

    return dict(
        id=str(scene).replace("clipgt-", ""),
        turn=turn,
        vmed=float(np.median(speed)),
        near_mean=near_mean,
        mindist=mindist,
        nveh=nveh,
        brightness=bright,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--summary", help="results-summary.json; paths are resolved from it")
    g.add_argument("--paths", help="JSON: clip_id -> .usdz path, if already resolved")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    paths = json.load(open(a.paths)) if a.paths else resolve_paths(a.summary)
    print(f"{len(paths)} scenes to scan")
    out, failed = {}, []
    for i, (cid, path) in enumerate(sorted(paths.items()), 1):
        try:
            rec = scan(path)
        except Exception as exc:  # a truncated or unreadable archive
            rec, exc_note = None, f"{type(exc).__name__}: {exc}"
        else:
            exc_note = "no rig trajectory / too few poses"
        if rec is None:
            failed.append((cid, exc_note))
            continue
        # metadata.yaml is authoritative for the id, but warn if it disagrees.
        if rec["id"] != cid:
            rec["id_from_metadata"] = rec["id"]
            rec["id"] = cid
        out[cid] = rec
        if i % 50 == 0:
            print(f"  {i}/{len(paths)}", flush=True)

    json.dump(out, open(a.out, "w"))
    print(f"{len(out)}/{len(paths)} scanned -> {a.out}")
    for cid, why in failed:
        print(f"  FAILED {cid}: {why}")


if __name__ == "__main__":
    main()
