# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""Deferred, observational rear-approach analysis; never modify the 441 run.

Read only pose records for population analysis. Camera payloads are loaded only
for six selected clips, after evaluation and leaderboard fitting have finished.
The historical axe-v9 ASLs were deleted by its original launcher, so replay that
small subset with the immutable, documented axe-v9 image for paired footage.
"""

import argparse
import csv
import fcntl
import io
import json
import os
import socket
import struct
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
from alpasim_grpc.v0.logging_pb2 import LogEntry
from google.protobuf.message import DecodeError

ROOT = Path(__file__).resolve().parents[2]
NEW = ROOT / "runs/leaderboard-stage3-5cam-ep05-20261007"
OLD = ROOT / "runs/leaderboard-merged-route-ep30"
FIT = ROOT / "runs/leaderboard-35-with-stage3-5cam-ep05/local_leaderboard.csv"
OUT = ROOT.parent.parent / "stage3_5cam_rear_response_20261007"
REPLAY_NAME = "rear-response-axe-v9-replay-20261007"
REPLAY = ROOT / "runs" / REPLAY_NAME
IMAGE = "sha256:85134c1ea9f09d140be610ca063c50ec60e99980c392e5bf81fe6563af503765"
CHECKPOINT = "364c801e397e9d6e36ea1f9027dc4ca804ad2e429bb588cf4e84185ca851de3d"
PREFIX = "axe-rear-response-v9"
CAMERAS = [
    "camera_front_wide_120fov",
    "camera_rear_left_70fov",
    "camera_rear_right_70fov",
]
VEHICLES = {"automobile", "car", "truck", "bus", "van", "vehicle"}
KST = ZoneInfo("Asia/Seoul")


def log(message):
    print(f"[{datetime.now(KST):%F %T}] {message}", flush=True)


def write_json(path, data):
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )


def write_csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8-sig")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def messages(path, fields):
    """Skip unrelated protobuf envelopes with seek, including embedded PNGs."""
    size = path.stat().st_size
    with path.open("rb") as f:
        while f.tell() < size:
            prefix = f.read(4)
            if len(prefix) != 4:
                raise ValueError(f"Truncated ASL prefix: {path}")
            n = struct.unpack(">I", prefix)[0]
            end = f.tell() + n
            if n == 0 or end > size:
                raise ValueError(f"Truncated/empty ASL envelope: {path}")
            key_bytes = bytearray()
            key = shift = 0
            while True:
                b = f.read(1)
                if not b or len(key_bytes) == 10:
                    raise ValueError(f"Malformed protobuf key: {path}")
                key_bytes.extend(b)
                key |= (b[0] & 127) << shift
                if not b[0] & 128:
                    break
                shift += 7
            if key >> 3 in fields:
                yield LogEntry.FromString(bytes(key_bytes) + f.read(end - f.tell()))
            else:
                f.seek(end)


def pose_xyh(p):
    q = p.quat
    yaw = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
    return [p.vec.x, p.vec.y, yaw]


def smooth(values):
    return np.convolve(np.pad(values, (2, 2), mode="edge"), np.ones(5) / 5, "valid")


def read_trace(path):
    actors = defaultdict(list)
    metadata = None
    for m in messages(path, {1, 2}):
        kind = m.WhichOneof("log_entry")
        if kind == "rollout_metadata":
            metadata = m.rollout_metadata
        elif kind == "actor_poses":
            for actor in m.actor_poses.actor_poses:
                actors[actor.actor_id].append(
                    [m.actor_poses.timestamp_us, *pose_xyh(actor.actor_pose)]
                )
    if metadata is None or len(actors["EGO"]) < 3:
        raise ValueError(f"Missing metadata/ego poses: {path}")
    definitions = {d.actor_id: d for d in metadata.actor_definitions.actor_aabb}
    ego = np.asarray(actors["EGO"], dtype=float)
    ego = ego[np.unique(ego[:, 0], return_index=True)[1]]
    start = metadata.session_metadata.start_timestamp_us
    times = (ego[:, 0] - start) / 1e6
    forward = np.column_stack((np.cos(ego[:, 3]), np.sin(ego[:, 3])))
    velocity = np.gradient(ego[:, 1:3], times, axis=0)
    speed = smooth(np.sum(velocity * forward, axis=1))
    traffic = {}
    for actor_id, rows in actors.items():
        if actor_id == "EGO" or actor_id not in definitions or len(rows) < 3:
            continue
        if definitions[actor_id].actor_label.lower() not in VEHICLES:
            continue
        a = np.asarray(rows, dtype=float)
        a = a[np.unique(a[:, 0], return_index=True)[1]]
        at = (a[:, 0] - start) / 1e6
        present = (times >= at[0]) & (times <= at[-1])
        xy = np.column_stack([np.interp(times, at, a[:, k]) for k in [1, 2]])
        yaw = np.interp(times, at, np.unwrap(a[:, 3]))
        av = np.column_stack(
            [np.interp(times, at, v) for v in np.gradient(a[:, 1:3], at, axis=0).T]
        )
        rel = xy - ego[:, 1:3]
        x = np.sum(rel * forward, axis=1)
        y = -rel[:, 0] * forward[:, 1] + rel[:, 1] * forward[:, 0]
        d = definitions[actor_id].aabb
        ed = definitions["EGO"].aabb
        delta = yaw - ego[:, 3]
        longitudinal_extent = (
            ed.size_x + abs(np.cos(delta)) * d.size_x + abs(np.sin(delta)) * d.size_y
        ) / 2
        lateral_limit = (
            ed.size_y + abs(np.cos(delta)) * d.size_y + abs(np.sin(delta)) * d.size_x
        ) / 2 + 0.75
        aligned = np.cos(delta) >= np.cos(np.deg2rad(45))
        corridor = present & aligned & (abs(y) <= lateral_limit)
        gap = -x - longitudinal_extent
        closing = smooth(np.sum((av - velocity) * forward, axis=1))
        ttc = np.full(len(times), np.inf)
        np.divide(gap, closing, out=ttc, where=(gap > 0) & (closing > 0))
        traffic[actor_id] = {
            "gap": gap,
            "closing": closing,
            "ttc": ttc,
            "corridor": corridor,
            "front_gap": x - longitudinal_extent,
            "x": x,
            "y": y,
        }
    gt = metadata.ego_rig_recorded_ground_truth_trajectory.poses
    gt_data = np.asarray([[p.timestamp_us, *pose_xyh(p.pose)] for p in gt], dtype=float)
    gt_times = (gt_data[:, 0] - start) / 1e6
    gt_xy = np.column_stack([np.interp(times, gt_times, gt_data[:, k]) for k in [1, 2]])
    gt_velocity = np.gradient(gt_xy, times, axis=0)
    gt_speed = smooth(np.linalg.norm(gt_velocity, axis=1))
    gt_yaw = [
        pose_xyh(p.pose)[2] for p in gt if ego[0, 0] <= p.timestamp_us <= ego[-1, 0]
    ]
    heading_span = (
        float(np.rad2deg(np.ptp(np.unwrap(gt_yaw)))) if len(gt_yaw) > 1 else None
    )
    stop = float("inf")
    rear_time = None
    metric_file = path.with_name("metrics.parquet")
    if metric_file.exists():
        df = pl.read_parquet(metric_file).filter(
            pl.col("valid") & (pl.col("values") > 0)
        )
        failures = df.filter(
            pl.col("name").is_in(
                ["collision_any", "offroad", "left_corridor_laterally"]
            )
        )
        if failures.height:
            stop = (failures["timestamps_us"].min() - start) / 1e6
        rear = df.filter(pl.col("name") == "collision_rear")
        if rear.height:
            rear_time = float((rear["timestamps_us"].min() - start) / 1e6)
    return {
        "times": times,
        "ego": ego,
        "speed": speed,
        "gt_speed": gt_speed,
        "traffic": traffic,
        "force_gt_s": metadata.force_gt_duration / 1e6,
        "stop_s": stop,
        "rear_collision_s": rear_time,
        "heading_span_deg": heading_span,
        "asl": path,
        "start_us": start,
    }


def approach_event(trace, max_ttc=5.0, max_gap=50.0):
    """First persistent, same-direction vehicle closing from the rear."""
    events = []
    t = trace["times"]
    for actor_id, a in trace["traffic"].items():
        mask = (
            a["corridor"]
            & (a["gap"] > 0)
            & (a["gap"] <= max_gap)
            & (a["closing"] >= 1.0)
            & (a["ttc"] <= max_ttc)
            & (t >= trace["force_gt_s"] + 0.5)
            & (t < trace["stop_s"])
        )
        indices = np.flatnonzero(mask)
        groups = np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1)
        for group in groups:
            if len(group) and t[group[-1]] - t[group[0]] >= 0.5 - 1e-6:
                i = group[0]
                events.append(
                    {
                        "actor_id": actor_id,
                        "onset_s": float(t[i]),
                        "gap_m": float(a["gap"][i]),
                        "closing_mps": float(a["closing"][i]),
                        "ttc_s": float(a["ttc"][i]),
                    }
                )
                break
    return min(events, key=lambda e: (e["onset_s"], e["actor_id"])) if events else None


def response(trace, event):
    empty = {
        "response_observed": False,
        "speed_change_mps": None,
        "accelerated_1mps": None,
        "front_gap_at_onset_m": None,
        "front_blocked_10m": None,
        "gt_speed_change_mps": None,
        "speed_change_minus_gt_mps": None,
    }
    if event is None:
        return {**empty, "response_status": "no qualifying rear approach"}
    t, onset = trace["times"], event["onset_s"]
    before = (t >= onset - 0.5) & (t < onset)
    after = (t >= onset + 1) & (t <= onset + 2)
    if (
        t[-1] < onset + 2
        or trace["stop_s"] <= onset + 2
        or not before.any()
        or not after.any()
    ):
        return {
            **empty,
            "response_status": "censored by collision/offroad/corridor/end before 2-second response",
        }
    i = int(np.argmin(abs(t - onset)))
    front = [
        a["front_gap"][i]
        for a in trace["traffic"].values()
        if a["corridor"][i] and 0 < a["front_gap"][i] <= 50
    ]
    front_gap = float(min(front)) if front else None
    delta = float(trace["speed"][after].mean() - trace["speed"][before].mean())
    gt_delta = None
    if "gt_speed" in trace:
        gt_delta = float(
            trace["gt_speed"][after].mean() - trace["gt_speed"][before].mean()
        )
    return {
        "response_observed": True,
        "speed_change_mps": delta,
        "accelerated_1mps": delta >= 1,
        "front_gap_at_onset_m": front_gap,
        "front_blocked_10m": front_gap is not None and front_gap < 10,
        "response_status": "observed",
        "gt_speed_change_mps": gt_delta,
        "speed_change_minus_gt_mps": delta - gt_delta if gt_delta is not None else None,
    }


def summary_rows(run):
    rows = json.loads((run / "aggregate/results-summary.json").read_text())["rollouts"]
    by_id = {r["clipgt_id"]: r for r in rows}
    if len(rows) != len(by_id):
        raise ValueError("Requires one rollout per clip")
    return by_id


def metric(row, name):
    value = (row.get("metrics") or {}).get(name)
    return float(value) if value is not None else None


def paired_rows(old, new):
    if len(old) != 441 or set(old) != set(new):
        raise ValueError("Requires the same 441 clips in both summaries")
    result = []
    for clip in sorted(old):
        a, b = old[clip], new[clip]
        ar, br = metric(a, "collision_rear"), metric(b, "collision_rear")
        # Physical failures such as offroad are legitimate measured outcomes.
        # Only missing metrics/RPC errors prevent collision comparison.
        comparable = (
            ar is not None
            and br is not None
            and not (a.get("metrics") or {}).get("error")
            and not (b.get("metrics") or {}).get("error")
        )
        transition = "unavailable due to evaluation failure/missing metric"
        if comparable:
            transition = {
                (False, False): "neither rear collision",
                (True, False): "axe-v9 rear collision avoided",
                (True, True): "rear collision in both",
                (False, True): "new rear collision",
            }[(ar > 0, br > 0)]
        safe = comparable and all(
            metric(b, k) == 0
            for k in ["collision_at_fault", "offroad", "left_corridor_laterally"]
        )
        result.append(
            {
                "clip_id": clip.removeprefix("clipgt-"),
                "clipgt_id": clip,
                "axe_v9_score": a["score"],
                "five_camera_score": b["score"],
                "score_change": b["score"] - a["score"],
                "axe_v9_rear_collision": ar,
                "five_camera_rear_collision": br,
                "collision_transition": transition,
                "avoided_without_other_hard_failure": safe and ar > 0 and br == 0,
                "axe_v9_failure_reason": a.get("failure_reason"),
                "five_camera_failure_reason": b.get("failure_reason"),
            }
        )
        for name in [
            "collision_at_fault",
            "offroad",
            "left_corridor_laterally",
            "progress_clipped_rel",
        ]:
            result[-1][f"axe_v9_{name}"] = metric(a, name)
            result[-1][f"five_camera_{name}"] = metric(b, name)
    return result


def asl_for(row, run):
    path = run / "rollouts" / row["clipgt_id"] / row["rollout_id"] / "rollout.asl"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def population(old, new, output):
    rows = paired_rows(old, new)
    for index, row in enumerate(rows):
        clip = row["clipgt_id"]
        try:
            trace = read_trace(asl_for(new[clip], NEW))
            event = approach_event(trace)
            row.update(response(trace, event))
            row.update(
                {
                    "rear_approach": event is not None,
                    "approach_actor_id": event["actor_id"] if event else None,
                    "approach_onset_s": event["onset_s"] if event else None,
                    "gap_at_onset_m": event["gap_m"] if event else None,
                    "closing_at_onset_mps": event["closing_mps"] if event else None,
                    "ttc_at_onset_s": event["ttc_s"] if event else None,
                    "gt_heading_span_deg": trace["heading_span_deg"],
                    "rear_collision_time_s": trace["rear_collision_s"],
                    "rear_approach_ttc3_gap30": approach_event(trace, 3, 30)
                    is not None,
                }
            )
        except (
            OSError,
            ValueError,
            KeyError,
            DecodeError,
            pl.exceptions.PolarsError,
        ) as exc:
            row.update({"trace_error": str(exc), "response_observed": False})
        if index % 48 == 0:
            log(f"Population traces {index + 1}/441")
    write_csv(output / "all_441_paired_results.csv", rows)
    selected = []
    for category in [
        "axe-v9 rear collision avoided",
        "rear collision in both",
        "new rear collision",
    ]:
        candidates = [
            r
            for r in rows
            if r["collision_transition"] == category
            and r.get("rear_approach")
            and (
                category != "axe-v9 rear collision avoided"
                or r["avoided_without_other_hard_failure"]
            )
        ]
        selected.extend(candidates[:2])
    fallback = [
        r
        for r in rows
        if r.get("rear_approach")
        and r["clipgt_id"] not in {x["clipgt_id"] for x in selected}
    ]
    selected.extend(fallback[: max(0, 6 - len(selected))])
    write_csv(output / "selected_cases.csv", selected)
    (output / "selected_clipgt_ids.txt").write_text(
        "".join(r["clipgt_id"] + "\n" for r in selected)
    )
    observed = [r for r in rows if r.get("response_observed")]
    count = lambda name: sum(r.get(name) is True for r in rows)
    data = {
        "clips": 441,
        "axe_v9_rear_collisions": sum(
            (metric(r, "collision_rear") or 0) > 0 for r in old.values()
        ),
        "five_camera_rear_collisions": sum(
            (metric(r, "collision_rear") or 0) > 0 for r in new.values()
        ),
        "transitions": {
            c: sum(r["collision_transition"] == c for r in rows)
            for c in sorted({r["collision_transition"] for r in rows})
        },
        "rear_approach_clips": count("rear_approach"),
        "response_observed_clips": len(observed),
        "accelerated_at_least_1mps_clips": count("accelerated_1mps"),
        "accelerated_percent_of_observable_approaches": (
            100 * count("accelerated_1mps") / len(observed) if observed else None
        ),
        "avoided_without_other_hard_failure": count(
            "avoided_without_other_hard_failure"
        ),
        "trace_error_clips": sum(bool(r.get("trace_error")) for r in rows),
        "case_selection": "Up to two per collision transition, UUID order, with qualifying approach; avoided cases require no other hard failure. Fill to six in UUID order. Case selection is descriptive, not an unbiased acceleration estimate.",
        "axe_v9_missing_rear_metric_clips": sum(
            metric(r, "collision_rear") is None for r in old.values()
        ),
        "five_camera_missing_rear_metric_clips": sum(
            metric(r, "collision_rear") is None for r in new.values()
        ),
        "causal_claim": False,
    }
    write_json(output / "population_summary.json", data)
    return rows, selected, data


def plots(rows, selected, data, output):
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    counts = [data["axe_v9_rear_collisions"], data["five_camera_rear_collisions"]]
    bars = ax.bar(
        ["axe-v9", "stage3-5cam-ep05"],
        np.asarray(counts) / 441 * 100,
        color=["#666666", "#2377b8"],
    )
    for bar, n in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{n}/441 ({n / 441:.1%})",
            ha="center",
        )
    ax.set_ylabel("Clips with rear collision (%)")
    ax.set_ylim(0, max(counts) / 441 * 100 + 6)
    ax.set_title("Same 441 clips; original model configurations")
    fig.tight_layout()
    fig.savefig(output / "01_rear_collision_comparison.png", dpi=180)
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, blocked, title in zip(
        axs,
        [False, True],
        ["Front clearance >= 10m / no front vehicle", "Front vehicle within 10m"],
    ):
        vals = [
            r["speed_change_mps"]
            for r in rows
            if r.get("response_observed") and r["front_blocked_10m"] == blocked
        ]
        lower = min(-8, np.floor(min(vals))) if vals else -8
        upper = max(8, np.ceil(max(vals))) if vals else 8
        ax.hist(vals, bins=np.arange(lower, upper + 0.5, 0.5), color="#2377b8")
        ax.axvline(1, color="#a33333", linestyle="--", label="+1 m/s threshold")
        ax.set_title(f"{title}\nN = {len(vals)}")
        ax.set_xlabel("Ego speed change after approach (m/s)")
        ax.set_ylabel("Clips")
        ax.legend()
    fig.suptitle("5-camera observed response; collision/end-censored windows excluded")
    fig.tight_layout()
    fig.savefig(output / "02_approach_speed_change.png", dpi=180)
    plt.close(fig)


def gpu_idle():
    rows = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            "4,5",
            "--query-gpu=memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()
    return len(rows) == 2 and all(
        int(a.strip()) < 4000 and int(b.strip()) <= 5
        for a, b in (r.split(",") for r in rows)
    )


def cleanup_replay():
    drivers = subprocess.check_output(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"name=^{PREFIX}-g[45]-0$",
            "--format",
            "{{.Names}}",
        ],
        text=True,
    ).splitlines()
    services = subprocess.check_output(
        [
            "docker",
            "ps",
            "-a",
            "--filter",
            f"label=com.docker.compose.project={REPLAY_NAME}",
            "--format",
            "{{.Names}}",
        ],
        text=True,
    ).splitlines()
    names = sorted(set(drivers + services))
    if names:
        log("Stopping only rear-response replay containers: " + ", ".join(names))
        subprocess.run(["docker", "stop", "-t", "10", *names], check=True)
        subprocess.run(["docker", "rm", *names], check=True)


def replay(selected, output):
    if not selected:
        return {}
    if (REPLAY / "aggregate/results-summary.json").exists():
        rows = summary_rows(REPLAY)
        if set(rows) != {r["clipgt_id"] for r in selected}:
            raise ValueError("Existing replay has different clips")
        return rows
    image = json.loads(
        subprocess.check_output(["docker", "image", "inspect", IMAGE], text=True)
    )[0]
    if image["Config"]["Labels"]["org.alpasim.checkpoint.sha256"] != CHECKPOINT:
        raise ValueError("axe-v9 immutable image checkpoint differs")
    for name in [f"{PREFIX}-g4-0", f"{PREFIX}-g5-0"]:
        exists = (
            subprocess.run(
                ["docker", "container", "inspect", name],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            ).returncode
            == 0
        )
        if exists:
            raise ValueError(f"Refusing to replace existing container {name}")
    idle_checks = 0
    while idle_checks < 2:
        idle_checks = idle_checks + 1 if gpu_idle() else 0
        if idle_checks < 2:
            log(
                "Waiting for free GPUs 4,5 for the six-clip axe-v9 replay; leaving all existing jobs alone"
            )
            time.sleep(30)
    for port in [7260, 7261, *range(22000, 22040)]:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    REPLAY.mkdir(parents=True, exist_ok=True)
    write_json(
        output / "replay_provenance.json",
        {
            "image_id": IMAGE,
            "checkpoint_sha256": CHECKPOINT,
            "clips": [r["clipgt_id"] for r in selected],
            "preset": "dev",
            "mpc_lat_lon_idx": [1, 0.25, 3],
            "purpose": "Recover unavailable historical trajectories for descriptive paired videos, not replace original leaderboard scores",
        },
    )
    env = {
        **os.environ,
        "UV_OFFLINE": "1",
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "IMAGE": IMAGE,
        "EXPECTED_CHECKPOINT_SHA256": CHECKPOINT,
        "OFFICIAL_ENV_ONLY": "1",
        "CONTAINER_PREFIX": PREFIX,
        "BASE_PORT": "7260",
        "GPU_INDICES_CSV": "4,5",
        "REPLICAS_PER_GPU": "1",
        "READY_TIMEOUT_SEC": "1800",
    }
    try:
        with (output / "baseline_driver_start.log").open("w+") as stream:
            subprocess.run(
                [str(ROOT / "e2e_challenge/axe_local_eval/start_drivers.sh")],
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
            stream.seek(0)
            addresses = stream.read().strip().splitlines()[-1]
        if not addresses.startswith("["):
            raise ValueError("Missing baseline driver addresses")
        env.update(
            {
                "RUN_NAME": REPLAY_NAME,
                "RUN_DIR": str(REPLAY),
                "PRESET": "dev",
                "CONTESTANT_IMAGE": IMAGE,
                "DRIVER_ADDRESSES": addresses,
                "SCENE_IDS_FILE": str(output / "selected_clipgt_ids.txt"),
                "SCENE_LIMIT": "0",
                "N_ROLLOUTS": "1",
                "ROLLOUT_WORKERS": "2",
                "RENDER_GPUS_CSV": "4,5",
                "RENDERER_REPLICAS_PER_GPU": "1",
                "NRE_CACHE_SIZE": "1",
                "RENDER_VIDEO": "false",
                "KEEP_ROLLOUTS": "1",
                "ENABLE_AUTORESUME": "true",
                "SERVICE_STARTUP_TIMEOUT_SEC": "1800",
                "FAST_STARTUP": "1",
                "RENDER_CACHE_WARM": "0",
                "ALPASIM_IMAGE": "nvcr.io/nvidia/nre/nre-ga:26.04",
                "NRE_IMAGE": "nvcr.io/nvidia/nre/nre-ga:26.04",
                "DRIVER_CONCURRENT_ROLLOUTS": "1",
                "MPC_OVERRIDES": "controller.gains.long_position_weight=0.25 controller.gains.lat_position_weight=1.0 controller.gains.idx_start_penalty=3",
                "EXTRA_OVERRIDES": "wizard.baseport=22000",
            }
        )
        log(
            f"Replaying {len(selected)} selected axe-v9 clips; original scoring runs unchanged"
        )
        with (ROOT / "runs" / f"{REPLAY_NAME}.progress.log").open("a") as stream:
            subprocess.run(
                [str(ROOT / "e2e_challenge/axe_local_eval/run_curated_val.sh")],
                cwd=ROOT,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=True,
            )
    finally:
        cleanup_replay()
    rows = summary_rows(REPLAY)
    if set(rows) != {r["clipgt_id"] for r in selected}:
        raise ValueError("Incomplete selected baseline replay")
    return rows


def cameras(trace):
    result = defaultdict(list)
    for m in messages(trace["asl"], {11}):
        img = m.driver_camera_image.camera_image
        if img.logical_id in CAMERAS:
            result[img.logical_id].append(
                ((img.frame_end_us - trace["start_us"]) / 1e6, img.image_bytes)
            )
    return {k: sorted(v, key=lambda a: a[0]) for k, v in result.items()}


def case_assets(clip, old_row, new_row, historical_row, output):
    import matplotlib

    matplotlib.use("Agg")
    import imageio.v2 as imageio
    from matplotlib import pyplot as plt
    from PIL import Image, ImageDraw, ImageFont

    a, b = read_trace(asl_for(old_row, REPLAY)), read_trace(asl_for(new_row, NEW))
    ae, be = approach_event(a), approach_event(b)
    anchor = (
        min([e for e in [ae, be] if e], key=lambda e: e["onset_s"])
        if ae or be
        else None
    )
    result = {
        "clipgt_id": clip,
        "historical_axe_v9_score": historical_row["score"],
        "replayed_axe_v9_score": old_row["score"],
        "five_camera_score": new_row["score"],
        "replay_score_change_from_historical": old_row["score"]
        - historical_row["score"],
        "anchor_onset_s": anchor["onset_s"] if anchor else None,
        "anchor_actor_id": anchor["actor_id"] if anchor else None,
        "axe_v9_approach_present": ae is not None,
        "five_camera_approach_present": be is not None,
        "same_approach_vehicle_in_both": bool(
            ae and be and ae["actor_id"] == be["actor_id"]
        ),
    }
    for name, trace in [("axe_v9", a), ("five_camera", b)]:
        result.update({f"{name}_{k}": v for k, v in response(trace, anchor).items()})
        actor = trace["traffic"].get(anchor["actor_id"]) if anchor else None
        i = int(np.argmin(abs(trace["times"] - anchor["onset_s"]))) if anchor else 0
        result[f"{name}_common_vehicle_behind_at_anchor"] = bool(
            actor and actor["corridor"][i] and actor["gap"][i] > 0
        )
    case = output / "cases" / clip.removeprefix("clipgt-")
    case.mkdir(parents=True, exist_ok=True)
    timeseries = []
    for name, trace in [("axe-v9 replay", a), ("stage3-5cam-ep05", b)]:
        actor = trace["traffic"].get(anchor["actor_id"]) if anchor else None
        for i, t in enumerate(trace["times"]):
            behind = actor is not None and actor["corridor"][i] and actor["x"][i] < 0
            timeseries.append(
                {
                    "subject": name,
                    "time_s": float(t),
                    "ego_speed_mps": float(trace["speed"][i]),
                    "rear_gap_m": float(actor["gap"][i]) if behind else None,
                    "closing_speed_mps": float(actor["closing"][i]) if behind else None,
                    "rear_ttc_s": (
                        float(actor["ttc"][i])
                        if behind and np.isfinite(actor["ttc"][i])
                        else None
                    ),
                }
            )
    write_csv(case / "time_series.csv", timeseries)
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for name, trace, color in [
        ("axe-v9 replay", a, "#555555"),
        ("stage3-5cam-ep05", b, "#2377b8"),
    ]:
        axs[0].plot(trace["times"], trace["speed"] * 3.6, color=color, label=name)
        actor = trace["traffic"].get(anchor["actor_id"]) if anchor else None
        if actor:
            valid = actor["corridor"] & (actor["x"] < 0)
            axs[1].plot(
                trace["times"], np.where(valid, actor["gap"], np.nan), color=color
            )
            axs[2].plot(
                trace["times"], np.where(valid, actor["closing"], np.nan), color=color
            )
        if trace["rear_collision_s"] is not None:
            for ax in axs:
                ax.axvline(trace["rear_collision_s"], color=color, linestyle=":")
    for ax in axs:
        if anchor:
            ax.axvline(anchor["onset_s"], color="#a33333", linestyle="--")
        ax.grid(alpha=0.2)
    axs[0].set_ylabel("Ego speed (km/h)")
    axs[0].plot(
        b["times"],
        b["gt_speed"] * 3.6,
        color="#999999",
        linestyle="--",
        label="Recorded GT speed",
    )
    axs[0].legend()
    axs[1].set_ylabel("Rear bumper gap (m)")
    axs[2].set_ylabel("Closing speed (m/s)")
    axs[2].set_xlabel("Simulation time (s)")
    fig.suptitle(clip + "\nDashed: common approach anchor; dotted: rear collision")
    fig.tight_layout()
    fig.savefig(case / "speed_gap_comparison.png", dpi=160)
    plt.close(fig)
    frames = [cameras(a), cameras(b)]
    font = ImageFont.truetype("DejaVuSans.ttf", 19)
    indices = [
        {cam: np.asarray([x[0] for x in streams.get(cam, [])]) for cam in CAMERAS}
        for streams in frames
    ]
    movie = case / "paired_front_rear.mp4"
    with imageio.get_writer(
        movie, fps=5, codec="libx264", quality=6, ffmpeg_params=["-threads", "2"]
    ) as writer:
        for t in np.arange(0, max(a["times"][-1], b["times"][-1]) + 0.05, 0.1):
            canvas = Image.new("RGB", (1536, 704), "#171d25")
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (12, 8), f"{clip} | t={t:.1f}s | 0.5x playback", font=font, fill="white"
            )
            for row, (trace, streams) in enumerate(zip([a, b], frames)):
                label = (
                    "axe-v9 replay (3 model cameras)"
                    if row == 0
                    else "stage3-5cam-ep05 (5 model cameras)"
                )
                speed = np.interp(t, trace["times"], trace["speed"]) * 3.6
                ended = t > trace["times"][-1] + 0.1
                draw.text(
                    (12, 40 + row * 320),
                    f"{label} | ego {speed:.1f} km/h"
                    + (" | rollout ended" if ended else ""),
                    font=font,
                    fill="white",
                )
                for col, cam in enumerate(CAMERAS):
                    ts = indices[row][cam]
                    if len(ts):
                        i = int(np.argmin(abs(ts - t)))
                        if abs(ts[i] - t) < 0.15 and not ended:
                            image = (
                                Image.open(io.BytesIO(streams[cam][i][1]))
                                .convert("RGB")
                                .resize((512, 256))
                            )
                            canvas.paste(image, (col * 512, 76 + row * 320))
                    camera_label = ["front", "rear left", "rear right"][col]
                    if row == 0 and col > 0:
                        camera_label += " (recorded; not model input)"
                    draw.text(
                        (col * 512 + 8, 332 + row * 320),
                        camera_label,
                        font=font,
                        fill="white",
                    )
            writer.append_data(np.asarray(canvas))
    result["video_path"] = str(movie)
    result["plot_path"] = str(case / "speed_gap_comparison.png")
    return result


def report(data, cases, output, replay_error=None):
    lines = [
        "후방 차량 접근 시 자차 가속 반응 분석",
        "",
        "연구 질문: 상황에 따라 후방 입력이 필요한지, 후방 접근 상황의 관찰된 행동으로 확인한다.",
        "",
        f"전체 441개 후방 충돌: axe-v9 {data['axe_v9_rear_collisions']}개 → 5카메라 {data['five_camera_rear_collisions']}개.",
        f"후방 충돌 지표가 없는 평가 오류 클립: axe-v9 {data['axe_v9_missing_rear_metric_clips']}개 / 5카메라 {data['five_camera_missing_rear_metric_clips']}개. 충돌 미상은 무충돌로 분류하지 않는다.",
        f"후방 충돌이 사라지고 책임 충돌/도로 이탈/corridor 이탈도 없는 클립: {data['avoided_without_other_hard_failure']}개.",
        f"5카메라의 후방 접근 클립 {data['rear_approach_clips']}개 중 충돌/종료 전 2초 반응 관찰 가능 {data['response_observed_clips']}개, 속도 +1m/s 이상 {data['accelerated_at_least_1mps_clips']}개.",
        "",
        "후방 접근은 같은 방향(45도 이내) 차량이 자차 뒤의 차폭 기반 통로에 있고, 범퍼 간 거리 0–50m, 접근 상대속도 ≥1m/s, 현재 속도 기반 TTC ≤5초가 0.5초 이상 지속되는 경우다. 자차 초기 GT 강제 주행 및 직후 0.5초는 제외한다. 실제 차선 판정이나 충돌 확률은 아니다. TTC≤3초/거리≤30m 기준도 CSV에 병기한다.",
        "",
        "속도 반응 = 접근 직전 0.5초 평균 속도 대비 접근 후 1–2초 평균 속도의 차이. 위치 미분 속도에 0.5초 이동 평균을 적용한다. 반응 창 내 충돌/도로 이탈/corridor 이탈/주행 종료가 있으면 관찰 불가로 분리한다. 앞차 10m 이내와 전방 여유를 나눠 그래프로 표시한다.",
        "CSV에는 같은 시점의 기록 GT 속도 변화와 그 차이도 포함한다. 커브 탈출 등 원래 속도가 증가하는 구간인지 확인하는 참고치이며 GT도 후방 차량을 관찰한 실제 운전자 기록이므로 후방 카메라가 없는 대조군으로 간주하지 않는다.",
        "",
        "전체 비교는 과거 441개 원본 점수/충돌 기록을 사용한다. 영상용 axe-v9 재실행은 선택한 최대 6개만이며 원본 점수를 대체하지 않는다. 재실행 점수가 과거와 다르면 selected_case_comparison.csv에 차이를 공개한다. 대표 사례는 충돌 개선/유지/새 충돌에서 UUID 순으로 최대 2개씩 고르고, 후방 접근 사례로 부족분을 채운다. 이 사례들의 비율을 전체 성향으로 일반화하지 않는다.",
        "",
        "대표 사례 그래프는 두 모델 중 먼저 검출된 접근 시각/같은 차량 ID를 공통 기준으로 사용한다. 두 모델의 주행이 달라 접근 조건도 달라질 수 있으므로 CSV의 각 모델 접근 여부를 함께 확인해야 한다.",
        "",
        "이 결과는 모델 사이의 행동 차이와 접근-가속의 시간적 관계를 보여 준다. 학습 가중치/BEV/임계값 등도 달라 후방 카메라 자체의 인과 효과나 차량의 의도를 입증하지 않는다. 이를 분리하려면 같은 5카메라 체크포인트의 후방 입력만 바꾸는 별도 대조 실험이 필요하다.",
        "",
        f"상세 로그를 읽지 못한 클립: {data['trace_error_clips']}개. 평가 오류와 관찰 불가 클립은 all_441_paired_results.csv에 유지한다.",
        "",
        "파일: all_441_paired_results.csv, population_summary.json, 01_rear_collision_comparison.png, 02_approach_speed_change.png, selected_case_comparison.csv, cases/<clip_id>/의 MP4/PNG/time_series.csv.",
    ]
    if replay_error:
        lines += ["", "대표 사례 재실행/영상 생성 오류: " + replay_error]
    (output / "REPORT.txt").write_text("\n".join(lines) + "\n")


def wait_until_finished(output):
    log(
        "Waiting for complete 441 summary, CPU leaderboard, and evaluation container release"
    )
    while True:
        if (NEW / "aggregate/results-summary.json").exists() and FIT.exists():
            names = subprocess.check_output(
                [
                    "docker",
                    "ps",
                    "-q",
                    "--filter",
                    "name=^axe-5cam-ep05-drv-g[0-7]-[0-5]$",
                ],
                text=True,
            ).strip()
            if not names:
                if len(summary_rows(NEW)) != 441:
                    raise ValueError("Final evaluation summary incomplete")
                return
        write_json(
            output / "status.json",
            {
                "status": "waiting_for_evaluation_and_leaderboard",
                "checked_at_kst": datetime.now(KST).isoformat(),
            },
        )
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--no-replay", action="store_true")
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.check:
        import imageio_ffmpeg

        baseline = summary_rows(OLD)
        assert len(baseline) == 441
        assert (
            sum((metric(r, "collision_rear") or 0) > 0 for r in baseline.values()) == 87
        )
        sample = next((NEW / "rollouts").glob("*/*/_complete")).with_name("rollout.asl")
        trace = read_trace(sample)
        log(
            f"PASS: axe-v9 441/87 rear collisions; pose reader {len(trace['times'])} timestamps/{len(trace['traffic'])} vehicles; ffmpeg {imageio_ffmpeg.get_ffmpeg_exe()}"
        )
        return
    if (output / "_complete").exists():
        log("Analysis already complete; outputs preserved")
        return
    lock_handle = (ROOT / ".cache/rear-response-20261007.lock").open("a")
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    (output / "analysis.pid").write_text(str(os.getpid()) + "\n")
    if args.wait:
        wait_until_finished(output)
    old, new = summary_rows(OLD), summary_rows(NEW)
    log("Evaluation finished; beginning CPU-only rear-approach population analysis")
    write_json(
        output / "status.json",
        {"status": "analyzing_441", "started_at_kst": datetime.now(KST).isoformat()},
    )
    rows, selected, data = population(old, new, output)
    plots(rows, selected, data, output)
    report(data, [], output)
    case_rows = []
    error = None
    if selected and not args.no_replay:
        try:
            write_json(
                output / "status.json",
                {
                    "status": "replaying_selected_axe_v9_clips",
                    "selected_clips": len(selected),
                },
            )
            baseline_replay = replay(selected, output)
            for row in selected:
                clip = row["clipgt_id"]
                log("Building paired camera video and plot: " + clip)
                case_rows.append(
                    case_assets(
                        clip, baseline_replay[clip], new[clip], old[clip], output
                    )
                )
        except (
            OSError,
            ValueError,
            KeyError,
            RuntimeError,
            subprocess.SubprocessError,
        ) as exc:
            error = repr(exc)
            log("Representative-case analysis failed: " + error)
    write_csv(output / "selected_case_comparison.csv", case_rows)
    report(data, case_rows, output, error)
    status = "complete" if not error else "population_complete_cases_failed"
    write_json(
        output / "status.json",
        {
            "status": status,
            "finished_at_kst": datetime.now(KST).isoformat(),
            "case_videos": len(case_rows),
            "replay_error": error,
        },
    )
    if not error and not args.no_replay:
        (output / "_complete").touch()
    log(f"{status}: {output}")


if __name__ == "__main__":
    main()
