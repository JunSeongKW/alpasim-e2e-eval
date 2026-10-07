# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""Check exposure detection, reaction censoring, and ASL payload skipping."""

import importlib.util
import struct
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from alpasim_grpc.v0.logging_pb2 import LogEntry

SPEC = importlib.util.spec_from_file_location(
    "rear_response", Path(__file__).with_name("rear_response.py")
)
rear = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rear)


def trace():
    t = np.arange(0, 10.01, 0.1)
    return {
        "times": t,
        "force_gt_s": 1.7,
        "stop_s": np.inf,
        "speed": 5 + t,
        "traffic": {
            "following_car": {
                "corridor": np.ones(len(t), dtype=bool),
                "gap": 12 - t,
                "closing": np.full(len(t), 3.0),
                "ttc": (12 - t) / 3,
                "front_gap": -20 + t,
            }
        },
    }


def test_persistent_approach_excludes_initial_gt_and_records_positive_reaction():
    sample = trace()
    event = rear.approach_event(sample)
    assert event["onset_s"] >= 2.2
    result = rear.response(sample, event)
    assert result["response_observed"]
    assert result["accelerated_1mps"]
    assert 1.5 < result["speed_change_mps"] < 2.0
    assert not result["front_blocked_10m"]


def test_short_exposure_and_nonclosing_vehicle_are_not_reactions():
    sample = trace()
    sample["traffic"]["following_car"]["corridor"][:] = False
    sample["traffic"]["following_car"]["corridor"][30:34] = True
    assert rear.approach_event(sample) is None
    sample = trace()
    sample["traffic"]["following_car"]["closing"][:] = -1
    assert rear.approach_event(sample) is None


def test_collision_within_response_window_is_censored_not_counted_as_acceleration():
    sample = trace()
    event = rear.approach_event(sample)
    sample["stop_s"] = event["onset_s"] + 1.5
    result = rear.response(sample, event)
    assert not result["response_observed"]
    assert result["speed_change_mps"] is None
    assert result["accelerated_1mps"] is None


def test_gt_speed_trend_is_reported_separately_from_observed_acceleration():
    sample = trace()
    sample["gt_speed"] = sample["speed"].copy()
    result = rear.response(sample, rear.approach_event(sample))
    assert result["accelerated_1mps"]
    assert result["speed_change_minus_gt_mps"] == 0


def test_physical_failures_remain_comparable_and_rpc_error_is_not_no_collision():
    clean = {
        "score": 1.0,
        "failure_reason": None,
        "metrics": {
            "collision_rear": 0,
            "collision_at_fault": 0,
            "offroad": 0,
            "left_corridor_laterally": 0,
        },
    }
    old = {f"clipgt-{i:04d}": deepcopy(clean) for i in range(441)}
    new = deepcopy(old)
    old["clipgt-0000"]["failure_reason"] = "offroad"
    old["clipgt-0000"]["metrics"].update({"collision_rear": 1, "offroad": 1})
    old["clipgt-0001"]["metrics"] = {"error": "RPC unavailable"}
    new["clipgt-0002"]["metrics"].update({"collision_rear": 1, "collision_at_fault": 1})
    new["clipgt-0002"]["failure_reason"] = "collision_at_fault"
    rows = rear.paired_rows(old, new)
    assert len(rows) == 441
    assert rows[0]["collision_transition"] == "axe-v9 rear collision avoided"
    assert (
        rows[1]["collision_transition"]
        == "unavailable due to evaluation failure/missing metric"
    )
    assert rows[2]["collision_transition"] == "new rear collision"


def test_camera_comparison_encodes_a_real_mp4_with_low_cpu_use(tmp_path, monkeypatch):
    import io

    import imageio.v2 as imageio
    from PIL import Image

    t = np.arange(0, 0.31, 0.1)
    sample = {
        "times": t,
        "force_gt_s": 1.7,
        "stop_s": np.inf,
        "speed": np.full(len(t), 5.0),
        "gt_speed": np.full(len(t), 5.0),
        "traffic": {},
        "rear_collision_s": None,
    }
    buffer = io.BytesIO()
    Image.new("RGB", (32, 16), "#2377b8").save(buffer, format="PNG")
    streams = {cam: [(float(s), buffer.getvalue()) for s in t] for cam in rear.CAMERAS}
    monkeypatch.setattr(rear, "read_trace", lambda _: sample)
    monkeypatch.setattr(rear, "asl_for", lambda *_: tmp_path / "unused.asl")
    monkeypatch.setattr(rear, "cameras", lambda _: streams)
    row = {"score": 1.0}
    result = rear.case_assets("clipgt-synthetic", row, row, row, tmp_path)
    path = Path(result["video_path"])
    assert path.stat().st_size > 1000
    with imageio.get_reader(path) as reader:
        assert reader.count_frames() == len(t)
    assert Path(result["plot_path"]).exists()


def test_reader_skips_large_camera_payload_and_rejects_truncated_envelope(tmp_path):
    metadata = LogEntry()
    metadata.rollout_metadata.session_metadata.scene_id = "clipgt-test"
    camera = LogEntry()
    camera.driver_camera_image.camera_image.image_bytes = b"x" * 100000
    poses = LogEntry()
    poses.actor_poses.timestamp_us = 123
    path = tmp_path / "rollout.asl"
    data = b"".join(
        struct.pack(">I", len(b)) + b
        for b in [m.SerializeToString() for m in [metadata, camera, poses]]
    )
    path.write_bytes(data)
    assert [m.WhichOneof("log_entry") for m in rear.messages(path, {1, 2})] == [
        "rollout_metadata",
        "actor_poses",
    ]
    path.write_bytes(data[:-1])
    with pytest.raises(ValueError, match="Truncated"):
        list(rear.messages(path, {1, 2}))
