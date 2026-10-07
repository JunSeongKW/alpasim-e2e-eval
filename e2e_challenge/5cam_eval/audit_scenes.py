# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""Require all five calibrated cameras in the existing 441-scene artifacts."""

import argparse
import importlib.util
import json
import zipfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
SENSORS = {
    "CAM_L1": "camera_rear_left_70fov",
    "CAM_L0": "camera_cross_left_120fov",
    "CAM_F0": "camera_front_wide_120fov",
    "CAM_R0": "camera_cross_right_120fov",
    "CAM_R1": "camera_rear_right_70fov",
}


def audit(output: Path) -> None:
    spec = importlib.util.spec_from_file_location(
        "scene_scan", ROOT / "e2e_challenge/axe_local_eval/scan_scene_attrs.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = module.resolve_paths(
        str(ROOT / "runs/leaderboard-merged-route-ep30/aggregate/results-summary.json")
    )
    if len(paths) != 441:
        raise RuntimeError(f"Expected 441 artifacts, found {len(paths)}")
    records, fixture = [], None
    for clip, path in sorted(paths.items()):
        with zipfile.ZipFile(path) as archive:
            payload = json.loads(archive.read("rig_trajectories.json"))
            scene = yaml.safe_load(archive.read("metadata.yaml"))["scene_id"]
        if str(scene).removeprefix("clipgt-") != clip:
            raise RuntimeError(f"Scene identity mismatch: {clip} != {scene}")
        by_sensor = {
            c["logical_sensor_name"]: (key, c)
            for key, c in payload["camera_calibrations"].items()
        }
        cameras = {}
        for logical, sensor in SENSORS.items():
            key, camera = by_sensor[sensor]
            transform = np.asarray(camera["T_sensor_rig"])
            params = camera["camera_model"]["parameters"]
            if camera["camera_model"]["type"] != "ftheta":
                raise RuntimeError(f"{clip}/{sensor}: requires ftheta")
            if transform.shape != (4, 4) or not np.isfinite(transform).all():
                raise RuntimeError(f"{clip}/{sensor}: invalid extrinsics")
            if not params.get("angle_to_pixeldist_poly"):
                raise RuntimeError(f"{clip}/{sensor}: missing lens polynomial")
            for trajectory in payload["rig_trajectories"]:
                if not trajectory["cameras_frame_timestamps_us"].get(key):
                    raise RuntimeError(f"{clip}/{sensor}: missing shutter timestamps")
            cameras[logical] = camera
        records.append({"clipgt_id": str(scene), "cameras": list(cameras)})
        if fixture is None:
            fixture = {
                "clipgt_id": str(scene),
                "cameras": cameras,
                "rig_bbox": payload["rig_trajectories"][0]["rig_bbox"],
            }
    output.mkdir(parents=True, exist_ok=True)
    (output / "scene_audit.json").write_text(
        json.dumps({"scene_count": len(records), "records": records}, indent=2) + "\n"
    )
    (output / "camera_fixture.json").write_text(json.dumps(fixture, indent=2) + "\n")
    print(
        f"PASS: all {len(records)} scenes have five ftheta cameras and shutter timestamps"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
