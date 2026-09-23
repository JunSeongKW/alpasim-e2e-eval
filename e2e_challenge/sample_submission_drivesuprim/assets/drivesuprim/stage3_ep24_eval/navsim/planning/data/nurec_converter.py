"""Convert a portable NuRec export into the NAVSIM log contract used by AXE.

NuRec installations are not assumed to be importable on training machines.  The
boundary is therefore a JSONL manifest: one line per log, containing ``frames``.
Images stay on the storage server; generated NAVSIM logs only refer to them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


CAMERAS = ("cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0")


class NuRecFormatError(ValueError):
    """Raised when a manifest cannot satisfy AXE's scene contract."""


def _token(log_name: str, frame_index: int) -> str:
    return hashlib.sha1(f"nurec:{log_name}:{frame_index}".encode()).hexdigest()


def _safe_name(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not value:
        raise NuRecFormatError("log_name becomes empty after sanitization")
    return value


def _quaternion(frame: Mapping[str, Any]) -> List[float]:
    if "ego2global_rotation" in frame:
        q = list(frame["ego2global_rotation"])
        if len(q) != 4:
            raise NuRecFormatError("ego2global_rotation must be [w, x, y, z]")
        return [float(x) for x in q]
    yaw = float(frame.get("ego_yaw", 0.0))
    return [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]


def _vector(value: Any, size: int, field: str, default: Sequence[float]) -> List[float]:
    result = list(default if value is None else value)
    if len(result) != size:
        raise NuRecFormatError(f"{field} must contain {size} values, got {len(result)}")
    return [float(x) for x in result]


def _camera(camera: Mapping[str, Any], name: str) -> Dict[str, Any]:
    path = camera.get("data_path") or camera.get("image_path")
    if not path:
        raise NuRecFormatError(f"camera {name} is missing data_path")
    return {
        "data_path": str(path),
        "sensor2lidar_rotation": camera.get("sensor2lidar_rotation"),
        "sensor2lidar_translation": camera.get("sensor2lidar_translation"),
        "cam_intrinsic": camera.get("cam_intrinsic", camera.get("intrinsics")),
        "distortion": camera.get("distortion"),
    }


def _annotations(frame: Mapping[str, Any]) -> Dict[str, Any]:
    anns = frame.get("anns", frame.get("annotations", {}))
    boxes = anns.get("gt_boxes", anns.get("boxes", []))
    names = anns.get("gt_names", anns.get("names", []))
    count = len(boxes)
    result = {
        "gt_boxes": boxes,
        "gt_names": names,
        "gt_velocity_3d": anns.get("gt_velocity_3d", anns.get("velocity_3d", [[0, 0, 0]] * count)),
        "instance_tokens": anns.get("instance_tokens", [f"instance-{i}" for i in range(count)]),
        "track_tokens": anns.get("track_tokens", [f"track-{i}" for i in range(count)]),
    }
    lengths = {key: len(value) for key, value in result.items()}
    if len(set(lengths.values())) != 1:
        raise NuRecFormatError(f"annotation arrays have unequal lengths: {lengths}")
    return result


def convert_log(record: Mapping[str, Any], sensor_root: Path, check_files: bool = True) -> tuple[str, str, List[Dict[str, Any]]]:
    log_name = _safe_name(str(record["log_name"]))
    split = str(record.get("split", "train"))
    if split not in {"train", "val"}:
        raise NuRecFormatError(f"{log_name}: split must be train or val, got {split!r}")
    map_location = record.get("map_location")
    if not map_location:
        raise NuRecFormatError(f"{log_name}: map_location is required (must match an installed nuPlan map)")
    frames = record.get("frames", [])
    if not frames:
        raise NuRecFormatError(f"{log_name}: frames is empty")

    output: List[Dict[str, Any]] = []
    previous_timestamp = None
    scene_token = str(record.get("scene_token", f"nurec-{log_name}"))
    for index, frame in enumerate(frames):
        timestamp = int(frame.get("timestamp_us", frame.get("timestamp", index * 500_000)))
        if previous_timestamp is not None and timestamp <= previous_timestamp:
            raise NuRecFormatError(f"{log_name}: timestamps must be strictly increasing")
        previous_timestamp = timestamp
        translation = _vector(frame.get("ego2global_translation", frame.get("ego_translation")), 3,
                              "ego translation", [0, 0, 0])
        velocity = _vector(frame.get("ego_velocity"), 2, "ego_velocity", [0, 0])
        acceleration = _vector(frame.get("ego_acceleration"), 2, "ego_acceleration", [0, 0])
        command = frame.get("driving_command", [1, 0, 0, 0])
        command = _vector(command, 4, "driving_command", [1, 0, 0, 0])
        source_cameras = {str(k).lower(): v for k, v in frame.get("cameras", frame.get("cams", {})).items()}
        missing = [name for name in CAMERAS if name not in source_cameras]
        if missing:
            raise NuRecFormatError(f"{log_name} frame {index}: missing cameras {missing}")
        cameras = {name.upper(): _camera(source_cameras[name], name) for name in CAMERAS}
        if check_files:
            absent = [entry["data_path"] for entry in cameras.values()
                      if not (sensor_root / entry["data_path"]).is_file()]
            if absent:
                raise NuRecFormatError(f"{log_name} frame {index}: missing sensor files, first={absent[0]}")
        output.append({
            "token": str(frame.get("token", _token(log_name, index))),
            "timestamp": timestamp,
            "log_name": log_name,
            "scene_token": scene_token,
            "map_location": str(map_location),
            "ego2global_translation": translation,
            "ego2global_rotation": _quaternion(frame),
            "ego_dynamic_state": velocity + acceleration,
            "driving_command": command,
            "roadblock_ids": list(frame.get("roadblock_ids", [])),
            "traffic_lights": list(frame.get("traffic_lights", [])),
            "anns": _annotations(frame),
            "cams": cameras,
            # NAVSIM constructs a Path even when lidar is disabled.  A harmless
            # sentinel keeps camera-only AXE configurations loadable.
            "lidar_path": frame.get("lidar_path") or "__unused__.pcd",
        })
    return log_name, split, output


def read_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as error:
                raise NuRecFormatError(f"{path}:{line_number}: {error}") from error


def prepare(manifest: Path, sensor_root: Path, output_root: Path, check_files: bool = True) -> Dict[str, Any]:
    logs_dir = output_root / "navsim_logs" / "trainval"
    logs_dir.mkdir(parents=True, exist_ok=True)
    splits: Dict[str, List[str]] = {"train": [], "val": []}
    frame_count = 0
    seen = set()
    for record in read_jsonl(manifest):
        log_name, split, frames = convert_log(record, sensor_root, check_files)
        if log_name in seen:
            raise NuRecFormatError(f"duplicate log_name: {log_name}")
        seen.add(log_name)
        with (logs_dir / f"{log_name}.pkl").open("wb") as stream:
            pickle.dump(frames, stream, protocol=pickle.HIGHEST_PROTOCOL)
        splits[split].append(log_name)
        frame_count += len(frames)
    if not splits["train"] or not splits["val"]:
        raise NuRecFormatError("manifest must contain at least one train and one val log")
    index = {"format_version": 1, "sensor_root": str(sensor_root.resolve()), "logs": splits,
             "num_logs": len(seen), "num_frames": frame_count}
    (output_root / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--skip-file-check", action="store_true", help="Allow preparing before storage is mounted")
    args = parser.parse_args()
    index = prepare(args.manifest, args.sensor_root, args.output_root, not args.skip_file_check)
    print(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
