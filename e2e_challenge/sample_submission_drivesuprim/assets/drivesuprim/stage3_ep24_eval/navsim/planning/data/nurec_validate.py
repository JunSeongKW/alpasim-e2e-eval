"""Validate the 0.5-second NuRec contract required by DriveSuprim training."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from navsim.planning.data.nurec_attach_images import DEFAULT_TRAILING_FRAMES_WITHOUT_IMAGES


EXPECTED_FRAME_INTERVAL_US = 500_000
FRAME_INTERVAL_TOLERANCE_US = 20_000
MIN_LOG_FRAMES = 12  # 4 history + current/future window used by the NuRec split
REQUIRED_CAMERAS = ("CAM_L0", "CAM_F0", "CAM_R0")

#: ``num_future_frames`` of ``train_test_split/nurec.yaml``.  NAVSIM loads
#: sensors only for the first ``num_history_frames`` of a scene window, so the
#: last frames of a log can only ever be future frames -- they are never read as
#: an observation and need no image.  Demanding one there rejects a whole clip
#: over a frame training would not have opened; see nurec_attach_images, which
#: applies the same rule when it decides what to attach.
DEFAULT_TRAILING_FRAMES_WITHOUT_IMAGES = 8


class NuRecValidationError(ValueError):
    """Raised when prepared NuRec data is not safe to pass to training."""


def _sensor_path(sensor_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else sensor_root / path


def validate_log(
    frames: Sequence[Mapping[str, Any]],
    source: Path,
    sensor_root: Path,
    map_root: Path,
    check_images: bool = True,
    trailing_frames_without_images: int = DEFAULT_TRAILING_FRAMES_WITHOUT_IMAGES,
) -> Dict[str, int]:
    if len(frames) < MIN_LOG_FRAMES:
        raise NuRecValidationError(f"{source}: needs at least {MIN_LOG_FRAMES} frames, got {len(frames)}")

    timestamps = [int(frame["timestamp"]) for frame in frames]
    errors = [abs((current - previous) - EXPECTED_FRAME_INTERVAL_US) for previous, current in zip(timestamps, timestamps[1:])]
    if errors and max(errors) > FRAME_INTERVAL_TOLERANCE_US:
        raise NuRecValidationError(
            f"{source}: expected 0.5s frames; maximum interval error is {max(errors)}us"
        )

    map_names = {str(frame.get("map_location", "")) for frame in frames}
    if len(map_names) != 1 or not next(iter(map_names)).startswith("nurec:"):
        raise NuRecValidationError(f"{source}: expected one nurec:<clip-id> map, got {sorted(map_names)}")
    map_name = next(iter(map_names))
    map_path = map_root / f"{map_name.split(':', 1)[1]}.pkl"
    if not map_path.is_file():
        raise NuRecValidationError(f"{source}: missing map bundle {map_path}")

    frames_with_route = sum(bool(frame.get("roadblock_ids")) for frame in frames)
    if frames_with_route != len(frames):
        raise NuRecValidationError(
            f"{source}: {len(frames) - frames_with_route}/{len(frames)} frames have no route"
        )

    checked_images = 0
    frames_without_images = 0
    if check_images:
        # Only frames before the tail are ever read as an observation.
        required_until = len(frames) - trailing_frames_without_images
        for frame_index, frame in enumerate(frames):
            cameras = frame.get("cams", {})
            missing = []
            for camera in REQUIRED_CAMERAS:
                entry = cameras.get(camera) or {}
                data_path = entry.get("data_path")
                if not data_path:
                    raise NuRecValidationError(f"{source}: frame {frame_index} missing {camera} path")
                image_path = _sensor_path(sensor_root, str(data_path))
                if image_path.is_file():
                    checked_images += 1
                else:
                    missing.append(image_path)
            if not missing:
                continue
            if frame_index < required_until:
                raise NuRecValidationError(
                    f"{source}: missing rendered image {missing[0]} for frame {frame_index} of "
                    f"{len(frames)}, which is not among the last {trailing_frames_without_images} "
                    f"frames -- training would read it"
                )
            frames_without_images += 1

    return {
        "frames": len(frames),
        "frames_with_route": frames_with_route,
        "checked_images": checked_images,
        "frames_without_images": frames_without_images,
        "max_frame_interval_error_us": max(errors, default=0),
    }


def validate_dataset(
    log_root: Path,
    sensor_root: Path,
    map_root: Path,
    check_images: bool = True,
    max_logs: Optional[int] = None,
    trailing_frames_without_images: int = DEFAULT_TRAILING_FRAMES_WITHOUT_IMAGES,
) -> Dict[str, Any]:
    files = sorted(log_root.glob("*.pkl"))
    if max_logs is not None:
        files = files[:max_logs]
    if not files:
        raise NuRecValidationError(f"no log pickles found in {log_root}")

    summary: Dict[str, Any] = {
        "logs": 0,
        "frames": 0,
        "frames_with_route": 0,
        "checked_images": 0,
        "frames_without_images": 0,
        "frame_interval_us": EXPECTED_FRAME_INTERVAL_US,
        "max_frame_interval_error_us": 0,
        "required_cameras": list(REQUIRED_CAMERAS),
    }
    for source in files:
        with source.open("rb") as stream:
            frames = pickle.load(stream)
        stats = validate_log(
            frames, source, sensor_root, map_root, check_images, trailing_frames_without_images
        )
        summary["logs"] += 1
        for key in ("frames", "frames_with_route", "checked_images", "frames_without_images"):
            summary[key] += stats[key]
        summary["max_frame_interval_error_us"] = max(
            summary["max_frame_interval_error_us"], stats["max_frame_interval_error_us"]
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--skip-images", action="store_true")
    parser.add_argument("--max-logs", type=int)
    parser.add_argument(
        "--trailing-frames-without-images",
        type=int,
        default=DEFAULT_TRAILING_FRAMES_WITHOUT_IMAGES,
        help="num_future_frames of the scene filter; earlier frames must all have images",
    )
    args = parser.parse_args()
    summary = validate_dataset(
        args.log_root,
        args.sensor_root,
        args.map_root,
        check_images=not args.skip_images,
        max_logs=args.max_logs,
        trailing_frames_without_images=args.trailing_frames_without_images,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
