"""Attach nearest rendered NuRec images to the 0.5-second NAVSIM logs.

A log's last frames need no image.  NAVSIM builds a scene from
``num_history_frames`` + ``num_future_frames`` stored frames and only ever loads
sensors for the history frames (``AgentInput.from_scene_dict_list``), so the
final ``num_future_frames`` of a log can only ever appear as future frames and
are never read as an observation.  The 2 Hz render stops one frame short of the
41-frame logs, which under a whole-log requirement rejected 1,562 of 1,603 clips
over an image no training window would have opened.

Missing images are therefore tolerated, but only as a suffix: a gap anywhere
before the tail still rejects the clip, because that one *would* be read.
"""

from __future__ import annotations

import argparse
import json
import pickle
from bisect import bisect_left
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


CAMERA_DIRECTORIES = {
    "CAM_L0": "camera_cross_left_120fov",
    "CAM_F0": "camera_front_wide_120fov",
    "CAM_R0": "camera_cross_right_120fov",
}
DEFAULT_MAX_TIMESTAMP_ERROR_US = 60_000

#: ``num_future_frames`` of ``train_test_split/nurec.yaml``.  The last frames of
#: a log can only ever be future frames of a scene, and future frames carry no
#: sensor data.
DEFAULT_TRAILING_FRAMES_WITHOUT_IMAGES = 8


class NuRecImageMatchError(ValueError):
    """Raised when rendered frames cannot satisfy a prepared log."""


def _clip_id(frames: Sequence[Mapping[str, Any]], source: Path) -> str:
    name = str(frames[0].get("log_name") or source.stem)
    return name[len("nurec-") :] if name.startswith("nurec-") else name


def clip_id_of(frames: Sequence[Mapping[str, Any]], source: Path) -> str:
    """The NuRec clip id a prepared log came from."""
    return _clip_id(frames, source)


def _timestamped_images(directory: Path) -> Tuple[List[int], Dict[int, Path]]:
    values: Dict[int, Path] = {}
    for path in directory.glob("*.png"):
        try:
            values[int(path.stem)] = path
        except ValueError:
            continue
    return sorted(values), values


def _nearest_timestamp(timestamps: Sequence[int], target: int) -> int:
    index = bisect_left(timestamps, target)
    candidates = [value for value in (index - 1, index) if 0 <= value < len(timestamps)]
    if not candidates:
        raise NuRecImageMatchError("rendered camera directory contains no timestamped PNG files")
    return timestamps[min(candidates, key=lambda value: abs(timestamps[value] - target))]


def camera_indices_for(clip_id: str, clip_root: Path) -> Dict[str, Tuple[List[int], Dict[int, Path]]]:
    """Index the rendered PNGs of one clip by timestamp, per camera."""
    if not (clip_root / "_complete").is_file():
        raise NuRecImageMatchError(f"{clip_id}: renderer completion marker is missing")
    indices: Dict[str, Tuple[List[int], Dict[int, Path]]] = {}
    for navsim_name, directory_name in CAMERA_DIRECTORIES.items():
        directory = clip_root / directory_name
        timestamps, paths = _timestamped_images(directory)
        if not timestamps:
            raise NuRecImageMatchError(f"{clip_id}: no PNG frames in {directory}")
        indices[navsim_name] = (timestamps, paths)
    return indices


def match_frames_to_images(
    frames: Sequence[Mapping[str, Any]],
    camera_indices: Mapping[str, Tuple[List[int], Dict[int, Path]]],
    clip_id: str,
    max_timestamp_error_us: int = DEFAULT_MAX_TIMESTAMP_ERROR_US,
    trailing_frames_without_images: int = DEFAULT_TRAILING_FRAMES_WITHOUT_IMAGES,
) -> Tuple[List[Optional[Dict[str, Path]]], int]:
    """Pair each frame with one rendered PNG per camera.

    Returns one entry per frame -- a ``{camera: path}`` mapping, or ``None`` when
    no render is close enough -- and the largest timestamp error accepted.  A
    ``None`` is only allowed among the last ``trailing_frames_without_images``
    frames; an earlier one raises, since NAVSIM would try to open it.
    """

    matches: List[Optional[Dict[str, Path]]] = []
    max_error = 0
    for frame in frames:
        target = int(frame["timestamp"])
        paths_for_frame: Dict[str, Path] = {}
        worst = 0
        for navsim_name, (timestamps, paths) in camera_indices.items():
            matched = _nearest_timestamp(timestamps, target)
            error = abs(matched - target)
            worst = max(worst, error)
            paths_for_frame[navsim_name] = paths[matched]
        if worst > max_timestamp_error_us:
            matches.append(None)
        else:
            max_error = max(max_error, worst)
            matches.append(paths_for_frame)

    first_unmatched = next((index for index, entry in enumerate(matches) if entry is None), None)
    if first_unmatched is not None:
        tail_start = len(matches) - trailing_frames_without_images
        if first_unmatched < tail_start:
            raise NuRecImageMatchError(
                f"{clip_id}: frame {first_unmatched} of {len(matches)} has no rendered image "
                f"within {max_timestamp_error_us}us, and it is not among the last "
                f"{trailing_frames_without_images} frames -- NAVSIM would read it"
            )
    return matches, max_error


def attach_images(
    frames: List[Dict[str, Any]],
    source: Path,
    image_root: Path,
    max_timestamp_error_us: int = DEFAULT_MAX_TIMESTAMP_ERROR_US,
    trailing_frames_without_images: int = DEFAULT_TRAILING_FRAMES_WITHOUT_IMAGES,
) -> Dict[str, int]:
    clip_id = _clip_id(frames, source)
    camera_indices = camera_indices_for(clip_id, image_root / clip_id)
    matches, max_error = match_frames_to_images(
        frames, camera_indices, clip_id, max_timestamp_error_us, trailing_frames_without_images
    )

    attached = 0
    for frame, paths_for_frame in zip(frames, matches):
        if paths_for_frame is None:
            continue
        for navsim_name, path in paths_for_frame.items():
            camera = frame["cams"].setdefault(navsim_name, {})
            camera["data_path"] = str(path.relative_to(image_root))
            attached += 1
    return {
        "frames": len(frames),
        "frames_with_images": sum(entry is not None for entry in matches),
        "images": attached,
        "max_timestamp_error_us": max_error,
    }


def process_logs(
    log_root: Path,
    image_root: Path,
    allow_incomplete: bool = False,
    max_timestamp_error_us: int = DEFAULT_MAX_TIMESTAMP_ERROR_US,
    trailing_frames_without_images: int = DEFAULT_TRAILING_FRAMES_WITHOUT_IMAGES,
) -> Dict[str, Any]:
    files = sorted(log_root.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"no prepared logs found in {log_root}")
    summary: Dict[str, Any] = {
        "logs": len(files),
        "attached_logs": 0,
        "attached_frames": 0,
        "attached_images": 0,
        "frames_without_images": 0,
        "incomplete_clips": [],
        "max_timestamp_error_us": 0,
    }
    for source in files:
        with source.open("rb") as stream:
            frames = pickle.load(stream)
        try:
            stats = attach_images(
                frames, source, image_root, max_timestamp_error_us, trailing_frames_without_images
            )
        except NuRecImageMatchError:
            if not allow_incomplete:
                raise
            summary["incomplete_clips"].append(_clip_id(frames, source))
            continue
        # These are generated prepared logs, not the private source conversion.
        # Replacing one pickle is atomic so interrupted sync cannot corrupt it.
        temporary = source.with_suffix(".pkl.tmp")
        with temporary.open("wb") as stream:
            pickle.dump(frames, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(source)
        summary["attached_logs"] += 1
        summary["attached_frames"] += stats["frames"]
        summary["attached_images"] += stats["images"]
        summary["frames_without_images"] += stats["frames"] - stats["frames_with_images"]
        summary["max_timestamp_error_us"] = max(summary["max_timestamp_error_us"], stats["max_timestamp_error_us"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--max-timestamp-error-us", type=int, default=DEFAULT_MAX_TIMESTAMP_ERROR_US)
    parser.add_argument(
        "--trailing-frames-without-images",
        type=int,
        default=DEFAULT_TRAILING_FRAMES_WITHOUT_IMAGES,
        help="num_future_frames of the scene filter; frames past the tail must all have images",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            process_logs(
                args.log_root,
                args.image_root,
                allow_incomplete=args.allow_incomplete,
                max_timestamp_error_us=args.max_timestamp_error_us,
                trailing_frames_without_images=args.trailing_frames_without_images,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
