"""Write a copy of the NAVSIM logs that describes the rectified images.

The prepared logs carry one fixed nuPlan pinhole for all 1,603 clips: fx=fy=1545,
cx=960, cy=560, and a single set of extrinsics.  The rendered images are neither
-- their lens is an f-theta whose principal point sits 185 px lower, mounted at a
pose that changes clip to clip.  Once the renders have been rectified
(:mod:`navsim.planning.data.nurec_rectify.images`) the intrinsics are known
exactly, and the extrinsics come from the clip's own ``T_sensor_rig``.

Per rendered camera this rewrites four things:

``data_path``          the rectified PNG, relative to the rectified image root
``cam_intrinsic``      :data:`target.TARGET_K`, the pinhole the images now are
``distortion``         zeros -- the lens is gone
``sensor2lidar_*``     the clip's ``T_sensor_rig``

``sensor2lidar`` needs no frame juggling, and that is worth stating rather than
assuming.  The prepared logs have ``lidar2ego`` identity, so sensor-to-lidar and
sensor-to-ego are the same transform; and the ego frame *is* the NuRec rig frame
-- ``ego2global`` reproduces ``rig_trajectories.json``'s ``T_rig_worlds`` to
millimetres at matching timestamps.  So the USDZ's sensor-to-rig is already the
sensor-to-lidar the logs want.  ``tests/test_nurec_rectified_logs.py`` checks
that identity against the real data rather than leaving it as a comment.

The five cameras NuRec does not render (CAM_L1/L2/R1/R2/B0) are left exactly as
they were: they still hold nuPlan's calibration and still point at images that
do not exist.  ``bev_num_cameras: 3`` never reads them.  A five-camera config
would, and this tree cannot serve one.

The output is a whole prepared root, not just the logs: ``scripts/nurec/train.sh``
reads ``navsim_log_path`` and ``original_sensor_path`` out of
``$NUREC_PREPARED_ROOT/index.json``, so an ``index.json`` naming the rectified
sensor root is written beside the logs, and ``maps``, ``routes`` and
``metric_cache`` -- all keyed by log name, none of them touched by rectification
-- are symlinked back to the source root.  ``NUREC_PREPARED_ROOT`` can then point
here and nothing else in the training command changes.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from navsim.planning.data.nurec_attach_images import (
    DEFAULT_MAX_TIMESTAMP_ERROR_US,
    DEFAULT_TRAILING_FRAMES_WITHOUT_IMAGES,
    NuRecImageMatchError,
    camera_indices_for,
    clip_id_of,
    match_frames_to_images,
)
from navsim.planning.data.nurec_rectify.calibration import (
    CAMERA_SENSORS,
    ClipCalibration,
    load_cache,
)
from navsim.planning.data.nurec_rectify.target import TARGET_DISTORTION, TARGET_K

#: Keyed by log name and unchanged by rectification, so the rectified root
#: borrows them rather than copying 1,603 of each.
LINKED_FROM_SOURCE = ("maps", "routes", "metric_cache")

#: ``data_path`` for a frame with no render.  Nothing opens it -- it is only
#: ever a future frame -- but if anything ever does, it should say why it fails
#: instead of pointing at a stale path under the un-rectified render tree.
NO_RENDER_PREFIX = "__no_render__"


def rewrite_frames(
    frames: List[Dict[str, Any]],
    calibration: ClipCalibration,
    image_root: Path,
    clip_id: str,
    max_timestamp_error_us: int = DEFAULT_MAX_TIMESTAMP_ERROR_US,
    trailing_frames_without_images: int = DEFAULT_TRAILING_FRAMES_WITHOUT_IMAGES,
) -> Dict[str, Any]:
    """Point one log's rendered cameras at the rectified images, in place."""

    camera_indices = camera_indices_for(clip_id, image_root / clip_id)
    matches, max_error = match_frames_to_images(
        frames, camera_indices, clip_id, max_timestamp_error_us, trailing_frames_without_images
    )

    attached = 0
    for frame, paths_for_frame in zip(frames, matches):
        for navsim_name, sensor in CAMERA_SENSORS.items():
            camera = frame["cams"].setdefault(navsim_name, {})
            if paths_for_frame is None:
                camera["data_path"] = (
                    f"{NO_RENDER_PREFIX}/{clip_id}/{sensor}/{frame['timestamp']}.png"
                )
            else:
                camera["data_path"] = str(paths_for_frame[navsim_name].relative_to(image_root))
                attached += 1
            entry = calibration[navsim_name]
            camera["cam_intrinsic"] = TARGET_K.copy()
            camera["distortion"] = TARGET_DISTORTION.copy()
            # A copy per frame, not a shared view of the cached 4x4: the logs
            # are pickled, and anything that later edits one frame's extrinsics
            # must not reach into the other forty.
            camera["sensor2lidar_rotation"] = np.array(entry.rotation, dtype=np.float64)
            camera["sensor2lidar_translation"] = np.array(entry.translation, dtype=np.float64)
    return {
        "frames": len(frames),
        "frames_with_images": sum(entry is not None for entry in matches),
        "images": attached,
        "max_timestamp_error_us": max_error,
    }


def _link_prepared_root(source_prepared: Path, output_prepared: Path, image_root: Path,
                        written: List[str]) -> Dict[str, Any]:
    """Give the rectified logs the rest of a prepared root: index and sidecars."""

    for name in LINKED_FROM_SOURCE:
        source = source_prepared / name
        if not source.exists():
            continue
        link = output_prepared / name
        if link.is_symlink() or link.exists():
            if link.is_symlink() and link.readlink() == source.resolve():
                continue
            link.unlink()
        link.symlink_to(source.resolve(), target_is_directory=True)

    index_path = source_prepared / "index.json"
    if not index_path.is_file():
        return {"index": None}
    index = json.loads(index_path.read_text())
    index["sensor_root"] = str(image_root.resolve())
    if index.get("map_root"):
        index["map_root"] = str((output_prepared / "maps").resolve())
    # A log that could not be rewritten has no images here, so it must not stay
    # in a split that training will try to load.
    kept = set(written)
    index["logs"] = {
        split: [name for name in names if f"nurec-{name}" in kept or name in kept]
        for split, names in index.get("logs", {}).items()
    }
    index["num_logs"] = len(kept)
    index["rectified"] = True
    (output_prepared / "index.json").write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
    return {
        "index": str(output_prepared / "index.json"),
        "index_logs": {split: len(names) for split, names in index["logs"].items()},
    }


LOGS_WITHIN_PREPARED = Path("navsim_logs") / "trainval"


def write_logs(
    source_prepared: Path,
    output_prepared: Path,
    calibration_path: Path,
    image_root: Path,
    allow_incomplete: bool = True,
    max_timestamp_error_us: int = DEFAULT_MAX_TIMESTAMP_ERROR_US,
    trailing_frames_without_images: int = DEFAULT_TRAILING_FRAMES_WITHOUT_IMAGES,
    log_names: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Write a rectified prepared root.  The source root is not touched.

    In goes the root ``NUREC_PREPARED_ROOT`` names today; out comes one that can
    replace it -- rewritten logs under ``navsim_logs/trainval``, an ``index.json``
    naming the rectified sensor root, and the sidecars symlinked back.
    """

    if output_prepared.resolve() == source_prepared.resolve():
        raise ValueError("refusing to rewrite the prepared root in place; choose another output root")
    source_root = source_prepared / LOGS_WITHIN_PREPARED
    output_root = output_prepared / LOGS_WITHIN_PREPARED

    calibration = load_cache(calibration_path)
    wanted = set(log_names) if log_names is not None else None
    sources = sorted(source_root.glob("*.pkl"))
    if not sources:
        raise FileNotFoundError(f"no prepared logs found in {source_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, Any] = {
        "logs": 0,
        "written_logs": 0,
        "frames": 0,
        "images": 0,
        "frames_without_images": 0,
        "max_timestamp_error_us": 0,
        "skipped": {},
    }
    written_names: List[str] = []
    for source in sources:
        if wanted is not None and source.stem not in wanted:
            continue
        summary["logs"] += 1
        with source.open("rb") as stream:
            frames = pickle.load(stream)
        clip_id = clip_id_of(frames, source)
        clip_calibration = calibration.get(clip_id)
        if clip_calibration is None:
            if not allow_incomplete:
                raise NuRecImageMatchError(f"{clip_id}: not in the calibration cache")
            summary["skipped"][clip_id] = "no calibration"
            continue
        try:
            stats = rewrite_frames(
                frames, clip_calibration, image_root, clip_id,
                max_timestamp_error_us, trailing_frames_without_images,
            )
        except NuRecImageMatchError as error:
            if not allow_incomplete:
                raise
            summary["skipped"][clip_id] = str(error)
            continue

        destination = output_root / source.name
        temporary = destination.with_suffix(".pkl.tmp")
        with temporary.open("wb") as stream:
            pickle.dump(frames, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(destination)
        summary["written_logs"] += 1
        summary["frames"] += stats["frames"]
        summary["images"] += stats["images"]
        summary["frames_without_images"] += stats["frames"] - stats["frames_with_images"]
        summary["max_timestamp_error_us"] = max(
            summary["max_timestamp_error_us"], stats["max_timestamp_error_us"]
        )
        written_names.append(source.stem)

    summary["prepared_root"] = _link_prepared_root(
        source_prepared, output_prepared, image_root, written_names
    )

    manifest = {
        "format_version": 1,
        "source_prepared_root": str(source_prepared),
        "image_root": str(image_root),
        "calibration": str(calibration_path),
        "rectified_cameras": sorted(CAMERA_SENSORS),
        "uncorrected_cameras": ["CAM_L1", "CAM_L2", "CAM_R1", "CAM_R2", "CAM_B0"],
        "cam_intrinsic": TARGET_K.tolist(),
        "resolution_wh": [512, 256],
        "logs": summary["written_logs"],
        "frames": summary["frames"],
    }
    (output_prepared / "rectified_logs.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prepared-root", type=Path, required=True,
                        help="the root NUREC_PREPARED_ROOT names today")
    parser.add_argument("--output-prepared-root", type=Path, required=True,
                        help="where to write the rectified replacement for it")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--fail-on-incomplete", action="store_true")
    parser.add_argument("--max-timestamp-error-us", type=int, default=DEFAULT_MAX_TIMESTAMP_ERROR_US)
    parser.add_argument("--trailing-frames-without-images", type=int,
                        default=DEFAULT_TRAILING_FRAMES_WITHOUT_IMAGES)
    parser.add_argument("--log-name", action="append", dest="log_names", default=None)
    args = parser.parse_args()

    summary = write_logs(
        args.source_prepared_root, args.output_prepared_root, args.calibration, args.image_root,
        allow_incomplete=not args.fail_on_incomplete,
        max_timestamp_error_us=args.max_timestamp_error_us,
        trailing_frames_without_images=args.trailing_frames_without_images,
        log_names=args.log_names,
    )
    skipped = summary.pop("skipped")
    print(json.dumps({**summary, "skipped_logs": len(skipped)}, indent=2))
    for clip_id, reason in list(skipped.items())[:5]:
        print(f"  skipped {clip_id}: {reason}")


if __name__ == "__main__":
    main()
