"""Rectify the rendered f-theta PNGs into the pinhole view the model reads.

Input is the render tree the NuRec renderer wrote,
``{render_root}/{clip}/{sensor}/{timestamp}.png`` at the camera's native
resolution.  Output mirrors it exactly, one 512x256 PNG per input, so the log
rewriter can swap ``data_path`` roots and change nothing else.

The rectifier is per (clip, camera) -- both the lens polynomials and the
principal point differ from clip to clip -- and costs about 0.1 s to build
against 2-10 ms to apply, so it is built once per camera and reused for the
clip's forty frames.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from navsim.planning.data.nurec_rectify import ftheta_proto
from navsim.planning.data.nurec_rectify.calibration import (
    CAMERA_SENSORS,
    ClipCalibration,
    NuRecCalibrationError,
    load_cache,
)
from navsim.planning.data.nurec_rectify.rectification import (
    build_ftheta_rectifier_for_resolution,
)
from navsim.planning.data.nurec_rectify.target import (
    TARGET_HEIGHT,
    TARGET_WIDTH,
    rectification_target_config,
)

COMPLETION_MARKER = "_complete"
PNG_COMPRESSION = 3


class NuRecRectifyError(RuntimeError):
    """Raised when a clip's renders cannot be rectified."""


@dataclass
class ClipResult:
    clip_id: str
    images: int
    skipped: bool = False
    error: Optional[str] = None


def build_rectifier(calibration, source_resolution_hw: Tuple[int, int]):
    """A rectifier for one camera, built the way the challenge driver builds it."""
    camera = ftheta_proto.available_camera_from_usdz(
        calibration.sensor_name, calibration.camera_model
    )
    return build_ftheta_rectifier_for_resolution(
        camera_proto=camera,
        target_cfg=rectification_target_config(),
        source_resolution_hw=source_resolution_hw,
    )


def _source_resolution(paths: List[Path], sensor: str) -> Tuple[int, int]:
    """Read the resolution off the renders rather than trusting the USDZ.

    The renders are not always at the camera's native resolution -- one pass was
    made at 1900x1080 against a 1920x1080 lens -- and the rectifier scales the
    f-theta model to whatever it is actually given.
    """
    probe = cv2.imread(str(paths[0]), cv2.IMREAD_COLOR)
    if probe is None:
        raise NuRecRectifyError(f"{sensor}: cannot read {paths[0]}")
    return probe.shape[0], probe.shape[1]


def rectify_clip(
    clip_id: str,
    calibration: ClipCalibration,
    render_root: Path,
    output_root: Path,
    overwrite: bool = False,
) -> ClipResult:
    """Rectify every rendered frame of one clip into ``output_root``."""

    clip_out = output_root / clip_id
    if not overwrite and (clip_out / COMPLETION_MARKER).is_file():
        return ClipResult(clip_id=clip_id, images=0, skipped=True)

    clip_in = render_root / clip_id
    if not (clip_in / COMPLETION_MARKER).is_file():
        raise NuRecRectifyError(f"{clip_id}: renderer completion marker is missing")

    written = 0
    per_camera: Dict[str, int] = {}
    for navsim_name, sensor in CAMERA_SENSORS.items():
        source_dir = clip_in / sensor
        frames = sorted(source_dir.glob("*.png"))
        if not frames:
            raise NuRecRectifyError(f"{clip_id}: no PNG frames in {source_dir}")

        rectifier = build_rectifier(calibration[navsim_name],
                                    _source_resolution(frames, sensor))
        target_dir = clip_out / sensor
        target_dir.mkdir(parents=True, exist_ok=True)
        for frame in frames:
            image = cv2.imread(str(frame), cv2.IMREAD_COLOR)
            if image is None:
                raise NuRecRectifyError(f"{clip_id}: cannot read {frame}")
            rectified = rectifier.rectify(image)
            if rectified.shape[:2] != (TARGET_HEIGHT, TARGET_WIDTH):
                raise NuRecRectifyError(
                    f"{clip_id}: rectified {frame.name} is {rectified.shape[:2]}, "
                    f"expected {(TARGET_HEIGHT, TARGET_WIDTH)}"
                )
            destination = target_dir / frame.name
            # cv2 picks its encoder off the extension, so the staging name has
            # to keep .png last.
            staging = target_dir / f".{frame.stem}.tmp.png"
            if not cv2.imwrite(str(staging), rectified,
                               [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION]):
                raise NuRecRectifyError(f"{clip_id}: cannot write {destination}")
            os.replace(staging, destination)
            written += 1
        per_camera[sensor] = len(frames)

    marker = (
        f"format=rectified_pinhole\n"
        f"resolution={TARGET_WIDTH}x{TARGET_HEIGHT}\n"
        f"source={render_root}\n"
        f"png_count={written}\n"
        + "".join(f"camera_counts={name}={count}\n" for name, count in per_camera.items())
    )
    (clip_out / COMPLETION_MARKER).write_text(marker, encoding="utf-8")
    return ClipResult(clip_id=clip_id, images=written)


_WORKER: Dict[str, Any] = {}


def _initialise(calibration_path: str, render_root: str, output_root: str, overwrite: bool) -> None:
    # cv2 defaults to one thread per core inside every worker, which on a shared
    # box turns eight workers into eight times the contention for no throughput.
    cv2.setNumThreads(1)
    _WORKER["calibration"] = load_cache(Path(calibration_path))
    _WORKER["render_root"] = Path(render_root)
    _WORKER["output_root"] = Path(output_root)
    _WORKER["overwrite"] = overwrite


def _run_clip(clip_id: str) -> ClipResult:
    try:
        calibration = _WORKER["calibration"].get(clip_id)
        if calibration is None:
            raise NuRecCalibrationError(f"{clip_id}: not in the calibration cache")
        return rectify_clip(
            clip_id,
            calibration,
            _WORKER["render_root"],
            _WORKER["output_root"],
            _WORKER["overwrite"],
        )
    except Exception:  # a single bad clip must not abandon the other 1,602
        return ClipResult(clip_id=clip_id, images=0, error=traceback.format_exc(limit=3))


def rendered_clip_ids(render_root: Path) -> List[str]:
    return sorted(
        entry.name
        for entry in render_root.iterdir()
        if entry.is_dir() and not entry.name.startswith("_")
    )


def process_clips(
    calibration_path: Path,
    render_root: Path,
    output_root: Path,
    clip_ids: Optional[Iterable[str]] = None,
    workers: int = 8,
    overwrite: bool = False,
    progress_every: int = 25,
) -> Dict[str, Any]:
    clips = list(clip_ids) if clip_ids is not None else rendered_clip_ids(render_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        "clips": len(clips),
        "rectified_clips": 0,
        "skipped_clips": 0,
        "images": 0,
        "failures": {},
    }
    with ProcessPoolExecutor(
        max_workers=max(1, workers),
        initializer=_initialise,
        initargs=(str(calibration_path), str(render_root), str(output_root), overwrite),
    ) as pool:
        for done, result in enumerate(pool.map(_run_clip, clips, chunksize=1), 1):
            if result.error is not None:
                summary["failures"][result.clip_id] = result.error
            elif result.skipped:
                summary["skipped_clips"] += 1
            else:
                summary["rectified_clips"] += 1
                summary["images"] += result.images
            if progress_every and done % progress_every == 0:
                print(
                    f"[{done}/{len(clips)}] rectified={summary['rectified_clips']} "
                    f"skipped={summary['skipped_clips']} images={summary['images']} "
                    f"failed={len(summary['failures'])}",
                    flush=True,
                )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--render-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="rectify only the first N clips")
    parser.add_argument("--clip", action="append", dest="clips", default=None)
    args = parser.parse_args()

    clips = args.clips
    if clips is None:
        clips = rendered_clip_ids(args.render_root)
    if args.limit is not None:
        clips = clips[: args.limit]

    summary = process_clips(
        args.calibration, args.render_root, args.output_root,
        clip_ids=clips, workers=args.workers, overwrite=args.overwrite,
    )
    failures = summary.pop("failures")
    print(json.dumps({**summary, "failed_clips": len(failures)}, indent=2))
    for clip_id, error in list(failures.items())[:5]:
        print(f"--- {clip_id} ---\n{error}")


if __name__ == "__main__":
    main()
