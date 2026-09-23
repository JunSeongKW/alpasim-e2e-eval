"""Per-clip camera calibration, read out of the NuRec USDZ archives.

The NAVSIM logs carry one fixed nuPlan pinhole for every clip.  The renders they
point at were produced by a different rig: an f-theta lens whose principal point
sits 185 px below nuPlan's, mounted at a pose that changes from clip to clip.
Both live in the clip's ``rig_trajectories.json``, which is an ordinary member of
the USDZ zip -- reading it costs about 0.1 s and does not unpack the 1.9 GB
volume next to it.

Two things are extracted per camera:

``T_sensor_rig``
    Despite the name this is the *sensor-to-rig* transform: its columns are the
    camera's optical axes expressed in rig coordinates and its translation is
    the camera origin in the rig frame.  That is exactly NAVSIM's
    ``sensor2lidar`` convention, and exactly what the challenge driver feeds
    ``lidar2img`` at evaluation time (``cnx_bev_bridge._camera_to_rig_matrix``).

``camera_model``
    The f-theta polynomials the rectifier needs, in the JSON spelling that
    :func:`ftheta_proto.available_camera_from_usdz` consumes.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

# NAVSIM camera identifier -> the logical sensor name NuRec renders under.
# The same mapping the renderer used for its output directories.
CAMERA_SENSORS: Dict[str, str] = {
    "CAM_L0": "camera_cross_left_120fov",
    "CAM_F0": "camera_front_wide_120fov",
    "CAM_R0": "camera_cross_right_120fov",
}

CALIBRATION_MEMBER = "rig_trajectories.json"
FORMAT_VERSION = 1


class NuRecCalibrationError(ValueError):
    """Raised when a USDZ does not carry usable calibration for a clip."""


@dataclass(frozen=True)
class CameraCalibration:
    """One camera's pose and lens model, as stored in the clip."""

    sensor_name: str
    sensor_to_rig: np.ndarray  # 4x4, camera optical frame -> rig frame
    camera_model: Dict[str, Any]  # ftheta block, ready for available_camera_from_usdz

    @property
    def rotation(self) -> np.ndarray:
        """``sensor2lidar_rotation``: optical axes expressed in the rig frame."""
        return self.sensor_to_rig[:3, :3]

    @property
    def translation(self) -> np.ndarray:
        """``sensor2lidar_translation``: camera origin in the rig frame."""
        return self.sensor_to_rig[:3, 3]

    @property
    def native_resolution_wh(self) -> Tuple[int, int]:
        width, height = self.camera_model["parameters"]["resolution"]
        return int(width), int(height)


@dataclass(frozen=True)
class ClipCalibration:
    """The three rendered cameras of one clip."""

    clip_id: str
    cameras: Dict[str, CameraCalibration]

    def __getitem__(self, navsim_name: str) -> CameraCalibration:
        return self.cameras[navsim_name]

    def to_json(self) -> Dict[str, Any]:
        return {
            name: {
                "sensor_name": camera.sensor_name,
                "T_sensor_rig": camera.sensor_to_rig.tolist(),
                "camera_model": camera.camera_model,
            }
            for name, camera in self.cameras.items()
        }

    @classmethod
    def from_json(cls, clip_id: str, payload: Mapping[str, Any]) -> "ClipCalibration":
        cameras = {
            name: CameraCalibration(
                sensor_name=str(entry["sensor_name"]),
                sensor_to_rig=np.asarray(entry["T_sensor_rig"], dtype=np.float64),
                camera_model=dict(entry["camera_model"]),
            )
            for name, entry in payload.items()
        }
        return cls(clip_id=clip_id, cameras=cameras)


def _sensor_of(key: str, entry: Mapping[str, Any]) -> str:
    """The logical sensor name, which the calibration keys suffix with the clip."""
    name = entry.get("logical_sensor_name")
    return str(name) if name else key.split("@", 1)[0]


def read_usdz_calibration(usdz_path: Path, clip_id: Optional[str] = None) -> ClipCalibration:
    """Read the three rendered cameras' calibration out of one clip's USDZ."""

    clip_id = clip_id or Path(usdz_path).stem
    try:
        with zipfile.ZipFile(usdz_path) as archive:
            payload = json.loads(archive.read(CALIBRATION_MEMBER))
    except (OSError, KeyError, zipfile.BadZipFile, json.JSONDecodeError) as error:
        raise NuRecCalibrationError(f"{clip_id}: cannot read {CALIBRATION_MEMBER}: {error}") from error

    calibrations = payload.get("camera_calibrations") or {}
    by_sensor = {_sensor_of(key, entry): entry for key, entry in calibrations.items()}

    cameras: Dict[str, CameraCalibration] = {}
    for navsim_name, sensor_name in CAMERA_SENSORS.items():
        entry = by_sensor.get(sensor_name)
        if entry is None:
            raise NuRecCalibrationError(
                f"{clip_id}: {sensor_name} is not calibrated in {CALIBRATION_MEMBER} "
                f"(have {sorted(by_sensor)})"
            )
        transform = np.asarray(entry["T_sensor_rig"], dtype=np.float64)
        if transform.shape != (4, 4):
            raise NuRecCalibrationError(f"{clip_id}: {sensor_name}: T_sensor_rig is not 4x4")
        model = entry.get("camera_model") or {}
        if model.get("type") != "ftheta":
            raise NuRecCalibrationError(
                f"{clip_id}: {sensor_name}: expected an ftheta camera, got {model.get('type')!r}"
            )
        cameras[navsim_name] = CameraCalibration(
            sensor_name=sensor_name, sensor_to_rig=transform, camera_model=model
        )
    return ClipCalibration(clip_id=clip_id, cameras=cameras)


def usdz_path_for(usdz_root: Path, clip_id: str) -> Path:
    """The layout the sample set uses: ``{root}/{clip}/{clip}.usdz``."""
    return usdz_root / clip_id / f"{clip_id}.usdz"


def clip_ids_in(usdz_root: Path) -> List[str]:
    return sorted(entry.name for entry in usdz_root.iterdir() if entry.is_dir())


def _read_one(task: Tuple[Path, str]) -> Tuple[str, Optional[Dict[str, Any]], Optional[str]]:
    usdz_path, clip_id = task
    try:
        return clip_id, read_usdz_calibration(usdz_path, clip_id).to_json(), None
    except NuRecCalibrationError as error:
        return clip_id, None, str(error)


def build_cache(
    usdz_root: Path,
    output: Path,
    clip_ids: Optional[Iterable[str]] = None,
    workers: int = 8,
) -> Dict[str, Any]:
    """Extract every clip's calibration into a single JSON file.

    Re-running keeps clips that are already cached, so an interrupted pass costs
    only the clips it had not reached.
    """

    clips = list(clip_ids) if clip_ids is not None else clip_ids_in(usdz_root)
    cached: Dict[str, Any] = {}
    if output.is_file():
        existing = json.loads(output.read_text())
        if existing.get("format_version") == FORMAT_VERSION:
            cached = existing.get("clips", {})

    pending = [(usdz_path_for(usdz_root, clip), clip) for clip in clips if clip not in cached]
    failures: Dict[str, str] = {}
    if pending:
        with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
            for clip_id, payload, error in pool.map(_read_one, pending, chunksize=4):
                if payload is None:
                    failures[clip_id] = error or "unknown error"
                else:
                    cached[clip_id] = payload

    document = {
        "format_version": FORMAT_VERSION,
        "usdz_root": str(usdz_root),
        "cameras": CAMERA_SENSORS,
        "clips": {clip: cached[clip] for clip in sorted(cached)},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(document) + "\n", encoding="utf-8")
    temporary.replace(output)
    return {
        "clips_requested": len(clips),
        "clips_cached": len(cached),
        "clips_read_now": len(pending) - len(failures),
        "failures": failures,
    }


def load_cache(path: Path) -> Dict[str, ClipCalibration]:
    """Read a cache written by :func:`build_cache`."""

    document = json.loads(Path(path).read_text())
    if document.get("format_version") != FORMAT_VERSION:
        raise NuRecCalibrationError(
            f"{path}: format_version {document.get('format_version')!r} is not {FORMAT_VERSION}"
        )
    return {
        clip_id: ClipCalibration.from_json(clip_id, payload)
        for clip_id, payload in document["clips"].items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usdz-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--clip", action="append", dest="clips", default=None,
                        help="restrict to these clip ids (repeatable)")
    args = parser.parse_args()
    print(json.dumps(build_cache(args.usdz_root, args.output, args.clips, args.workers), indent=2))


if __name__ == "__main__":
    main()
