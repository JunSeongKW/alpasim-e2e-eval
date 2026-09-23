"""Check the rectified geometry by projecting annotated 3D boxes into the images.

Everything upstream of this is a claim: that ``T_sensor_rig`` is sensor-to-rig,
that the NuRec rig frame is the log's ego frame, that the rectified images carry
``TARGET_K``.  A transposed rotation or a sign error would still produce a
plausible pinhole image and a plausible projection matrix.  What it would not
produce is boxes that wrap the cars.

The projection is rebuilt from the rewritten log alone -- ``cam_intrinsic`` and
``sensor2lidar_*``, exactly as ``DriveSuprimFeatureBuilder._build_lidar2img``
does -- so this measures what training will read, not what this repository
believes it wrote.

Two outputs, and the second is the one that decides:

* a numeric report.  It catches the gross failures: boxes that all land behind
  the camera, projections that produce NaN, annotated cars that never fall in
  any of the three views.  It cannot catch an error that is consistent between
  the projection and itself, because nothing in it looks at a pixel.
* overlays.  These do look at pixels, and a calibration that is off by a
  rotation or by the 185 px principal point puts the boxes in the sky or in the
  road instead of around the vehicles.

One thing the report makes visible is worth knowing before reading it: the
rectified views are about 64 degrees wide, against the 120 degree lens they came
from, and the side cameras sit ~67 degrees off forward.  Most annotated boxes in
a clip are therefore outside every view -- ``boxes_framed`` is a small fraction
of ``annotations`` even when the calibration is perfect.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

CAMERAS = ("CAM_L0", "CAM_F0", "CAM_R0")

# gt_boxes are [x, y, z, length, width, height, heading] in the ego frame.
_CORNER_SIGNS = np.array(
    [[sx, sy, sz] for sx in (0.5, -0.5) for sy in (0.5, -0.5) for sz in (0.5, -0.5)],
    dtype=np.float64,
)
_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 3),
    (4, 5), (4, 6), (5, 7), (6, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
]

# Points on the road ahead, in the ego frame.
_GROUND_XY = np.stack(
    np.meshgrid(np.arange(6.0, 40.0, 2.0), np.arange(-20.0, 20.0, 2.0)), axis=-1
).reshape(-1, 2)
_GROUND_GRID = np.concatenate([_GROUND_XY, np.zeros((len(_GROUND_XY), 1))], axis=1)


def box_corners(box: Sequence[float]) -> np.ndarray:
    """The eight corners of one annotated box, in the ego frame."""
    x, y, z, length, width, height, heading = (float(v) for v in box[:7])
    local = _CORNER_SIGNS * np.array([length, width, height], dtype=np.float64)
    cos, sin = np.cos(heading), np.sin(heading)
    rotation = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    return local @ rotation.T + np.array([x, y, z])


def lidar2img(camera: Dict[str, Any]) -> np.ndarray:
    """Rebuild the projection as ``_build_lidar2img`` does when sx = sy = 1."""
    sensor_to_ego = np.eye(4, dtype=np.float64)
    sensor_to_ego[:3, :3] = np.asarray(camera["sensor2lidar_rotation"], dtype=np.float64)
    sensor_to_ego[:3, 3] = np.asarray(camera["sensor2lidar_translation"], dtype=np.float64)
    viewpad = np.eye(4, dtype=np.float64)
    viewpad[:3, :3] = np.asarray(camera["cam_intrinsic"], dtype=np.float64)
    return viewpad @ np.linalg.inv(sensor_to_ego)


def project(points: np.ndarray, projection: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project ego-frame points; returns pixels and a mask of points in front."""
    homogeneous = np.concatenate([points, np.ones((len(points), 1))], axis=1)
    projected = homogeneous @ projection.T
    depth = projected[:, 2]
    in_front = depth > 1e-3
    pixels = np.full((len(points), 2), np.nan)
    pixels[in_front] = projected[in_front, :2] / depth[in_front, None]
    return pixels, in_front


def _inside(pixels: np.ndarray, in_front: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
    width, height = size_wh
    return (
        in_front
        & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )


def _ego_yaw(quaternion: Sequence[float]) -> float:
    w, x, y, z = (float(v) for v in quaternion)
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def score_frame(frame: Dict[str, Any], size_wh: Tuple[int, int]) -> Dict[str, int]:
    """Count how the frame's annotations land in the three rectified views."""

    boxes = np.asarray(frame["anns"]["gt_boxes"], dtype=np.float64).reshape(-1, 7)
    totals = {
        "annotations": len(boxes),
        "boxes_in_front": 0,     # centre ahead of some camera
        "boxes_framed": 0,       # centre inside some camera's frame
        "framed_corners": 0,
        "framed_corners_inside": 0,
        "boxes_framed_whole": 0,  # all eight corners inside
        "ground_points": 0,
        "ground_points_below_horizon": 0,
        "nan_projections": 0,
    }
    for camera_name in CAMERAS:
        camera = frame["cams"][camera_name]
        projection = lidar2img(camera)
        principal_y = float(np.asarray(camera["cam_intrinsic"], dtype=np.float64)[1, 2])

        if len(boxes):
            centres, centre_front = project(boxes[:, :3], projection)
            totals["boxes_in_front"] += int(centre_front.sum())
            framed = _inside(centres, centre_front, size_wh)
            totals["boxes_framed"] += int(framed.sum())
            for box in boxes[framed]:
                pixels, in_front = project(box_corners(box), projection)
                totals["nan_projections"] += int(np.isnan(pixels[in_front]).sum())
                inside = _inside(pixels, in_front, size_wh)
                totals["framed_corners"] += 8
                totals["framed_corners_inside"] += int(inside.sum())
                totals["boxes_framed_whole"] += int(bool(inside.all()))

        # The road is flat at z = 0 in the rig frame and every camera sits about
        # a metre above it looking level, so ground ahead has to land *below*
        # the principal row.  A rotation that has been transposed puts it above.
        pixels, in_front = project(_GROUND_GRID, projection)
        visible = _inside(pixels, in_front, size_wh)
        totals["ground_points"] += int(visible.sum())
        totals["ground_points_below_horizon"] += int((visible & (pixels[:, 1] > principal_y)).sum())
    return totals


def draw_frame(
    frame: Dict[str, Any], image_root: Path, future_path: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    """One row of the contact sheet: the three views with boxes drawn on."""
    tiles: List[np.ndarray] = []
    for camera_name in CAMERAS:
        camera = frame["cams"][camera_name]
        image = cv2.imread(str(image_root / camera["data_path"]), cv2.IMREAD_COLOR)
        if image is None:
            return None
        canvas = image.copy()
        height, width = canvas.shape[:2]
        projection = lidar2img(camera)

        boxes = np.asarray(frame["anns"]["gt_boxes"], dtype=np.float64).reshape(-1, 7)
        for box in boxes:
            pixels, in_front = project(box_corners(box), projection)
            if not in_front.all():
                continue  # a box straddling the image plane would need clipping
            if not _inside(pixels, in_front, (width, height)).any():
                continue
            points = pixels.astype(np.int32)
            for start, end in _EDGES:
                cv2.line(canvas, tuple(points[start]), tuple(points[end]),
                         (0, 220, 255), 1, cv2.LINE_AA)

        if future_path is not None and len(future_path) > 1:
            dense = np.concatenate(
                [np.linspace(future_path[i], future_path[i + 1], 12)
                 for i in range(len(future_path) - 1)]
            )
            pixels, in_front = project(dense, projection)
            drawn = [tuple(p.astype(int)) for p, ok in zip(pixels, in_front) if ok]
            for start, end in zip(drawn, drawn[1:]):
                cv2.line(canvas, start, end, (0, 0, 255), 2, cv2.LINE_AA)

        intrinsics = np.asarray(camera["cam_intrinsic"], dtype=np.float64)
        cv2.drawMarker(canvas, (int(round(intrinsics[0, 2])), int(round(intrinsics[1, 2]))),
                       (255, 0, 255), cv2.MARKER_CROSS, 12, 1)
        cv2.putText(canvas, camera_name, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(canvas)
        tiles.append(np.zeros((canvas.shape[0], 3, 3), np.uint8))
    return np.hstack(tiles[:-1])


def future_path_in_ego_frame(frames: Sequence[Dict[str, Any]], index: int, count: int = 16) -> np.ndarray:
    """The ego's own future origins, on the ground, in frame ``index``'s frame."""
    origin = np.asarray(frames[index]["ego2global_translation"], dtype=np.float64)
    yaw = _ego_yaw(frames[index]["ego2global_rotation"])
    cos, sin = np.cos(-yaw), np.sin(-yaw)
    rotation = np.array([[cos, -sin], [sin, cos]])
    path = []
    for other in range(index, min(index + count, len(frames))):
        delta = np.asarray(frames[other]["ego2global_translation"], dtype=np.float64) - origin
        xy = rotation @ delta[:2]
        path.append([xy[0], xy[1], 0.0])
    return np.asarray(path)


def verify(
    log_root: Path,
    image_root: Path,
    out_dir: Optional[Path],
    limit: int,
    overlay_logs: int,
    size_wh: Tuple[int, int],
) -> Dict[str, Any]:
    logs = sorted(log_root.glob("*.pkl"))[:limit]
    if not logs:
        raise FileNotFoundError(f"no rectified logs in {log_root}")
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    totals: Dict[str, int] = {}
    rows: List[np.ndarray] = []
    for index, source in enumerate(logs):
        with source.open("rb") as stream:
            frames = pickle.load(stream)
        for frame in frames:
            for key, value in score_frame(frame, size_wh).items():
                totals[key] = totals.get(key, 0) + value
        if out_dir is not None and index < overlay_logs:
            busiest = max(range(len(frames)), key=lambda i: len(frames[i]["anns"]["gt_boxes"]))
            row = draw_frame(frames[busiest], image_root, future_path_in_ego_frame(frames, busiest))
            if row is not None:
                rows.append(row)

    def ratio(numerator: str, denominator: str) -> float:
        bottom = totals.get(denominator, 0)
        return round(totals.get(numerator, 0) / bottom, 4) if bottom else 0.0

    report: Dict[str, Any] = {
        "logs": len(logs),
        **totals,
        "framed_corner_inside_ratio": ratio("framed_corners_inside", "framed_corners"),
        "framed_whole_ratio": ratio("boxes_framed_whole", "boxes_framed"),
        "ground_below_horizon_ratio": ratio("ground_points_below_horizon", "ground_points"),
    }
    if rows:
        width = max(row.shape[1] for row in rows)
        padded = [np.pad(row, ((0, 0), (0, width - row.shape[1]), (0, 0))) for row in rows]
        sheet = out_dir / "box_projection_overlays.png"
        cv2.imwrite(str(sheet), np.vstack(padded))
        report["overlays"] = str(sheet)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--overlay-logs", type=int, default=4)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(verify(args.log_root, args.image_root, args.out_dir, args.limit,
                            args.overlay_logs, (args.width, args.height)), indent=2))


if __name__ == "__main__":
    main()
