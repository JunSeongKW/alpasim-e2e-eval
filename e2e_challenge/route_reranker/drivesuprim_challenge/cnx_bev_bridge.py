"""AlpaSim -> DriveSuprim ConvNeXt/BEVFormer input bridge."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .rectification import build_ftheta_rectifier_for_resolution
from .schema import RectificationTargetConfig


CAMERA_ORDER = (
    "CAM_L0",
    "CAM_F0",
    "CAM_R0",
)

# DriveSuprim/NAVSIM pinhole camera calibration before the model's
# 1920x1080 -> 512x256 resize.
#
# These values remain environment-overridable so the bridge does not
# hard-wire the assumption into the implementation.
TRAINING_NATIVE_W = int(
    os.environ.get("DRIVESUPRIM_TRAINING_CAMERA_WIDTH", "1920")
)
TRAINING_NATIVE_H = int(
    os.environ.get("DRIVESUPRIM_TRAINING_CAMERA_HEIGHT", "1080")
)

TRAINING_FX = float(
    os.environ.get("DRIVESUPRIM_PINHOLE_FX", "1545.0")
)
TRAINING_FY = float(
    os.environ.get("DRIVESUPRIM_PINHOLE_FY", "1545.0")
)
TRAINING_CX = float(
    os.environ.get("DRIVESUPRIM_PINHOLE_CX", "960.0")
)
TRAINING_CY = float(
    os.environ.get("DRIVESUPRIM_PINHOLE_CY", "560.0")
)

TARGET_W = int(
    os.environ.get("DRIVESUPRIM_BEV_IMAGE_WIDTH", "512")
)
TARGET_H = int(
    os.environ.get("DRIVESUPRIM_BEV_IMAGE_HEIGHT", "256")
)

CAMERA_HISTORY_OFFSETS_US = (
    1_000_000,
    500_000,
    0,
)

STATUS_HISTORY_OFFSETS_US = (
    500_000,
    0,
)

HISTORY_TOLERANCE_US = int(
    os.environ.get(
        "DRIVESUPRIM_HISTORY_TOLERANCE_US",
        "150000",
    )
)


@dataclass
class CameraGeometry:
    rectifier: Any
    lidar2img: np.ndarray
    target_k: np.ndarray


def _target_intrinsics() -> tuple[
    RectificationTargetConfig,
    np.ndarray,
]:
    sx = TARGET_W / float(TRAINING_NATIVE_W)
    sy = TARGET_H / float(TRAINING_NATIVE_H)

    fx = TRAINING_FX * sx
    fy = TRAINING_FY * sy
    cx = TRAINING_CX * sx
    cy = TRAINING_CY * sy

    cfg = RectificationTargetConfig(
        focal_length=(fx, fy),
        principal_point=(cx, cy),
        resolution_hw=(TARGET_H, TARGET_W),
        # DriveSuprim's lidar2img path is a linear pinhole projection.
        # Do not re-introduce OpenCV distortion here.
        radial=(),
        tangential=(),
        thin_prism=(),
    )

    k = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return cfg, k


def _quat_to_rotation_matrix(quat) -> np.ndarray:
    w = float(quat.w)
    x = float(quat.x)
    y = float(quat.y)
    z = float(quat.z)

    norm = math.sqrt(
        w * w + x * x + y * y + z * z
    )

    if norm < 1e-12:
        raise ValueError("zero-norm camera quaternion")

    w /= norm
    x /= norm
    y /= norm
    z /= norm

    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def _camera_to_rig_matrix(camera_proto) -> np.ndarray:
    """Return camera->rig SE(3).

    AlpaSim CameraDefinition describes rig_to_camera as the pose of the
    camera in the rig. Its translation is therefore the camera origin in
    rig coordinates, matching NAVSIM sensor2lidar semantics.
    """

    pose = camera_proto.rig_to_camera

    transform = np.eye(
        4,
        dtype=np.float64,
    )

    transform[:3, :3] = (
        _quat_to_rotation_matrix(pose.quat)
    )

    transform[:3, 3] = [
        float(pose.vec.x),
        float(pose.vec.y),
        float(pose.vec.z),
    ]

    return transform


def build_camera_geometry(
    camera_proto,
) -> CameraGeometry:
    model_type = (
        camera_proto
        .intrinsics
        .WhichOneof("camera_param")
    )

    if model_type != "ftheta_param":
        raise ValueError(
            f"{camera_proto.logical_id}: "
            f"expected ftheta_param, got {model_type}"
        )

    source_hw = (
        int(camera_proto.intrinsics.resolution_h),
        int(camera_proto.intrinsics.resolution_w),
    )

    target_cfg, target_k = (
        _target_intrinsics()
    )

    rectifier = (
        build_ftheta_rectifier_for_resolution(
            camera_proto=camera_proto,
            target_cfg=target_cfg,
            source_resolution_hw=source_hw,
        )
    )

    # AlpaSim payload gives the camera pose in rig coordinates:
    #
    #     camera -> rig
    #
    # DriveSuprim expects:
    #
    #     lidar/ego(rig) -> camera
    #
    # before applying the pinhole matrix.
    camera_to_rig = (
        _camera_to_rig_matrix(
            camera_proto
        )
    )

    rig_to_camera = np.linalg.inv(
        camera_to_rig
    )

    k4 = np.eye(
        4,
        dtype=np.float64,
    )

    k4[:3, :3] = (
        target_k.astype(np.float64)
    )

    lidar2img = (
        k4 @ rig_to_camera
    ).astype(np.float32)

    # Geometry sanity check:
    # a point 10 m along the camera optical axis must project
    # to the virtual pinhole principal point.
    optical_axis_rig = (
        camera_to_rig[:3, 2]
    )

    point_rig = (
        camera_to_rig[:3, 3]
        + 10.0 * optical_axis_rig
    )

    point_h = np.concatenate(
        (
            point_rig,
            np.array([1.0]),
        )
    )

    projected = (
        lidar2img @ point_h
    )

    if projected[2] <= 0.0:
        raise RuntimeError(
            f"{camera_proto.logical_id}: "
            "optical-axis sanity point has "
            f"non-positive depth {projected[2]}"
        )

    uv = (
        projected[:2]
        / projected[2]
    )

    expected = np.array(
        [
            target_k[0, 2],
            target_k[1, 2],
        ],
        dtype=np.float64,
    )

    error_px = float(
        np.linalg.norm(
            uv - expected
        )
    )

    if error_px > 1e-2:
        raise RuntimeError(
            f"{camera_proto.logical_id}: "
            "camera transform sanity check failed: "
            f"uv={uv}, expected={expected}, "
            f"error={error_px:.6f}px"
        )

    return CameraGeometry(
        rectifier=rectifier,
        lidar2img=lidar2img,
        target_k=target_k,
    )


def rectify_image(
    geometry: CameraGeometry,
    image: np.ndarray,
) -> np.ndarray:
    result = geometry.rectifier.rectify(
        image
    )

    if result.shape[:2] != (
        TARGET_H,
        TARGET_W,
    ):
        raise RuntimeError(
            "unexpected rectified image size: "
            f"{result.shape}; expected "
            f"({TARGET_H}, {TARGET_W}, 3)"
        )

    return np.ascontiguousarray(
        result
    )


def select_history_by_offsets(
    items: Sequence,
    *,
    anchor_timestamp_us: int,
    offsets_us: Sequence[int],
    tolerance_us: int = HISTORY_TOLERANCE_US,
):
    """Select history items nearest anchor-offset timestamps.

    offsets_us must be oldest -> newest, e.g.
    (1_000_000, 500_000, 0).
    """

    if not items:
        return None

    selected = []

    for offset_us in offsets_us:
        target_us = (
            int(anchor_timestamp_us)
            - int(offset_us)
        )

        item = min(
            items,
            key=lambda value: abs(
                int(value.timestamp_us)
                - target_us
            ),
        )

        error_us = abs(
            int(item.timestamp_us)
            - target_us
        )

        if error_us > tolerance_us:
            return None

        selected.append(item)

    timestamps = [
        int(item.timestamp_us)
        for item in selected
    ]

    # Do not silently reuse one 10-Hz frame as multiple 0.5-s history slots.
    if len(set(timestamps)) != len(timestamps):
        return None

    if timestamps != sorted(timestamps):
        raise RuntimeError(
            "selected history is not chronological: "
            f"{timestamps}"
        )

    return selected


def _yaw_from_quaternion(quat) -> float:
    return math.atan2(
        2.0
        * (
            float(quat.w) * float(quat.z)
            + float(quat.x) * float(quat.y)
        ),
        1.0
        - 2.0
        * (
            float(quat.y) ** 2
            + float(quat.z) ** 2
        ),
    )


def _nearest_pose(
    poses: Sequence,
    timestamp_us: int,
    tolerance_us: int,
):
    if not poses:
        return None

    pose = min(
        poses,
        key=lambda item: abs(
            int(item.timestamp_us)
            - int(timestamp_us)
        ),
    )

    if (
        abs(
            int(pose.timestamp_us)
            - int(timestamp_us)
        )
        > tolerance_us
    ):
        return None

    return pose


def build_relative_ego_poses(
    frame_timestamps_us: Sequence[int],
    pose_history: Sequence,
    *,
    tolerance_us: int = HISTORY_TOLERANCE_US,
) -> np.ndarray | None:
    """Build NAVSIM-style ego poses relative to the latest frame."""

    matched = []

    for timestamp_us in frame_timestamps_us:
        pose = _nearest_pose(
            pose_history,
            int(timestamp_us),
            tolerance_us,
        )

        if pose is None:
            return None

        matched.append(pose)

    current = matched[-1]

    current_x = float(
        current.pose.vec.x
    )
    current_y = float(
        current.pose.vec.y
    )
    current_yaw = (
        _yaw_from_quaternion(
            current.pose.quat
        )
    )

    c = math.cos(current_yaw)
    s = math.sin(current_yaw)

    relative = []

    for pose in matched:
        x = float(pose.pose.vec.x)
        y = float(pose.pose.vec.y)
        yaw = _yaw_from_quaternion(
            pose.pose.quat
        )

        dx = x - current_x
        dy = y - current_y

        # global -> current ego coordinates
        x_rel = c * dx + s * dy
        y_rel = -s * dx + c * dy

        yaw_rel = math.atan2(
            math.sin(yaw - current_yaw),
            math.cos(yaw - current_yaw),
        )

        relative.append(
            [
                x_rel,
                y_rel,
                yaw_rel,
            ]
        )

    result = np.asarray(
        relative,
        dtype=np.float32,
    )

    # Last/current pose should be numerically zero.
    if not np.allclose(
        result[-1],
        0.0,
        atol=1e-4,
    ):
        raise RuntimeError(
            "current BEV ego pose is not zero: "
            f"{result[-1]}"
        )

    return result


def stack_lidar2img(
    camera_geometry: dict[str, CameraGeometry],
    num_frames: int,
) -> np.ndarray:
    per_frame = np.stack(
        [
            camera_geometry[name].lidar2img
            for name in CAMERA_ORDER
        ],
        axis=0,
    )

    return np.stack(
        [per_frame] * int(num_frames),
        axis=0,
    ).astype(np.float32)
