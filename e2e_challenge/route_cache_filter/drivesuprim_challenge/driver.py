#!/usr/bin/env python
"""AlpaSim e2e challenge driver backed by DriveSuprim."""

from __future__ import annotations

import logging
import math
import os
import pickle
import signal
import threading
import time
from concurrent import futures
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

import grpc
import numpy as np
import torch
from alpasim_grpc import API_VERSION_MESSAGE
from alpasim_grpc.v0 import (
    common_pb2,
    egodriver_pb2,
    egodriver_pb2_grpc,
    sensorsim_pb2,
)
from PIL import Image

from vavam_challenge.rectification import (
    RectificationTargetConfig,
    build_ftheta_rectifier_for_resolution,
)

from .cnx_bev_bridge import (
    CAMERA_HISTORY_OFFSETS_US,
    CAMERA_ORDER,
    HISTORY_TOLERANCE_US,
    STATUS_HISTORY_OFFSETS_US,
    CameraGeometry,
    build_camera_geometry,
    build_relative_ego_poses,
    rectify_image,
    select_history_by_offsets,
    stack_lidar2img,
)
from .route_cache import RouteCache

LOGGER = logging.getLogger(
    "drivesuprim_challenge_driver"
)
LOGGER.setLevel(logging.INFO)


_CAMERA_ALIASES = {
    "CAM_L0": "CAM_L0",
    "CAM_F0": "CAM_F0",
    "CAM_R0": "CAM_R0",
    "CAMERA_CROSS_LEFT_120FOV": "CAM_L0",
    "CAMERA_FRONT_WIDE_120FOV": "CAM_F0",
    "CAMERA_CROSS_RIGHT_120FOV": "CAM_R0",
}



_BEV_CAMERA_ORDER = (
    "CAM_L0",
    "CAM_F0",
    "CAM_R0",
)

# ------------------------------------------------------------------
# Virtual pinhole target. Select the profile per model image so the shared
# challenge adapter can preserve each checkpoint's training geometry.
# ------------------------------------------------------------------

_VIRTUAL_PINHOLE_WIDTH = 512
_VIRTUAL_PINHOLE_HEIGHT = 256

_PINHOLE_PROFILE = os.getenv(
    "DRIVESUPRIM_PINHOLE_PROFILE", "K377_512X256"
).strip().upper()
_PINHOLE_PROFILES = {
    "K377_512X256": (377.0, 377.0, 256.0, 128.0),
    # Original NAVSIM geometry after resizing 1920x1080 to 512x256.
    "NAVSIM_512X256": (
        412.0,
        366.22222222222223,
        256.0,
        132.74074074074073,
    ),
}
if _PINHOLE_PROFILE not in _PINHOLE_PROFILES:
    raise ValueError(
        "Unsupported DRIVESUPRIM_PINHOLE_PROFILE="
        f"{_PINHOLE_PROFILE!r}; expected one of "
        f"{sorted(_PINHOLE_PROFILES)}"
    )
(
    _VIRTUAL_FX,
    _VIRTUAL_FY,
    _VIRTUAL_CX,
    _VIRTUAL_CY,
) = _PINHOLE_PROFILES[_PINHOLE_PROFILE]


def _rectification_target_config() -> RectificationTargetConfig:
    return RectificationTargetConfig(
        focal_length=(
            _VIRTUAL_FX,
            _VIRTUAL_FY,
        ),
        principal_point=(
            _VIRTUAL_CX,
            _VIRTUAL_CY,
        ),
        resolution_hw=(
            _VIRTUAL_PINHOLE_HEIGHT,
            _VIRTUAL_PINHOLE_WIDTH,
        ),
        radial=(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        tangential=(
            0.0,
            0.0,
        ),
        thin_prism=(
            0.0,
            0.0,
            0.0,
            0.0,
        ),
        max_overscan_scale=2.0,
        safety_margin_px=4,
    )


_VIRTUAL_K = np.array(
    [
        [_VIRTUAL_FX, 0.0, _VIRTUAL_CX],
        [0.0, _VIRTUAL_FY, _VIRTUAL_CY],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)


def _quat_to_rotation_matrix(quat) -> np.ndarray:
    """Return a 3x3 rotation matrix from common.Quat (w,x,y,z)."""

    w = float(quat.w)
    x = float(quat.x)
    y = float(quat.y)
    z = float(quat.z)

    norm = math.sqrt(
        w * w
        + x * x
        + y * y
        + z * z
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


def _camera_pose_to_rig2cam(camera) -> np.ndarray:
    """Convert AlpaSim camera pose into rig->camera transform.

    The translation contained in AvailableCamera.rig_to_camera is the
    physical sensor position in the rig frame. Therefore the Pose is used
    as camera->rig and inverted here, matching DriveSuprim's
    cam2lidar -> inverse -> lidar2cam logic.
    """

    pose = camera.rig_to_camera

    cam2rig = np.eye(
        4,
        dtype=np.float64,
    )

    cam2rig[:3, :3] = (
        _quat_to_rotation_matrix(
            pose.quat
        )
    )

    cam2rig[:3, 3] = np.array(
        [
            float(pose.vec.x),
            float(pose.vec.y),
            float(pose.vec.z),
        ],
        dtype=np.float64,
    )

    return np.linalg.inv(
        cam2rig
    )


def _build_virtual_lidar2img(camera) -> np.ndarray:
    """Build DriveSuprim-compatible rig/ego -> pinhole image projection."""

    rig2cam = _camera_pose_to_rig2cam(
        camera
    )

    viewpad = np.eye(
        4,
        dtype=np.float64,
    )

    viewpad[:3, :3] = (
        _VIRTUAL_K.astype(
            np.float64
        )
    )

    return (
        viewpad
        @ rig2cam
    ).astype(np.float32)


def _projection_probe(
    lidar2img: np.ndarray,
) -> tuple[float, float, float]:
    """Project a point 10 m in front of the rig for calibration sanity."""

    point = np.array(
        [
            10.0,
            0.0,
            0.0,
            1.0,
        ],
        dtype=np.float64,
    )

    projected = (
        lidar2img.astype(
            np.float64
        )
        @ point
    )

    depth = float(
        projected[2]
    )

    if abs(depth) < 1e-9:
        return (
            float("nan"),
            float("nan"),
            depth,
        )

    return (
        float(projected[0] / depth),
        float(projected[1] / depth),
        depth,
    )


def _nearest_pose(
    poses,
    timestamp_us: int,
):
    if not poses:
        raise RuntimeError(
            "no ego poses available"
        )

    return min(
        poses,
        key=lambda pose: abs(
            int(pose.timestamp_us)
            - int(timestamp_us)
        ),
    )


def _build_bev_ego_pose(
    frames,
    poses,
) -> np.ndarray:
    """Build [T,3] historical ego poses relative to current frame."""

    matched = [
        _nearest_pose(
            poses,
            frame.timestamp_us,
        )
        for frame in frames
    ]

    current = matched[-1]

    current_x = float(
        current.pose.vec.x
    )
    current_y = float(
        current.pose.vec.y
    )
    current_yaw = _yaw(
        current.pose.quat
    )

    c = math.cos(
        current_yaw
    )
    s = math.sin(
        current_yaw
    )

    result = []

    for pose in matched:
        dx = (
            float(pose.pose.vec.x)
            - current_x
        )
        dy = (
            float(pose.pose.vec.y)
            - current_y
        )

        # global -> current ego frame
        x_rel = (
            c * dx
            + s * dy
        )
        y_rel = (
            -s * dx
            + c * dy
        )

        yaw = _yaw(
            pose.pose.quat
        )

        yaw_rel = math.atan2(
            math.sin(
                yaw - current_yaw
            ),
            math.cos(
                yaw - current_yaw
            ),
        )

        result.append(
            [
                x_rel,
                y_rel,
                yaw_rel,
            ]
        )

    return np.asarray(
        result,
        dtype=np.float32,
    )


_RUNTIME_DIRS = {
    "XDG_CACHE_HOME": "/tmp/.cache",
    "TORCH_HOME": "/tmp/torch",
    "MPLCONFIGDIR": "/tmp/matplotlib",
    "CUDA_CACHE_PATH": "/tmp/nv",
    "NUMBA_CACHE_DIR": "/tmp/numba",
}


for _key, _value in _RUNTIME_DIRS.items():
    os.environ.setdefault(
        _key,
        _value,
    )


os.environ.setdefault(
    "OMP_NUM_THREADS",
    "1",
)
os.environ.setdefault(
    "MKL_NUM_THREADS",
    "1",
)


def _env_flag(
    name: str,
    default: bool = False,
) -> bool:
    value = os.environ.get(
        name,
        "1" if default else "0",
    )

    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass
class CachedPlan:
    created_time_us: int
    times_s: np.ndarray
    positions_xy: np.ndarray
    yaws: np.ndarray


@dataclass
class CameraFrame:
    """One synchronized L0/F0/R0 camera observation."""

    timestamp_us: int
    images: dict[str, np.ndarray]


@dataclass
class StatusFrame:
    """One timestamped DriveSuprim ego-status vector."""

    timestamp_us: int
    values: np.ndarray


@dataclass
class SessionState:
    session_uuid: str = ""

    latest_pose: common_pb2.PoseAtTime | None = None

    poses: list[
        common_pb2.PoseAtTime
    ] = field(
        default_factory=list
    )

    dynamic_states: list[
        tuple[int, common_pb2.DynamicState]
    ] = field(
        default_factory=list
    )

    current_images: dict[
        str,
        np.ndarray,
    ] = field(
        default_factory=dict
    )

    current_frame_times: dict[
        str,
        int,
    ] = field(
        default_factory=dict
    )

    # BEVFormer uses current + two history frames.
    camera_history: list[
        CameraFrame
    ] = field(
        default_factory=list
    )

    # Keep dense 10-Hz statuses, then sample NAVSIM's 0.5-s history
    # at inference time.
    status_history: list[
        StatusFrame
    ] = field(
        default_factory=list
    )

    # Calibration received in DriveSessionRequest.
    camera_specs: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    rectifiers: dict[
        str,
        Any,
    ] = field(
        default_factory=dict
    )

    rectifier_source_resolutions: dict[
        str,
        tuple[int, int],
    ] = field(
        default_factory=dict
    )

    lidar2img_by_camera: dict[
        str,
        np.ndarray,
    ] = field(
        default_factory=dict
    )

    camera_geometry: dict[
        str,
        CameraGeometry,
    ] = field(
        default_factory=dict
    )

    route_command: np.ndarray = field(
        default_factory=lambda: np.array(
            [0.0, 1.0, 0.0, 0.0],
            dtype=np.float32,
        )
    )

    latest_route: egodriver_pb2.Route | None = None
    cached_plan: CachedPlan | None = None
    latest_prediction: Any | None = None
    debug_vocab_sent: bool = False

    # Route messages arrive in the rig frame of their own timestamp, and the
    # runtime fires them as a task alongside the ego pose rather than after it,
    # so the pose needed to place a route in the persistent frame may not have
    # landed yet. Hold the raw message until its pose shows up instead of
    # guessing with the newest one.
    route_cache: RouteCache = field(default_factory=RouteCache)
    pending_routes: list[tuple[int, np.ndarray]] = field(default_factory=list)


def _route_waypoints_for_policy(
    route: egodriver_pb2.Route | None,
) -> np.ndarray:
    """Return the runtime's prepared 20-slot route, preserving NaN padding."""

    points = np.full((20, 3), np.nan, dtype=np.float32)
    if route is None:
        return points

    count = min(20, len(route.waypoints))
    for index, point in enumerate(route.waypoints[:count]):
        points[index] = (float(point.x), float(point.y), float(point.z))
    return points


def _route_cache_enabled() -> bool:
    return _env_flag("DRIVESUPRIM_ROUTE_CACHE", default=False)


def _route_corridor_m() -> float:
    """Corridor half-width the filter rejects at, matching the scorer's 4 m."""
    try:
        return float(os.environ.get("DRIVESUPRIM_ROUTE_CORRIDOR_M", "4.0"))
    except ValueError:
        return 4.0


# A route timestamp and an ego-pose timestamp for the same step are the same
# number; anything further apart than half a control step is a different step.
_ROUTE_POSE_TOLERANCE_US = 50_000


def _resolve_pending_routes(session) -> None:
    """Move route messages into the cache as soon as their pose is available.

    Called from `drive`, by which point the step's pose has certainly arrived,
    so in practice nothing waits longer than one step. A message whose pose
    never shows up is dropped rather than placed with the wrong pose: a route
    put down at the wrong position would corrupt the cache permanently, while
    a dropped one costs a single 4 m increment of coverage.
    """
    if not session.pending_routes or not session.poses:
        return

    newest_pose_us = int(session.poses[-1].timestamp_us)
    still_pending = []
    for timestamp_us, waypoints in session.pending_routes:
        pose = _nearest_pose(session.poses, timestamp_us)
        offset = abs(int(pose.timestamp_us) - timestamp_us)
        if offset <= _ROUTE_POSE_TOLERANCE_US:
            session.route_cache.add(
                waypoints,
                float(pose.pose.vec.x),
                float(pose.pose.vec.y),
                _yaw(pose.pose.quat),
            )
        elif timestamp_us > newest_pose_us:
            still_pending.append((timestamp_us, waypoints))   # pose not here yet
        else:
            LOGGER.warning(
                "dropping route at %d: nearest ego pose is %d us away",
                timestamp_us,
                offset,
            )
    session.pending_routes = still_pending


def _use_navsim_2hz_history() -> bool:
    return _env_flag("DRIVESUPRIM_NAVSIM_2HZ_HISTORY", default=False)


def _camera_model_summary(
    camera,
) -> str:
    spec = camera.intrinsics

    model_type = spec.WhichOneof(
        "camera_param"
    )

    base = (
        f"model={model_type} "
        f"resolution={spec.resolution_w}x"
        f"{spec.resolution_h}"
    )

    if model_type == "opencv_pinhole_param":
        p = spec.opencv_pinhole_param

        return (
            f"{base} "
            f"fx={p.focal_length_x:.6f} "
            f"fy={p.focal_length_y:.6f} "
            f"cx={p.principal_point_x:.6f} "
            f"cy={p.principal_point_y:.6f} "
            f"radial={list(p.radial_coeffs)} "
            f"tangential={list(p.tangential_coeffs)}"
        )

    if model_type == "opencv_fisheye_param":
        p = spec.opencv_fisheye_param

        return (
            f"{base} "
            f"fx={p.focal_length_x:.6f} "
            f"fy={p.focal_length_y:.6f} "
            f"cx={p.principal_point_x:.6f} "
            f"cy={p.principal_point_y:.6f} "
            f"max_angle={p.max_angle:.6f} "
            f"radial={list(p.radial_coeffs)}"
        )

    if model_type == "ftheta_param":
        p = spec.ftheta_param

        return (
            f"{base} "
            f"cx={p.principal_point_x:.6f} "
            f"cy={p.principal_point_y:.6f} "
            f"max_angle={p.max_angle:.6f} "
            f"pixeldist_to_angle_degree="
            f"{len(p.pixeldist_to_angle_poly)} "
            f"angle_to_pixeldist_degree="
            f"{len(p.angle_to_pixeldist_poly)}"
        )

    return base


def _pose_summary(
    pose,
) -> str:
    return (
        "translation="
        f"({pose.vec.x:.6f},"
        f"{pose.vec.y:.6f},"
        f"{pose.vec.z:.6f}) "
        "quat="
        f"(w={pose.quat.w:.6f},"
        f"x={pose.quat.x:.6f},"
        f"y={pose.quat.y:.6f},"
        f"z={pose.quat.z:.6f})"
    )


class PolicyHandle:
    def __init__(
        self,
        **policy_kwargs: str,
    ) -> None:
        self._kwargs = policy_kwargs
        self._lock = threading.Lock()
        self._policy: Any | None = None
        self._error: BaseException | None = None

    def start(self) -> None:
        threading.Thread(
            target=self._load,
            name="drivesuprim-loader",
            daemon=True,
        ).start()

    def _load(self) -> None:
        try:
            from .policy import DriveSuprimPolicy

            policy = DriveSuprimPolicy(
                **self._kwargs
            )

        except BaseException as exc:
            with self._lock:
                self._error = exc

            LOGGER.exception(
                "DriveSuprim policy load failed"
            )
            return

        with self._lock:
            self._policy = policy

        print(
            "[DriveSuprim] policy load complete",
            flush=True,
        )

        LOGGER.info(
            "DriveSuprim policy load complete"
        )

    def get(self) -> Any | None:
        with self._lock:
            return self._policy

    def error(self) -> BaseException | None:
        with self._lock:
            return self._error


class DriveSuprimChallengeDriver(
    egodriver_pb2_grpc.EgodriverServiceServicer
):
    def __init__(
        self,
        policy_handle: PolicyHandle,
        inference_interval_us: int,
    ) -> None:
        self._policy_handle = policy_handle
        self._inference_interval_us = (
            inference_interval_us
        )

        self._sessions: dict[
            str,
            SessionState,
        ] = {}

        self._lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self._server: grpc.Server | None = None
        self._ego_footprint_logged = False
        self._route_center_dx = 0.0

    def attach_server(
        self,
        server: grpc.Server,
    ) -> None:
        self._server = server

    def _apply_ego_footprint(
        self,
        vehicle,
    ) -> None:
        """Size the model's feasibility gate from the reported ego box.

        The gate rejects candidate trajectories whose ego rectangle overlaps a
        detected agent or leaves the drivable segmentation, and it sizes that
        rectangle from `feasibility_ego_length` / `_width` -- shipped as a
        hardcoded 4.6 x 1.9 m. Evaluation scores collisions and offroad against
        the box the rollout spec carries, so any other box makes the gate
        disagree with the scorer it exists to satisfy.

        A simulator that predates the field sends nothing, which decodes as 0.0;
        the hardcoded defaults are kept in that case.
        """
        if not _env_flag(
            "DRIVESUPRIM_EGO_FOOTPRINT_FROM_API",
            default=True,
        ):
            return

        box = getattr(vehicle, "bounding_box", None)
        length = float(getattr(box, "size_x", 0.0) or 0.0)
        width = float(getattr(box, "size_y", 0.0) or 0.0)

        if length <= 0.0 or width <= 0.0:
            if not self._ego_footprint_logged:
                self._ego_footprint_logged = True
                print(
                    "[DriveSuprim] ego footprint not supplied by the "
                    "simulator; keeping built-in defaults",
                    flush=True,
                )
            return

        # Guard against a malformed spec silently shrinking or inflating the
        # gate: a wrong box here is worse than the default one.
        if not (1.0 <= length <= 8.0 and 0.5 <= width <= 4.0):
            LOGGER.warning(
                "ignoring implausible ego footprint %.3f x %.3f m",
                length,
                width,
            )
            return

        policy = self._policy_handle.get()
        config = getattr(policy, "_config", None)
        if config is None:
            return

        previous = (
            getattr(config, "feasibility_ego_length", None),
            getattr(config, "feasibility_ego_width", None),
        )
        if previous == (length, width):
            return

        # One model instance serves every session in this process, so the gate
        # geometry is process-wide. Concurrent sessions on differently sized
        # vehicles would fight over it; the challenge topology runs one rollout
        # per driver replica, so log rather than serialise.
        if self._ego_footprint_logged:
            LOGGER.warning(
                "ego footprint changed mid-process: %s -> (%.3f, %.3f)",
                previous,
                length,
                width,
            )

        config.feasibility_ego_length = length
        config.feasibility_ego_width = width
        self._ego_footprint_logged = True

        # common.Pose carries its translation as `vec`, not `translation`.
        # The rotation is the identity in AlpaSim -- the rig and the box share
        # their axes and differ only in origin -- so only the translation is
        # needed to place the footprint.
        pose = getattr(vehicle, "rig_to_bounding_box", None)
        vec = getattr(pose, "vec", None)
        offset = (
            float(getattr(vec, "x", 0.0) or 0.0),
            float(getattr(vec, "y", 0.0) or 0.0),
            float(getattr(vec, "z", 0.0) or 0.0),
        )

        # The gate applies the translation only, which is exact while the two
        # frames share their axes. A simulator that ever sends a real rotation
        # here would otherwise be honoured silently and wrongly, so say so.
        # Identity is (w=1, x=y=z=0); |w| < 1 means some rotation is present.
        quat = getattr(pose, "quat", None)
        quat_w = float(getattr(quat, "w", 1.0) or 0.0)
        rotation_is_identity = abs(quat_w) >= 0.999
        if not rotation_is_identity:
            LOGGER.warning(
                "rig->box transform carries a rotation (quat.w=%.6f); the "
                "feasibility gate only applies the translation, so the ego "
                "rectangle will be misplaced. Refusing the offset.",
                quat_w,
            )

        # Off by default: the checkpoint was trained with the gate centred on
        # the pose, so moving the rectangle is a train/inference change that has
        # to earn its place in an A/B, not a free correction.
        # Only the forward component is applied. Across 100 curated_val clips
        # the lateral offset was exactly 0 every time -- the rig sits on the
        # vehicle centreline -- so a lateral term would be dead weight. It is
        # still checked, because a silent mismatch is the failure worth
        # preventing: if one ever shows up the offset is refused rather than
        # applied half-right. z is ignored outright; both gates are planar.
        lateral_is_zero = abs(offset[1]) <= 1e-3
        if not lateral_is_zero:
            LOGGER.warning(
                "rig->box offset has a lateral component (dy=%.4f m) that the "
                "gate does not model; refusing the offset.",
                offset[1],
            )

        # Kept regardless of the env flag above, for the route-corridor filter.
        # That filter is not part of the trained model -- it is a geometric
        # check against the scorer's own criterion, and the scorer measures the
        # footprint centre. Matching it there is a correction, not a
        # train/inference change, so it does not need the flag's caution.
        self._route_center_dx = (
            offset[0]
            if (rotation_is_identity and lateral_is_zero and abs(offset[0]) <= 5.0)
            else 0.0
        )

        applied_dx = 0.0
        if _env_flag(
            "DRIVESUPRIM_EGO_CENTER_OFFSET",
            default=False,
        ):
            if not (rotation_is_identity and lateral_is_zero):
                pass  # already warned; keep the pose-centred placement
            elif abs(offset[0]) <= 5.0:
                config.feasibility_ego_center_dx_m = offset[0]
                applied_dx = offset[0]
            else:
                LOGGER.warning(
                    "ignoring implausible rig->box forward offset %.3f m",
                    offset[0],
                )

        print(
            "[DriveSuprim] ego footprint from API:"
            f" length={length:.3f}m width={width:.3f}m"
            f" (was {previous[0]}x{previous[1]})"
            f" | rig->box offset=({offset[0]:.3f},"
            f" {offset[1]:.3f}, {offset[2]:.3f})"
            f" | applied dx={applied_dx:.3f}",
            flush=True,
        )

    def start_session(
        self,
        request,
        context,
    ):
        available = []
        mapped = {}

        camera_specs = {}
        rectifiers = {}
        lidar2img_by_camera = {}

        for camera in (
            request
            .rollout_spec
            .vehicle
            .available_cameras
        ):
            source_id = (
                camera
                .logical_id
                .upper()
            )

            available.append(
                camera.logical_id
            )

            mapped_id = (
                _CAMERA_ALIASES.get(
                    source_id
                )
            )

            if mapped_id is None:
                continue

            mapped[
                camera.logical_id
            ] = mapped_id

            saved = (
                sensorsim_pb2
                .AvailableCamerasReturn
                .AvailableCamera()
            )
            saved.CopyFrom(camera)

            camera_specs[
                mapped_id
            ] = saved

            model_type = (
                saved
                .intrinsics
                .WhichOneof(
                    "camera_param"
                )
            )

            if model_type != "ftheta_param":
                raise RuntimeError(
                    f"{source_id}: expected "
                    f"ftheta_param, got "
                    f"{model_type}"
                )

            # The camera proto and decoded NRE frame can disagree on width
            # (for example 1900 vs 1920). Build the map lazily from the first
            # actual frame instead of trusting session metadata.
            rectifiers[
                mapped_id
            ] = None

            lidar2img = (
                _build_virtual_lidar2img(
                    saved
                )
            )

            lidar2img_by_camera[
                mapped_id
            ] = lidar2img

            u, v, depth = (
                _projection_probe(
                    lidar2img
                )
            )

            summary = (
                _camera_model_summary(
                    saved
                )
            )

            pose_summary = (
                _pose_summary(
                    saved.rig_to_camera
                )
            )

            print(
                "[DriveSuprim] camera calibration: "
                f"{source_id} -> {mapped_id} | "
                f"{summary} | "
                f"{pose_summary}",
                flush=True,
            )

            print(
                "[DriveSuprim] virtual pinhole: "
                f"{mapped_id} "
                f"K=(fx={_VIRTUAL_FX:.3f},"
                f" fy={_VIRTUAL_FY:.3f},"
                f" cx={_VIRTUAL_CX:.3f},"
                f" cy={_VIRTUAL_CY:.3f}) "
                f"probe10m=(u={u:.2f},"
                f" v={v:.2f},"
                f" depth={depth:.2f})",
                flush=True,
            )

        self._apply_ego_footprint(
            request.rollout_spec.vehicle
        )

        with self._lock:
            self._sessions[
                request.session_uuid
            ] = SessionState(
                session_uuid=request.session_uuid,
                camera_specs=(
                    camera_specs
                ),
                rectifiers=(
                    rectifiers
                ),
                lidar2img_by_camera=(
                    lidar2img_by_camera
                ),
            )

        LOGGER.info(
            "started session %s; "
            "available cameras=%s, "
            "DriveSuprim mapping=%s",
            request.session_uuid,
            available,
            mapped,
        )

        required = set(
            _BEV_CAMERA_ORDER
        )

        if not required.issubset(
            camera_specs
        ):
            raise RuntimeError(
                "session lacks required "
                "DriveSuprim cameras: "
                f"have={sorted(camera_specs)}"
            )

        return (
            common_pb2
            .SessionRequestStatus()
        )

    def close_session(
        self,
        request,
        context,
    ):
        with self._lock:
            self._sessions.pop(
                request.session_uuid,
                None,
            )

        return common_pb2.Empty()

    def submit_image_observation(
        self,
        request,
        context,
    ):
        session = self._session(
            request.session_uuid,
            context,
        )

        source_camera_id = (
            request
            .camera_image
            .logical_id
            .upper()
        )

        camera_id = (
            _CAMERA_ALIASES.get(
                source_camera_id
            )
        )

        if camera_id is None:
            return common_pb2.Empty()

        image = np.asarray(
            Image.open(
                BytesIO(
                    request
                    .camera_image
                    .image_bytes
                )
            ).convert("RGB")
        )

        rectifier = self._rectifier_for_image(
            session,
            camera_id,
            source_resolution_hw=(
                int(image.shape[0]),
                int(image.shape[1]),
            ),
        )

        image = rectifier.rectify(
            image
        )

        image = np.asarray(
            image,
            dtype=np.uint8,
        )

        expected_shape = (
            _VIRTUAL_PINHOLE_HEIGHT,
            _VIRTUAL_PINHOLE_WIDTH,
            3,
        )

        if image.shape != expected_shape:
            raise RuntimeError(
                f"{camera_id}: rectified "
                f"image shape={image.shape}, "
                f"expected={expected_shape}"
            )

        with self._lock:
            session.current_images[
                camera_id
            ] = image

            session.current_frame_times[
                camera_id
            ] = int(
                request
                .camera_image
                .frame_end_us
            )

            self._commit_camera_frame_locked(
                session
            )

        return common_pb2.Empty()

    def _rectifier_for_image(
        self,
        session: SessionState,
        camera_id: str,
        *,
        source_resolution_hw: tuple[int, int],
    ):
        source_resolution_hw = tuple(
            int(value)
            for value in source_resolution_hw
        )

        with self._lock:
            rectifier = session.rectifiers.get(
                camera_id
            )
            cached_resolution = (
                session
                .rectifier_source_resolutions
                .get(camera_id)
            )

            if (
                rectifier is not None
                and cached_resolution
                == source_resolution_hw
            ):
                return rectifier

            camera_proto = session.camera_specs.get(
                camera_id
            )
            if camera_proto is None:
                raise RuntimeError(
                    "missing camera calibration for "
                    f"{camera_id}"
                )

            rectifier = (
                build_ftheta_rectifier_for_resolution(
                    camera_proto=camera_proto,
                    target_cfg=(
                        _rectification_target_config()
                    ),
                    source_resolution_hw=(
                        source_resolution_hw
                    ),
                )
            )
            session.rectifiers[camera_id] = (
                rectifier
            )
            (
                session
                .rectifier_source_resolutions[
                    camera_id
                ]
            ) = source_resolution_hw

        print(
            f"[DriveSuprim] {_PINHOLE_PROFILE} rectifier ready: "
            f"{camera_id} source="
            f"{source_resolution_hw[1]}x"
            f"{source_resolution_hw[0]} target="
            f"{_VIRTUAL_PINHOLE_WIDTH}x"
            f"{_VIRTUAL_PINHOLE_HEIGHT}",
            flush=True,
        )
        return rectifier

    def submit_egomotion_observation(
        self,
        request,
        context,
    ):
        session = self._session(
            request.session_uuid,
            context,
        )

        with self._lock:
            if request.trajectory.poses:
                session.poses.extend(
                    request.trajectory.poses
                )

                session.poses.sort(
                    key=lambda pose: (
                        pose.timestamp_us
                    )
                )

                session.poses = (
                    session.poses[-32:]
                )

                session.latest_pose = (
                    session.poses[-1]
                )

            for index, state in enumerate(
                request.dynamic_states
            ):
                timestamp = (
                    int(
                        request
                        .trajectory
                        .poses[index]
                        .timestamp_us
                    )
                    if index
                    < len(
                        request
                        .trajectory
                        .poses
                    )
                    else 0
                )

                session.dynamic_states.append(
                    (
                        timestamp,
                        state,
                    )
                )

            session.dynamic_states = (
                session.dynamic_states[-32:]
            )

        return common_pb2.Empty()

    def submit_route(
        self,
        request,
        context,
    ):
        session = self._session(
            request.session_uuid,
            context,
        )

        route = egodriver_pb2.Route()
        route.CopyFrom(
            request.route
        )

        with self._lock:
            session.latest_route = route
            if _route_cache_enabled():
                session.pending_routes.append(
                    (
                        int(route.timestamp_us),
                        np.asarray(
                            [
                                (point.x, point.y, point.z)
                                for point in route.waypoints
                            ],
                            dtype=np.float64,
                        ).reshape(-1, 3),
                    )
                )
                _resolve_pending_routes(session)

        return common_pb2.Empty()

    def submit_recording_ground_truth(
        self,
        request,
        context,
    ):
        return common_pb2.Empty()

    def drive(
        self,
        request,
        context,
    ):
        session = self._session(
            request.session_uuid,
            context,
        )

        now = int(
            request.time_now_us
        )

        with self._lock:
            if (
                session.latest_route is not None
                and session.dynamic_states
            ):
                session.route_command = (
                    _route_command(
                        session.latest_route,
                        session.dynamic_states[-1][1],
                    )
                )

        self._maybe_infer(
            session,
            now,
        )

        with self._lock:
            fallback_speed_mps = max(
                2.0,
                _estimate_speed_mps(
                    session
                ),
            )

            trajectory = (
                _trajectory_from_plan(
                    session.cached_plan,
                    session.latest_pose,
                    now,
                    fallback_speed_mps=(
                        fallback_speed_mps
                    ),
                )
            )

            command = (
                session.route_command.copy()
            )

            dynamic_state = (
                session.dynamic_states[-1][1]
                if session.dynamic_states
                else None
            )

            drivesuprim_debug = (
                _prediction_debug_payload(session)
            )

        command_index = int(
            np.argmax(command)
        )

        command_name = (
            "LEFT",
            "STRAIGHT",
            "RIGHT",
            "UNKNOWN",
        )[command_index]

        speed_mps = None
        acceleration_longitudinal_mps2 = None
        acceleration_lateral_mps2 = None

        if dynamic_state is not None:
            speed_mps = float(
                np.hypot(
                    dynamic_state.linear_velocity.x,
                    dynamic_state.linear_velocity.y,
                )
            )

            acceleration_longitudinal_mps2 = (
                float(
                    dynamic_state
                    .linear_acceleration
                    .x
                )
            )

            acceleration_lateral_mps2 = (
                float(
                    dynamic_state
                    .linear_acceleration
                    .y
                )
            )

        debug_info = (
            egodriver_pb2
            .DriveResponse
            .DebugInfo(
                unstructured_debug_info=(
                    pickle.dumps(
                        {
                            "command": command_index,
                            "command_name": command_name,
                            "command_one_hot": (
                                command.tolist()
                            ),
                            "speed_mps": speed_mps,
                            "acceleration_longitudinal_mps2": (
                                acceleration_longitudinal_mps2
                            ),
                            "acceleration_lateral_mps2": (
                                acceleration_lateral_mps2
                            ),
                            "drivesuprim_debug": (
                                drivesuprim_debug
                            ),
                        }
                    )
                )
            )
        )

        return egodriver_pb2.DriveResponse(
            trajectory=trajectory,
            debug_info=debug_info,
        )

    def get_version(
        self,
        request,
        context,
    ):
        require_policy = _env_flag(
            "DRIVESUPRIM_REQUIRE_POLICY_FOR_VERSION",
            default=False,
        )

        if require_policy:
            error = (
                self
                ._policy_handle
                .error()
            )

            if error is not None:
                context.abort(
                    grpc.StatusCode.UNAVAILABLE,
                    f"policy load failed: {error}",
                )

            if (
                self
                ._policy_handle
                .get()
                is None
            ):
                context.abort(
                    grpc.StatusCode.UNAVAILABLE,
                    "policy is still loading",
                )

        return common_pb2.VersionId(
            version_id=(
                "drivesuprim-e2e-driver"
            ),
            git_hash="local",
            grpc_api_version=(
                API_VERSION_MESSAGE
            ),
        )

    def shut_down(
        self,
        request,
        context,
    ):
        if self._server is not None:
            threading.Thread(
                target=self._stop_server,
                daemon=True,
            ).start()

        return common_pb2.Empty()

    def _stop_server(self) -> None:
        time.sleep(0.05)

        if self._server is not None:
            self._server.stop(
                grace=0.0
            )

    def _commit_camera_frame_locked(
        self,
        session: SessionState,
    ) -> None:
        required = {
            "CAM_L0",
            "CAM_F0",
            "CAM_R0",
        }

        if not required.issubset(
            session.current_images
        ):
            return

        times = [
            session.current_frame_times[name]
            for name in required
        ]

        tolerance = int(
            os.environ.get(
                "DRIVESUPRIM_CAMERA_SYNC_TOLERANCE_US",
                "50000",
            )
        )

        if (
            max(times) - min(times)
            > tolerance
        ):
            return

        frame_timestamp_us = int(
            round(
                float(
                    np.median(times)
                )
            )
        )

        session.camera_history.append(
            CameraFrame(
                timestamp_us=(
                    frame_timestamp_us
                ),
                images={
                    name: (
                        session
                        .current_images[name]
                        .copy()
                    )
                    for name in required
                },
            )
        )

        # Keep enough dense 10-Hz camera history to later sample
        # NAVSIM-style t-1.0s, t-0.5s, t.
        session.camera_history = (
            session.camera_history[-32:]
        )

        session.current_images.clear()
        session.current_frame_times.clear()

    def _maybe_infer(
        self,
        session: SessionState,
        now: int,
    ) -> None:
        if _env_flag(
            "DRIVESUPRIM_DISABLE_INFERENCE",
            default=False,
        ):
            return

        policy = (
            self
            ._policy_handle
            .get()
        )

        if policy is None:
            return

        with self._lock:
            # BEVFormer stage3 requires exactly:
            # current + two historical camera frames.
            if (
                len(
                    session.camera_history
                )
                < 3
                or session.latest_pose is None
                or not session.dynamic_states
                or len(
                    session.lidar2img_by_camera
                )
                < 3
            ):
                return

            if (
                session.cached_plan
                and (
                    now
                    - session
                    .cached_plan
                    .created_time_us
                    < self
                    ._inference_interval_us
                )
            ):
                return

            status = StatusFrame(
                timestamp_us=int(session.dynamic_states[-1][0]),
                values=_ego_status(session),
            )

            session.status_history.append(
                status
            )

            if _use_navsim_2hz_history():
                session.status_history = session.status_history[-32:]
                frames = select_history_by_offsets(
                    session.camera_history,
                    anchor_timestamp_us=session.camera_history[-1].timestamp_us,
                    offsets_us=CAMERA_HISTORY_OFFSETS_US,
                    tolerance_us=HISTORY_TOLERANCE_US,
                )
                selected_statuses = select_history_by_offsets(
                    session.status_history,
                    anchor_timestamp_us=status.timestamp_us,
                    offsets_us=STATUS_HISTORY_OFFSETS_US,
                    tolerance_us=HISTORY_TOLERANCE_US,
                )
                if frames is None or selected_statuses is None:
                    return
            else:
                session.status_history = session.status_history[-2:]
                frames = list(session.camera_history[-3:])
                selected_statuses = list(session.status_history)

            cameras = [
                frame.images
                for frame in frames
            ]

            statuses = [item.values for item in selected_statuses]

            try:
                bev_ego_pose = (
                    _build_bev_ego_pose(
                        frames,
                        session.poses,
                    )
                )
            except Exception:
                LOGGER.exception(
                    "failed to build "
                    "BEV ego-pose history"
                )
                return

            lidar2img_by_camera = {
                key: value.copy()
                for key, value in (
                    session
                    .lidar2img_by_camera
                    .items()
                )
            }

            anchor = (
                session.latest_pose
            )
            route_waypoints = _route_waypoints_for_policy(session.latest_route)

            # The model keeps receiving exactly what the runtime sends -- the
            # 42-80 m window it was trained on. The cache is handed over
            # separately and used only to filter candidates, so nothing about
            # the network's input distribution changes.
            cached_route = None
            if _route_cache_enabled() and anchor is not None:
                _resolve_pending_routes(session)
                ego_x = float(anchor.pose.vec.x)
                ego_y = float(anchor.pose.vec.y)
                ego_yaw = _yaw(anchor.pose.quat)
                if session.route_cache.has_near_field(ego_x, ego_y, ego_yaw):
                    cached_route = session.route_cache.query_rig(
                        ego_x, ego_y, ego_yaw
                    )

        with self._inference_lock:
            try:
                prediction = (
                    policy.predict(
                        cameras,
                        statuses,
                        route_waypoints=route_waypoints,
                        cached_route=cached_route,
                        route_corridor_m=_route_corridor_m(),
                        route_center_dx_m=self._route_center_dx,
                        lidar2img_by_camera=(
                            lidar2img_by_camera
                        ),
                        bev_ego_pose=(
                            bev_ego_pose
                        ),
                        debug_context=session.session_uuid,
                    )
                )

            except Exception:
                LOGGER.exception(
                    "DriveSuprim inference failed"
                )
                return

        plan = _make_plan(
            now,
            anchor,
            prediction.poses,
            policy.output_frequency_hz,
        )

        with self._lock:
            session.cached_plan = plan
            session.latest_prediction = prediction

    def _session(
        self,
        uuid: str,
        context,
    ) -> SessionState:
        with self._lock:
            session = self._sessions.get(
                uuid
            )

        if session is None:
            context.abort(
                grpc.StatusCode.NOT_FOUND,
                f"unknown session {uuid}",
            )

            raise AssertionError(
                "unreachable"
            )

        return session


def _route_command(
    route,
    dynamic_state: common_pb2.DynamicState,
) -> np.ndarray:
    """Classify remaining yaw error between ego motion and route."""

    command = np.zeros(
        4,
        dtype=np.float32,
    )

    valid_points = np.asarray(
        [
            (
                float(point.x),
                float(point.y),
            )
            for point in route.waypoints
            if (
                np.isfinite(point.x)
                and np.isfinite(point.y)
            )
        ],
        dtype=np.float64,
    ).reshape(-1, 2)

    if len(valid_points) < 1:
        command[3] = 1.0
        return command

    points = np.vstack(
        (
            np.zeros(
                (1, 2),
                dtype=np.float64,
            ),
            valid_points,
        )
    )

    segment_lengths = np.linalg.norm(
        np.diff(
            points,
            axis=0,
        ),
        axis=1,
    )

    keep = np.concatenate(
        (
            [True],
            segment_lengths > 1e-3,
        )
    )

    points = points[keep]

    if len(points) < 2:
        command[3] = 1.0
        return command

    cumulative_m = np.concatenate(
        (
            [0.0],
            np.cumsum(
                np.linalg.norm(
                    np.diff(
                        points,
                        axis=0,
                    ),
                    axis=1,
                )
            ),
        )
    )

    lookahead_m = float(
        os.environ.get(
            "DRIVESUPRIM_ROUTE_LOOKAHEAD_M",
            "20.0",
        )
    )

    center = int(
        np.argmin(
            np.abs(
                cumulative_m
                - lookahead_m
            )
        )
    )

    before = max(
        0,
        center - 1,
    )

    after = min(
        len(points) - 1,
        center + 1,
    )

    if before == after:
        command[3] = 1.0
        return command

    route_vector = (
        points[after]
        - points[before]
    )

    route_yaw_rad = math.atan2(
        float(route_vector[1]),
        float(route_vector[0]),
    )

    velocity_x = float(
        dynamic_state.linear_velocity.x
    )

    velocity_y = float(
        dynamic_state.linear_velocity.y
    )

    speed_mps = math.hypot(
        velocity_x,
        velocity_y,
    )

    min_motion_speed_mps = float(
        os.environ.get(
            "DRIVESUPRIM_ROUTE_MIN_MOTION_SPEED_MPS",
            "1.0",
        )
    )

    motion_yaw_rad = (
        math.atan2(
            velocity_y,
            velocity_x,
        )
        if speed_mps
        >= min_motion_speed_mps
        else 0.0
    )

    preview_s = float(
        os.environ.get(
            "DRIVESUPRIM_ROUTE_YAW_PREVIEW_S",
            "0.3",
        )
    )

    predicted_motion_yaw_rad = (
        motion_yaw_rad
        + float(
            dynamic_state.angular_velocity.z
        )
        * preview_s
    )

    yaw_error_rad = math.atan2(
        math.sin(
            route_yaw_rad
            - predicted_motion_yaw_rad
        ),
        math.cos(
            route_yaw_rad
            - predicted_motion_yaw_rad
        ),
    )

    threshold_rad = math.radians(
        float(
            os.environ.get(
                "DRIVESUPRIM_ROUTE_TURN_THRESHOLD_DEG",
                "7.5",
            )
        )
    )

    if yaw_error_rad > threshold_rad:
        command[0] = 1.0

    elif yaw_error_rad < -threshold_rad:
        command[2] = 1.0

    else:
        command[1] = 1.0

    return command


def _ego_status(
    session: SessionState,
) -> np.ndarray:
    if not session.dynamic_states:
        raise RuntimeError(
            "AlpaSim has not supplied "
            "a current DynamicState"
        )

    state = (
        session.dynamic_states[-1][1]
    )

    velocity = np.array(
        [
            state.linear_velocity.x,
            state.linear_velocity.y,
        ],
        dtype=np.float32,
    )

    acceleration = np.array(
        [
            state.linear_acceleration.x,
            state.linear_acceleration.y,
        ],
        dtype=np.float32,
    )

    return np.concatenate(
        (
            session.route_command,
            velocity,
            acceleration,
        )
    )


def _estimate_speed_mps(
    session: SessionState,
) -> float:
    if session.dynamic_states:
        state = (
            session.dynamic_states[-1][1]
        )

        return float(
            np.hypot(
                state.linear_velocity.x,
                state.linear_velocity.y,
            )
        )

    if len(session.poses) >= 2:
        previous = session.poses[-2]
        current = session.poses[-1]

        dt_s = (
            int(current.timestamp_us)
            - int(previous.timestamp_us)
        ) / 1_000_000.0

        if dt_s > 1e-6:
            distance_m = np.hypot(
                current.pose.vec.x
                - previous.pose.vec.x,
                current.pose.vec.y
                - previous.pose.vec.y,
            )

            return float(
                distance_m / dt_s
            )

    return 5.0


def _pack_submask(sub, vocab):
    """Pack a per-rule feasibility mask, or None when the rule did not run."""
    if sub is None or vocab is None:
        return None
    sub = np.asarray(sub, dtype=np.bool_).reshape(-1)
    if sub.size != vocab.shape[0]:
        return None
    return np.packbits(sub, bitorder="little")


def _prediction_debug_payload(
    session: SessionState,
) -> dict[str, Any] | None:
    """Serialize compact DriveSuprim BEV diagnostics for clipgt."""
    prediction = session.latest_prediction
    if prediction is None:
        return None

    vocab = None
    if prediction.candidate_vocab is not None:
        candidate_vocab = np.asarray(prediction.candidate_vocab)
        if (
            candidate_vocab.ndim == 3
            and candidate_vocab.shape[0] > 0
            and candidate_vocab.shape[2] >= 2
        ):
            vocab = candidate_vocab

    mask = None
    if vocab is not None:
        mask = prediction.feasibility_mask
        if (
            mask is None
            or np.asarray(mask).size != vocab.shape[0]
        ):
            mask = np.ones(
                vocab.shape[0],
                dtype=np.bool_,
            )
        mask = np.asarray(mask, dtype=np.bool_).reshape(-1)

    payload: dict[str, Any] = {
        "version": 1,
        "candidate_count": (
            0 if vocab is None else int(vocab.shape[0])
        ),
        "feasibility_mask_packed": (
            None
            if mask is None
            else np.packbits(mask, bitorder="little")
        ),
        # Per-rule masks, so the overlay can colour a drivable-area rejection
        # differently from a collision rejection. Same packing as above.
        "feasibility_drivable_packed": _pack_submask(
            prediction.feasibility_drivable_mask, vocab
        ),
        "feasibility_collision_packed": _pack_submask(
            prediction.feasibility_collision_mask, vocab
        ),
        "selected_index": prediction.selected_index,
        "coarse_path": (
            None
            if prediction.coarse_path is None
            else np.asarray(
                prediction.coarse_path,
                dtype=np.float16,
            )
        ),
        "coarse_candidates": (
            None
            if prediction.coarse_candidates is None
            else np.asarray(
                prediction.coarse_candidates,
                dtype=np.float16,
            )
        ),
        "final_path": np.asarray(
            prediction.poses,
            dtype=np.float16,
        ),
        "point_cloud_range": prediction.point_cloud_range,
        "drivable_probability": prediction.drivable_probability,
        "detected_agents": (
            None
            if prediction.detected_agents is None
            else np.asarray(
                prediction.detected_agents,
                dtype=np.float16,
            )
        ),
        "detection_classes": prediction.detection_classes,
        # Ranking-score decomposition: per-candidate value of each term, the
        # index the model drove, and per-term influence/leave-one-out stats.
        "rank_terms": (
            None
            if prediction.rank_terms is None
            else {
                name: np.asarray(value, dtype=np.float16)
                for name, value in prediction.rank_terms.items()
            }
        ),
        "rank_total": (
            None
            if prediction.rank_total is None
            else np.asarray(prediction.rank_total, dtype=np.float16)
        ),
        "rank_selected": prediction.rank_selected,
        "rank_stats": prediction.rank_stats,
        # Route-corridor gate. The cached route is sent every frame rather than
        # once, because unlike the candidate vocabulary it is not fixed -- it
        # is rebuilt in the current rig frame each step, and the point of the
        # overlay is to show what the filter was actually looking at.
        "cached_route": (
            None
            if prediction.cached_route is None
            else np.asarray(prediction.cached_route, dtype=np.float16)
        ),
        "route_mask": (
            None
            if prediction.feasibility_route_mask is None
            else np.asarray(prediction.feasibility_route_mask, dtype=np.bool_)
        ),
        "route_rejected_paths": (
            None
            if prediction.route_rejected_paths is None
            else np.asarray(prediction.route_rejected_paths, dtype=np.float16)
        ),
        "route_gate_poses": prediction.route_gate_poses,
        "route_reselected": bool(prediction.route_reselected),
    }

    if vocab is not None and not session.debug_vocab_sent:
        payload["candidate_vocab"] = vocab[:, :, :3].astype(
            np.float16,
            copy=False,
        )
        session.debug_vocab_sent = True

    return payload


def _yaw(
    quat,
) -> float:
    return math.atan2(
        2.0
        * (
            quat.w * quat.z
            + quat.x * quat.y
        ),
        1.0
        - 2.0
        * (
            quat.y * quat.y
            + quat.z * quat.z
        ),
    )


def _quat(
    yaw: float,
):
    return common_pb2.Quat(
        w=math.cos(
            yaw / 2
        ),
        z=math.sin(
            yaw / 2
        ),
    )


def _make_plan(
    now: int,
    anchor,
    poses: np.ndarray,
    frequency: float,
) -> CachedPlan:
    relative = np.asarray(
        poses,
        dtype=np.float64,
    )

    anchor_yaw = _yaw(
        anchor.pose.quat
    )

    c = math.cos(
        anchor_yaw
    )
    s = math.sin(
        anchor_yaw
    )

    rotation = np.array(
        [
            [c, -s],
            [s, c],
        ]
    )

    origin = np.array(
        [
            anchor.pose.vec.x,
            anchor.pose.vec.y,
        ]
    )

    positions = (
        relative[:, :2]
        @ rotation.T
        + origin
    )

    positions = np.vstack(
        (
            origin,
            positions,
        )
    )

    yaws = np.concatenate(
        (
            [anchor_yaw],
            relative[:, 2]
            + anchor_yaw,
        )
    )

    times = (
        np.arange(
            len(positions),
            dtype=np.float64,
        )
        / frequency
    )

    return CachedPlan(
        now,
        times,
        positions,
        yaws,
    )


def _trajectory_from_plan(
    plan,
    pose,
    now,
    frequency: float = 10.0,
    fallback_speed_mps: float = 5.0,
):
    if pose is None:
        pose = common_pb2.PoseAtTime(
            timestamp_us=now,
            pose=common_pb2.Pose(
                vec=common_pb2.Vec3(),
                quat=common_pb2.Quat(
                    w=1.0
                ),
            ),
        )

    trajectory = (
        common_pb2.Trajectory()
    )

    if plan is None:
        yaw = _yaw(
            pose.pose.quat
        )

        for index in range(41):
            dt = index / frequency

            trajectory.poses.append(
                common_pb2.PoseAtTime(
                    timestamp_us=(
                        now
                        + round(
                            dt
                            * 1_000_000
                        )
                    ),
                    pose=common_pb2.Pose(
                        vec=common_pb2.Vec3(
                            x=(
                                pose.pose.vec.x
                                + fallback_speed_mps
                                * dt
                                * math.cos(yaw)
                            ),
                            y=(
                                pose.pose.vec.y
                                + fallback_speed_mps
                                * dt
                                * math.sin(yaw)
                            ),
                            z=pose.pose.vec.z,
                        ),
                        quat=pose.pose.quat,
                    ),
                )
            )

        return trajectory

    elapsed = max(
        0.0,
        (
            now
            - plan.created_time_us
        )
        / 1_000_000.0,
    )

    sample_times = (
        np.arange(
            math.ceil(
                elapsed
                * frequency
            ),
            len(plan.times_s),
        )
        / frequency
    )

    if len(sample_times) < 2:
        return _trajectory_from_plan(
            None,
            pose,
            now,
            frequency,
            fallback_speed_mps=(
                fallback_speed_mps
            ),
        )

    xs = np.interp(
        sample_times,
        plan.times_s,
        plan.positions_xy[:, 0],
    )

    ys = np.interp(
        sample_times,
        plan.times_s,
        plan.positions_xy[:, 1],
    )

    yaws = np.interp(
        sample_times,
        plan.times_s,
        np.unwrap(
            plan.yaws
        ),
    )

    for (
        time_s,
        x,
        y,
        yaw,
    ) in zip(
        sample_times,
        xs,
        ys,
        yaws,
        strict=True,
    ):
        trajectory.poses.append(
            common_pb2.PoseAtTime(
                timestamp_us=(
                    plan.created_time_us
                    + round(
                        time_s
                        * 1_000_000
                    )
                ),
                pose=common_pb2.Pose(
                    vec=common_pb2.Vec3(
                        x=x,
                        y=y,
                        z=pose.pose.vec.z,
                    ),
                    quat=_quat(yaw),
                ),
            )
        )

    return trajectory


def main() -> None:
    torch.set_num_threads(
        max(
            1,
            int(
                os.environ.get(
                    "TORCH_NUM_THREADS",
                    "1",
                )
            ),
        )
    )

    for path in (
        _RUNTIME_DIRS.values()
    ):
        os.makedirs(
            path,
            exist_ok=True,
        )

    logging.basicConfig(
        level=os.environ.get(
            "ALPASIM_DRIVER_LOG_LEVEL",
            "INFO",
        ),
        format=(
            "%(asctime)s "
            "%(levelname)s "
            "%(name)s: "
            "%(message)s"
        ),
    )

    asset_dir = os.environ.get(
        "DRIVESUPRIM_ASSET_DIR",
        "/app/assets/drivesuprim",
    )

    # Support both names, but prefer the Dockerfile/run-script name.
    backbone = os.environ.get(
        "DRIVESUPRIM_BACKBONE_TYPE",
        os.environ.get(
            "DRIVESUPRIM_BACKBONE",
            "auto",
        ),
    )

    checkpoint_default = {
        "resnet34": "drivesuprim_r34.ckpt",
        "vit": "drivesuprim_vit.ckpt",
        "vov": "drivesuprim_vov.ckpt",
        "bevformer_m": (
            "drivesuprim_v2_"
            "convnextv2_stage3_epoch04.ckpt"
        ),
        "bevformer_r50": (
            "drivesuprim_v2_"
            "resnet50_stage3_epoch03.ckpt"
        ),
    }.get(
        backbone,
        "drivesuprim_r34.ckpt",
    )

    vocab_default = (
        "test_4096_kmeans.npy"
        if backbone in {"bevformer_m", "bevformer_r50"}
        else "test_8192_kmeans.npy"
    )

    config_default = {
        "bevformer_m": f"{asset_dir}/cnx_stage3_config.json",
        "bevformer_r50": f"{asset_dir}/r50_stage3_config.json",
    }.get(backbone, "")

    handle = PolicyHandle(
        checkpoint_path=os.environ.get(
            "DRIVESUPRIM_CHECKPOINT_PATH",
            f"{asset_dir}/{checkpoint_default}",
        ),
        vocab_path=os.environ.get(
            "DRIVESUPRIM_VOCAB_PATH",
            f"{asset_dir}/{vocab_default}",
        ),
        backbone_type=backbone,
        device=os.environ.get(
            "DRIVESUPRIM_DEVICE",
            "cuda",
        ),
        vit_checkpoint_path=os.environ.get(
            "DRIVESUPRIM_VIT_CHECKPOINT_PATH",
            f"{asset_dir}/da_vitl16.pth",
        ),
        vov_checkpoint_path=os.environ.get(
            "DRIVESUPRIM_VOV_CHECKPOINT_PATH",
            f"{asset_dir}/dd3d_det_final.pth",
        ),
        config_path=os.environ.get(
            "DRIVESUPRIM_CONFIG_PATH",
            config_default,
        ),
    )

    service = DriveSuprimChallengeDriver(
        handle,
        int(
            os.environ.get(
                "DRIVESUPRIM_INFERENCE_INTERVAL_US",
                "500000",
            )
        ),
    )

    workers = int(
        os.environ.get(
            "ALPASIM_DRIVER_GRPC_WORKERS",
            "4",
        )
    )

    server = grpc.server(
        futures.ThreadPoolExecutor(
            max_workers=workers
        )
    )

    egodriver_pb2_grpc.add_EgodriverServiceServicer_to_server(
        service,
        server,
    )

    service.attach_server(
        server
    )

    host = os.environ.get(
        "ALPASIM_DRIVER_HOST",
        "0.0.0.0",
    )

    port = int(
        os.environ.get(
            "ALPASIM_DRIVER_PORT",
            "6789",
        )
    )

    if (
        server.add_insecure_port(
            f"{host}:{port}"
        )
        == 0
    ):
        raise RuntimeError(
            f"failed to bind {host}:{port}"
        )

    signal.signal(
        signal.SIGTERM,
        lambda *_: server.stop(
            grace=0.0
        ),
    )

    signal.signal(
        signal.SIGINT,
        lambda *_: server.stop(
            grace=0.0
        ),
    )

    server.start()

    LOGGER.info(
        "DriveSuprim driver listening on %s:%d",
        host,
        port,
    )

    LOGGER.info(
        "DriveSuprim runtime: "
        "backbone=%s "
        "disable_inference=%s "
        "checkpoint=%s "
        "vocab=%s "
        "config=%s",
        backbone,
        _env_flag(
            "DRIVESUPRIM_DISABLE_INFERENCE",
            False,
        ),
        os.environ.get(
            "DRIVESUPRIM_CHECKPOINT_PATH",
            f"{asset_dir}/{checkpoint_default}",
        ),
        os.environ.get(
            "DRIVESUPRIM_VOCAB_PATH",
            f"{asset_dir}/{vocab_default}",
        ),
        os.environ.get(
            "DRIVESUPRIM_CONFIG_PATH",
            config_default,
        ),
    )

    handle.start()

    server.wait_for_termination()


if __name__ == "__main__":
    main()
