"""Minimal stand-ins for the pieces of ``alpasim_utils.geometry`` the NuRec
controller touches.

The real package is a Rust extension (``utils_rs``); only ``Pose`` and
``Trajectory`` reach the controller's maths, and only a handful of their methods.
Reproducing those exactly here lets the upstream controller run as a reference
implementation to check the batched port against, without building Rust.

Semantics follow ``utils_rs`` (pose.rs / trajectory.rs):

* quaternions are stored and returned in scipy order ``[x, y, z, w]``
* ``yaw`` is ``atan2(2(wz + xy), 1 - 2(yy + zz))``
* the time range is half-open, ``[first, last + 1)`` in microseconds, and
  interpolating outside it raises rather than extrapolating
* position interpolates linearly, orientation by shortest-arc quaternion slerp
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np


def _as_float(values: Sequence[float], size: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != size:
        raise ValueError(f"expected {size} values, got {array.size}")
    return array


class Pose:
    """Position plus orientation, quaternion in scipy order ``[x, y, z, w]``."""

    __slots__ = ("_position", "_quaternion")

    def __init__(self, position: Sequence[float], quaternion: Sequence[float]):
        self._position = _as_float(position, 3)
        self._quaternion = _as_float(quaternion, 4)

    @property
    def vec3(self) -> np.ndarray:
        return self._position

    @property
    def quat(self) -> np.ndarray:
        return self._quaternion

    def yaw(self) -> float:
        x, y, z, w = self._quaternion
        return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))

    @classmethod
    def from_xy_yaw(cls, x: float, y: float, yaw: float) -> "Pose":
        return cls([x, y, 0.0], [0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)])

    def __repr__(self) -> str:
        return f"Pose(xy=({self._position[0]:.3f}, {self._position[1]:.3f}), yaw={self.yaw():.4f})"


def _slerp(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """Shortest-arc quaternion slerp, matching glam's sign handling."""
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:  # nearly parallel: lerp and renormalise
        result = q0 + alpha * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    return (np.sin((1.0 - alpha) * theta) * q0 + np.sin(alpha * theta) * q1) / sin_theta


class Trajectory:
    """Timestamped poses with the interpolation rules ``utils_rs`` implements."""

    __slots__ = ("_timestamps", "_poses")

    def __init__(self, entries: Sequence[Tuple[int, Pose]]):
        if not entries:
            self._timestamps = np.empty(0, dtype=np.uint64)
            self._poses: List[Pose] = []
            return
        self._timestamps = np.asarray([int(t) for t, _ in entries], dtype=np.uint64)
        self._poses = [pose for _, pose in entries]

    @classmethod
    def from_poses(cls, timestamps_us: Sequence[int], poses: Sequence[Pose]) -> "Trajectory":
        return cls(list(zip(timestamps_us, poses)))

    def __len__(self) -> int:
        return len(self._poses)

    @property
    def time_range_us(self) -> range:
        if not self._poses:
            return range(0, 0)
        return range(int(self._timestamps[0]), int(self._timestamps[-1]) + 1)

    def get_pose(self, idx: int) -> Pose:
        return self._poses[idx]

    def interpolate(self, target_timestamps: np.ndarray) -> "Trajectory":
        if not self._poses:
            raise ValueError("Cannot interpolate on empty trajectory")
        targets = np.asarray(target_timestamps, dtype=np.uint64)
        start = int(self._timestamps[0])
        end = int(self._timestamps[-1]) + 1  # exclusive

        if len(self._poses) == 1:
            for value in targets:
                if int(value) != start:
                    raise ValueError(
                        f"Interpolation timestamp {int(value)} outside range [{start}, {end})"
                    )
            return Trajectory([(int(v), self._poses[0]) for v in targets])

        entries: List[Tuple[int, Pose]] = []
        for value in targets:
            t = int(value)
            if t < start or t >= end:
                raise ValueError(f"Interpolation timestamp {t} outside range [{start}, {end})")
            # largest i with timestamps[i] <= t, clamped to the last segment
            idx = min(int(np.searchsorted(self._timestamps, t, side="right")) - 1,
                      len(self._poses) - 2)
            t0, t1 = int(self._timestamps[idx]), int(self._timestamps[idx + 1])
            alpha = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            p0, p1 = self._poses[idx], self._poses[idx + 1]
            position = p0.vec3 + alpha * (p1.vec3 - p0.vec3)
            quaternion = _slerp(p0.quat, p1.quat, alpha)
            entries.append((t, Pose(position, quaternion)))
        return Trajectory(entries)

    def __repr__(self) -> str:
        rng = self.time_range_us
        return f"Trajectory(n_poses={len(self._poses)}, time_range_us={rng.start}..{rng.stop})"
