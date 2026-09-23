"""Remember the route the runtime has already handed over.

The challenge preset sets `route_start_offset_m: 40`, so every route message
begins about 42 m ahead of the ego and runs to 80 m; the near field is never
delivered. It is, however, delivered *earlier* -- the stretch that is 2 m ahead
now was 42 m ahead some forty metres of driving ago. Accumulating the messages
in a frame that does not move recovers it.

Three facts from the runtime make the accumulation sound, and the whole module
rests on them:

* the route arrives with every drive step (10 Hz), not once per episode;
* the underlying polyline is computed once and only ever appended to, so a
  waypoint kept from an earlier step cannot go stale;
* the ego pose arrives in the same stream as `local->rig`, giving a frame that
  outlives the step.

What cannot be recovered is the first ~42 m of an episode: nothing ever sent
it. `has_near_field` reports whether the cache has closed the gap yet, and is
the only honest activation signal -- counting metres driven would guess at it.
"""

from __future__ import annotations

import math

import numpy as np

# The runtime resamples every route to 20 points over 80 m.
RUNTIME_SPACING_M = 80.0 / 19.0


def _rig_to_local(points_xy: np.ndarray, ego_x: float, ego_y: float,
                  ego_yaw: float) -> np.ndarray:
    """Ego-relative (forward x, left y) -> the frame the ego poses live in.

    The inverse of the transform `_bev_ego_pose_history` already applies in
    driver.py; written the same way round so the two cannot disagree about
    which direction the yaw turns.
    """
    c, s = math.cos(ego_yaw), math.sin(ego_yaw)
    x, y = points_xy[:, 0], points_xy[:, 1]
    return np.stack([ego_x + c * x - s * y, ego_y + s * x + c * y], axis=1)


def _local_to_rig(points_xy: np.ndarray, ego_x: float, ego_y: float,
                  ego_yaw: float) -> np.ndarray:
    """The frame the ego poses live in -> ego-relative (forward x, left y)."""
    c, s = math.cos(ego_yaw), math.sin(ego_yaw)
    dx = points_xy[:, 0] - ego_x
    dy = points_xy[:, 1] - ego_y
    return np.stack([c * dx + s * dy, -s * dx + c * dy], axis=1)


def max_distance_to_polyline(points: np.ndarray, polyline: np.ndarray) -> float:
    """Farthest any of `points` strays from `polyline`, in metres.

    Distance is to the polyline's segments, not to its vertices: at 4.2 m
    spacing, vertex distance would read up to 2.1 m of pure sampling error as
    corridor departure.
    """
    return float(np.max(distances_to_polyline(points, polyline)))


def distances_to_polyline(points: np.ndarray, polyline: np.ndarray) -> np.ndarray:
    """Per-point distance [N] from `points` [N,2] to `polyline` [M,2]."""
    points = np.asarray(points, dtype=np.float64)
    polyline = np.asarray(polyline, dtype=np.float64)
    if len(polyline) == 1:
        return np.linalg.norm(points - polyline[0], axis=1)

    a = polyline[:-1]                      # [S,2] segment starts
    seg = polyline[1:] - a                 # [S,2] segment vectors
    seg_len_sq = np.einsum("ij,ij->i", seg, seg)
    seg_len_sq = np.where(seg_len_sq > 1e-12, seg_len_sq, 1.0)

    rel = points[:, None, :] - a[None, :, :]           # [N,S,2]
    t = np.einsum("nsi,si->ns", rel, seg) / seg_len_sq  # [N,S]
    t = np.clip(t, 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * seg[None, :, :]
    return np.min(np.linalg.norm(points[:, None, :] - closest, axis=2), axis=1)


class RouteCache:
    """The route so far, in the local frame, as one forward-ordered polyline."""

    def __init__(self, keep_behind_m: float = 60.0) -> None:
        self._points: np.ndarray | None = None   # [N,2] local frame, ordered
        self._keep_behind_m = float(keep_behind_m)
        self.messages_merged = 0

    def __len__(self) -> int:
        return 0 if self._points is None else len(self._points)

    def add(self, waypoints_rig: np.ndarray, ego_x: float, ego_y: float,
            ego_yaw: float) -> None:
        """Fold one route message, as received in the rig frame, into the cache.

        `prepare_for_policy` pads every message to 20 slots with NaN, and the
        trim leaves about half of them padding, so the NaNs are the common case
        rather than an error.
        """
        pts = np.asarray(waypoints_rig, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] < 2:
            return
        pts = pts[:, :2]
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) == 0:
            return

        local = _rig_to_local(pts, ego_x, ego_y, ego_yaw)
        if self._points is None:
            self._points = local
        else:
            # The newest message is always the authority on everything from its
            # own start outwards -- it is a correctly ordered resampling of the
            # live polyline. History is therefore only needed for what lies
            # *behind* that start, which is exactly the near field the offset
            # withheld. Splicing the two keeps the result ordered without
            # having to re-derive an arc length, and lets any error in the far
            # field be overwritten on the next step rather than accumulate.
            # What to retain is decided by direction of travel, not by distance:
            # each step advances the message's start by well under one waypoint
            # spacing, so the nearest cached point to the new start is normally
            # the one just behind it, and dropping everything up to the nearest
            # would discard the history on every single step.
            if len(local) >= 2:
                heading = local[1] - local[0]
            elif len(self._points) >= 2:
                # Near the end of a route the runtime can legitimately send
                # only the final waypoint.  It has no tangent of its own, so
                # use the cached route's tangent at the closest point when
                # deciding where to splice it.  Indexing local[1] here used to
                # abort the whole rollout for these singleton updates.
                nearest = int(np.argmin(
                    np.linalg.norm(self._points - local[0], axis=1)
                ))
                if nearest == 0:
                    heading = self._points[1] - self._points[0]
                else:
                    heading = self._points[nearest] - self._points[nearest - 1]
            else:
                # Neither side supplies a direction yet.  Keep the latest
                # point and wait for a later message with a usable segment.
                self._points = local
                self.messages_merged += 1
                self._prune(ego_x, ego_y)
                return
            norm = float(np.linalg.norm(heading))
            if norm < 1e-9:
                return
            heading = heading / norm
            behind = np.nonzero((self._points - local[0]) @ heading < 0.0)[0]
            split = int(behind[-1]) + 1 if len(behind) else 0
            if float(np.min(np.linalg.norm(self._points - local[0], axis=1))) > \
                    2.0 * RUNTIME_SPACING_M:
                split = len(self._points)   # no overlap: keep all of history
            self._points = np.concatenate([self._points[:split], local], axis=0)
        self.messages_merged += 1
        self._prune(ego_x, ego_y)

    def _prune(self, ego_x: float, ego_y: float) -> None:
        """Drop the tail the ego has left well behind, to bound memory."""
        if self._points is None or len(self._points) < 4:
            return
        d = np.linalg.norm(self._points - np.array([ego_x, ego_y]), axis=1)
        keep_from = max(
            0, int(np.argmin(d)) - int(self._keep_behind_m / RUNTIME_SPACING_M)
        )
        if keep_from > 0:
            self._points = self._points[keep_from:]

    def _arc_and_anchor(self, ego_x: float, ego_y: float) -> tuple[np.ndarray, int]:
        """Cumulative length along the cached polyline, and the ego's place on it."""
        step = np.linalg.norm(np.diff(self._points, axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(step)])
        anchor = int(np.argmin(
            np.linalg.norm(self._points - np.array([ego_x, ego_y]), axis=1)
        ))
        return arc, anchor

    def query_rig(self, ego_x: float, ego_y: float, ego_yaw: float,
                  ahead_m: float = 90.0, behind_m: float = 20.0) -> np.ndarray | None:
        """The cached route in the current rig frame, trimmed to the ego's view.

        Trimming is by distance *along the route*, not by the forward coordinate
        in the rig frame. Through a 90 degree turn the two disagree badly: a
        point 80 m along the route sits off to the side, its forward coordinate
        barely 30 m, and a forward-coordinate window would discard precisely
        the part of the route the turn is about.
        """
        if self._points is None or len(self._points) < 2:
            return None
        arc, anchor = self._arc_and_anchor(ego_x, ego_y)
        keep = (arc >= arc[anchor] - behind_m) & (arc <= arc[anchor] + ahead_m)
        if keep.sum() < 2:
            return None
        return _local_to_rig(self._points[keep], ego_x, ego_y, ego_yaw)

    def has_near_field(self, ego_x: float, ego_y: float, ego_yaw: float) -> bool:
        """Does the cache reach back to the ego, rather than starting ahead of it?

        True once the ego's closest point on the cached polyline has route
        behind it as well as ahead -- which is what "the near field is covered"
        means, and unlike a count of metres driven it stays correct when the
        ego stalls, reverses, or is repositioned.
        """
        if self._points is None or len(self._points) < 3:
            return False
        _, anchor = self._arc_and_anchor(ego_x, ego_y)
        return 0 < anchor < len(self._points) - 1
