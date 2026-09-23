"""Build the PAI-Track global route for prepared NuRec logs.

The Runtime hands the driver a fixed-size route just before every ``Drive()``
call: twenty ``(x, y, z)`` slots in the ego rig frame of that timestamp, of
which the ten covering roughly 42 m to 80 m ahead are finite and the remaining
ten are NaN padding. This module reproduces that message offline so training
sees the same signal the driver will see at inference.

The route is *not* the recorded trajectory. It is the centerline of the lane
sequence the recorded ego actually drove, which ``nurec_map_data`` already
resolved into ``matched_ego[i]["lane_id"]``.

Two Runtime rules truncate the message: a route with less than the full
horizon left, and a resampled gap outside the allowed band. In both cases
every waypoint from that point on is NaN.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


from navsim.common.route_contract import (  # noqa: F401  (re-exported)
    MAX_WAYPOINT_GAP_M,
    MIN_WAYPOINT_GAP_M,
    ROUTE_HORIZON_M,
    ROUTE_NEAR_CUTOFF_M,
    ROUTE_SLOTS,
    ROUTE_SPACING_M,
)

# If the ego sits farther than this from the route, we do not know where it is
# along the route and the projection lands somewhere arbitrary. Observed offsets
# are 0.44 m median and 2.0 m at p95, so this only trips on frames where the ego
# left the reconstructed lane map. Emitting NaN there is honest; emitting a
# waypoint 100 m away is not.
MAX_EGO_ROUTE_OFFSET_M = 10.0

_JOIN_EPS_M = 0.05


class RouteBuildError(ValueError):
    """Raised when a clip cannot produce a route polyline."""


def gt_lane_sequence(bundle: Mapping[str, Any]) -> List[str]:
    """Lane ids the recorded ego drove, in order, without consecutive repeats."""
    sequence: List[str] = []
    for match in bundle["matched_ego"]:
        lane_id = match.get("lane_id")
        if lane_id and (not sequence or sequence[-1] != lane_id):
            sequence.append(lane_id)
    return sequence


def _centerline(lanes: Mapping[str, Any], lane_id: str) -> np.ndarray:
    center = np.asarray(lanes[lane_id]["center"], dtype=np.float64)
    if center.ndim != 2 or center.shape[0] < 2:
        return np.empty((0, 2))
    return center[:, :2]


def _append_polyline(points: List[np.ndarray], segment: np.ndarray) -> None:
    """Append a centerline, dropping a start point that repeats the previous end."""
    if segment.shape[0] == 0:
        return
    if points and np.linalg.norm(segment[0] - points[-1]) < _JOIN_EPS_M:
        segment = segment[1:]
    points.extend(segment)


def _project_point(polyline: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Closest point on ``polyline`` to ``point``."""
    starts, ends = polyline[:-1], polyline[1:]
    deltas = ends - starts
    lengths = (deltas ** 2).sum(axis=1)
    lengths[lengths < 1e-12] = 1e-12
    t = np.clip(((point - starts) * deltas).sum(axis=1) / lengths, 0.0, 1.0)
    projections = starts + t[:, None] * deltas
    return projections[int(np.argmin(np.linalg.norm(projections - point, axis=1)))]


def _straightest_next(lanes: Mapping[str, Any], lane_id: str, heading: np.ndarray) -> Optional[str]:
    """Pick the connected lane that best continues the current heading.

    A roadblock can fan out into several successors. Taking an arbitrary one
    would make the route turn off at random; continuing straight matches what a
    navigation route means and keeps the result deterministic.
    """
    candidates = [c for c in sorted(lanes[lane_id].get("next") or []) if c in lanes]
    best, best_score = None, -np.inf
    for candidate in candidates:
        center = _centerline(lanes, candidate)
        if center.shape[0] < 2:
            continue
        direction = center[1] - center[0]
        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            continue
        score = float(np.dot(direction / norm, heading))
        if score > best_score:
            best, best_score = candidate, score
    return best


def _matched_lane_ids(frames: Sequence[Mapping[str, Any]], bundle: Mapping[str, Any]) -> List[Optional[str]]:
    """Lane the recorded ego was on, per log frame.

    ``matched_ego`` is sampled at 0.1 s while prepared logs are at 0.5 s, so each
    frame takes the nearest map ego sample.
    """
    matched = bundle["matched_ego"]
    times = np.asarray([int(m["timestamp_us"]) for m in matched])
    lane_ids: List[Optional[str]] = []
    for frame in frames:
        index = int(np.argmin(np.abs(times - int(frame["timestamp"]))))
        lane_ids.append(matched[index].get("lane_id"))
    return lane_ids


def route_polyline(
    frames: Sequence[Mapping[str, Any]],
    bundle: Mapping[str, Any],
    min_extension_m: float = ROUTE_HORIZON_M,
) -> np.ndarray:
    """Recorded path projected onto its lane centerlines, extended past the end.

    Concatenating whole centerlines instead would kink wherever the GT changes
    lane, because two parallel lanes are laterally offset: the route would step
    sideways between consecutive waypoints. Projecting the recorded path keeps
    the lane change as gradual as the ego drove it.

    The last recorded pose still needs ``ROUTE_HORIZON_M`` of route ahead of it,
    so the final lane is followed through ``next`` links to supply the rest.
    """
    lanes = bundle["lanes"]
    lane_ids = _matched_lane_ids(frames, bundle)

    points: List[np.ndarray] = []
    last_lane: Optional[str] = None
    for frame, lane_id in zip(frames, lane_ids):
        if not lane_id or lane_id not in lanes:
            continue
        center = _centerline(lanes, lane_id)
        if center.shape[0] < 2:
            continue
        ego_xy, _ = ego_pose(frame)
        points.append(_project_point(center, ego_xy))
        last_lane = lane_id
    if not points or last_lane is None:
        raise RouteBuildError(f"{bundle['clip_id']}: no matched lane, cannot build a route")

    # Poses that overshoot their matched lane all clamp to the same endpoint, so
    # the tail of the projected path can repeat. Collapse repeats now: the
    # extension below reads a heading off the last two points and would give up
    # on a zero-length step, leaving the clip with no route at all.
    points = _dedupe(points)

    # Continue along the remainder of the final lane, then onward. This runs
    # before the two-point check because a clip matched at a single pose still
    # has a usable route: the rest of that lane.
    center = _centerline(lanes, last_lane)
    ahead = _arclength(center)
    tail_start = project_arclength(center, ahead, points[-1])
    _append_polyline(points, center[ahead > tail_start + _JOIN_EPS_M])
    if len(points) < 2:
        raise RouteBuildError(f"{bundle['clip_id']}: route polyline collapsed to one point")

    visited = {last_lane}
    current, extension = last_lane, 0.0
    while extension < min_extension_m:
        heading = points[-1] - points[-2]
        norm = np.linalg.norm(heading)
        if norm < 1e-9:
            break
        nxt = _straightest_next(lanes, current, heading / norm)
        if nxt is None or nxt in visited:
            break
        before = len(points)
        _append_polyline(points, _centerline(lanes, nxt))
        if len(points) == before:
            break
        extension += float(
            np.linalg.norm(np.diff(np.asarray(points[before - 1:]), axis=0), axis=1).sum()
        )
        visited.add(nxt)
        current = nxt

    polyline = np.asarray(points, dtype=np.float64)
    keep = np.concatenate([[True], np.linalg.norm(np.diff(polyline, axis=0), axis=1) > 1e-9])
    return polyline[keep]


def _dedupe(points: List[np.ndarray]) -> List[np.ndarray]:
    """Drop consecutive points that sit on top of each other."""
    kept = [points[0]]
    for point in points[1:]:
        if np.linalg.norm(point - kept[-1]) > _JOIN_EPS_M:
            kept.append(point)
    return kept


def _arclength(polyline: np.ndarray) -> np.ndarray:
    steps = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(steps)])


def project_arclength(polyline: np.ndarray, cumulative: np.ndarray, point: np.ndarray) -> float:
    """Arclength of the closest point on ``polyline`` to ``point``."""
    starts, ends = polyline[:-1], polyline[1:]
    deltas = ends - starts
    lengths = (deltas ** 2).sum(axis=1)
    lengths[lengths < 1e-12] = 1e-12
    t = np.clip(((point - starts) * deltas).sum(axis=1) / lengths, 0.0, 1.0)
    projections = starts + t[:, None] * deltas
    index = int(np.argmin(np.linalg.norm(projections - point, axis=1)))
    return float(cumulative[index] + t[index] * np.sqrt(lengths[index]))


def route_waypoints(
    polyline: np.ndarray,
    cumulative: np.ndarray,
    ego_xy: np.ndarray,
    ego_heading: float,
    slots: int = ROUTE_SLOTS,
    spacing_m: float = ROUTE_SPACING_M,
    near_cutoff_m: float = ROUTE_NEAR_CUTOFF_M,
    max_offset_m: float = MAX_EGO_ROUTE_OFFSET_M,
    min_gap_m: float = MIN_WAYPOINT_GAP_M,
    max_gap_m: float = MAX_WAYPOINT_GAP_M,
) -> np.ndarray:
    """The route message for one timestamp: ``(slots, 3)`` with NaN padding."""
    start = project_arclength(polyline, cumulative, ego_xy)
    nearest = np.argmin(np.abs(cumulative - start))
    if float(np.linalg.norm(polyline[int(nearest)] - ego_xy)) > max_offset_m:
        return np.full((slots, 3), np.nan, dtype=np.float32)

    # Sample every slot first: the near-field trim below is measured on the
    # sampled points, not on the arclength that produced them.
    offsets = np.arange(slots, dtype=np.float64) * spacing_m
    targets = start + offsets
    available = targets <= cumulative[-1]
    sampled = np.stack(
        [np.interp(targets, cumulative, polyline[:, axis]) for axis in (0, 1)], axis=1
    )

    cos_h, sin_h = np.cos(ego_heading), np.sin(ego_heading)
    delta = sampled - ego_xy
    local = np.stack(
        [delta[:, 0] * cos_h + delta[:, 1] * sin_h, -delta[:, 0] * sin_h + delta[:, 1] * cos_h],
        axis=1,
    )

    reach = int(available.sum())  # route exhaustion: a prefix run
    if reach == 0:
        return np.full((slots, 3), np.nan, dtype=np.float32)

    # The near field is trimmed on distance travelled *along the sampled
    # points*, not on arclength. On a curve a chord is shorter than its arc, so
    # the cumulative chord reaches the cutoff one slot later than the arclength
    # does and the message comes back one waypoint short. Trimming on arclength
    # instead would silently paper over that.
    steps = np.linalg.norm(np.diff(local[:reach], axis=0), axis=1)
    chord = np.concatenate([[0.0], np.cumsum(steps)])
    first = int(np.searchsorted(chord, near_cutoff_m, side="left"))
    if first >= reach:
        return np.full((slots, 3), np.nan, dtype=np.float32)

    delivered = local[first:reach]
    if len(delivered) >= 2:
        # Once the delivered spacing leaves the band the route is no longer
        # trustworthy, and every waypoint after that gap is dropped. This
        # always leaves at least one, so it can never empty the message.
        gaps = np.linalg.norm(np.diff(delivered, axis=0), axis=1)
        outside = np.flatnonzero((gaps < min_gap_m) | (gaps > max_gap_m))
        if outside.size:
            delivered = delivered[: int(outside[0]) + 1]

    waypoints = np.full((slots, 3), np.nan, dtype=np.float32)
    waypoints[: len(delivered), :2] = delivered
    # z is always zero on the route; keep it finite exactly where x/y are.
    waypoints[: len(delivered), 2] = 0.0
    return waypoints


def ego_pose(frame: Mapping[str, Any]) -> Tuple[np.ndarray, float]:
    """Planar position and heading of a prepared NuRec log frame."""
    transform = np.asarray(frame["ego2global"], dtype=np.float64)
    return transform[:2, 3].copy(), float(np.arctan2(transform[1, 0], transform[0, 0]))


def build_routes(frames: Sequence[Mapping[str, Any]], bundle: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    """Route message per frame token."""
    polyline = route_polyline(frames, bundle)
    cumulative = _arclength(polyline)
    routes: Dict[str, np.ndarray] = {}
    for frame in frames:
        ego_xy, heading = ego_pose(frame)
        routes[str(frame["token"])] = route_waypoints(polyline, cumulative, ego_xy, heading)
    return routes


def _write_atomic(path: Path, payload: Any) -> None:
    part = path.with_suffix(path.suffix + f".{os.getpid()}.part")
    with part.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    part.replace(path)


def process_logs(log_root: Path, map_root: Path, output_root: Path) -> Dict[str, Any]:
    files = sorted(log_root.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"no prepared logs in {log_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {
        "logs": 0,
        "frames": 0,
        "failed_logs": [],
        "valid_slot_counts": {},
        "frames_without_route": 0,
        "slots": ROUTE_SLOTS,
        "spacing_m": ROUTE_SPACING_M,
        "near_cutoff_m": ROUTE_NEAR_CUTOFF_M,
    }
    for source in files:
        with source.open("rb") as stream:
            frames = pickle.load(stream)
        clip_id = str(frames[0]["map_location"]).split(":", 1)[1]
        map_path = map_root / f"{clip_id}.pkl"
        if not map_path.is_file():
            summary["failed_logs"].append({"log": source.stem, "reason": "missing map bundle"})
            continue
        with map_path.open("rb") as stream:
            bundle = pickle.load(stream)
        try:
            routes = build_routes(frames, bundle)
        except RouteBuildError as error:
            summary["failed_logs"].append({"log": source.stem, "reason": str(error)})
            continue
        _write_atomic(output_root / source.name, routes)
        summary["logs"] += 1
        summary["frames"] += len(routes)
        for waypoints in routes.values():
            count = int(np.isfinite(waypoints[:, 0]).sum())
            summary["valid_slot_counts"][count] = summary["valid_slot_counts"].get(count, 0) + 1
            summary["frames_without_route"] += int(count == 0)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    summary = process_logs(args.log_root, args.map_root, args.output_root)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
