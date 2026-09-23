"""Extract the compact vector-map contract embedded in a NuRec USDZ artifact.

The USDZ files are multi-gigabyte reconstruction artifacts, while their ClipGT
map parquet files are only a few megabytes.  This module converts those parquet
files into a small, pickleable bundle consumed by :mod:`nurec_map`.  No mesh,
volume, image, or checkpoint payload is copied.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import pickle
import zipfile
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Set, Tuple

import numpy as np
import pyarrow.parquet as pq
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union

from navsim.planning.data.nurec_drivable_surface import build_drivable_union, xodr_driving_surface


FORMAT_VERSION = 6
REQUIRED_PARQUETS = ("lane", "association", "egomotion_estimate")
OPTIONAL_POLYGON_PARQUETS = (
    "intersection_area",
    "crosswalk",
    "drivable_space",
    "gore_area",
    "road_island",
    "buffer_zone",
)


class NuRecMapFormatError(ValueError):
    """Raised when a USDZ does not contain a usable ClipGT vector map."""


def _read_parquet(archive: zipfile.ZipFile, stem: str, required: bool = False) -> List[Dict[str, Any]]:
    name = f"clipgt/{stem}.parquet"
    if name not in archive.namelist():
        if required:
            raise NuRecMapFormatError(f"missing required USDZ member: {name}")
        return []
    return pq.read_table(io.BytesIO(archive.read(name))).to_pylist()


def _points(records: Optional[Sequence[Mapping[str, Any]]]) -> List[List[float]]:
    if not records:
        return []
    return [[float(p["x"]), float(p["y"]), float(p.get("z", 0.0))] for p in records]


def _valid_polygon(points: Sequence[Sequence[float]]) -> bool:
    return len(points) >= 3 and Polygon(np.asarray(points)[:, :2]).buffer(0).area > 1e-3


def _association_index(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Set[str]]]:
    result: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    for row in rows:
        kind = row["key"].get("kind")
        if not kind:
            continue
        subjects = row["association"].get("subjects") or []
        objects = row["association"].get("objects") or []
        for subject in subjects:
            result[str(kind)][str(subject)].update(str(value) for value in objects)
    return result


def _components(nodes: Iterable[str], neighbors: Mapping[str, Set[str]]) -> List[Set[str]]:
    remaining = set(nodes)
    components: List[Set[str]] = []
    while remaining:
        root = min(remaining)
        component: Set[str] = set()
        queue = deque([root])
        while queue:
            node = queue.popleft()
            if node in component:
                continue
            component.add(node)
            remaining.discard(node)
            queue.extend(neighbors.get(node, set()) - component)
        components.append(component)
    return components


def _build_roadblocks(
    lane_ids: Set[str], associations: Mapping[str, Mapping[str, Set[str]]]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, str]]:
    """Group ClipGT lanes into nuPlan-style roadblocks and connectors."""
    lane_to_intersection: Dict[str, str] = {}
    for intersection_id, members in associations.get("INTERSECTION_AREA_TO_LANE", {}).items():
        for lane_id in members & lane_ids:
            lane_to_intersection.setdefault(lane_id, intersection_id)

    roadblocks: Dict[str, Dict[str, Any]] = {}
    lane_to_roadblock: Dict[str, str] = {}
    for intersection_id in sorted(set(lane_to_intersection.values())):
        members = sorted(lane for lane, value in lane_to_intersection.items() if value == intersection_id)
        roadblock_id = f"nurec-rbc:{intersection_id}"
        roadblocks[roadblock_id] = {"lane_ids": members, "is_connector": True}
        lane_to_roadblock.update({lane: roadblock_id for lane in members})

    non_intersection = lane_ids - set(lane_to_roadblock)
    sibling_neighbors: Dict[str, Set[str]] = defaultdict(set)
    for lane_id, siblings in associations.get("ROAD_SEGMENT_SIBLING_LANE", {}).items():
        if lane_id not in non_intersection:
            continue
        for sibling in siblings & non_intersection:
            sibling_neighbors[lane_id].add(sibling)
            sibling_neighbors[sibling].add(lane_id)

    for component in _components(non_intersection, sibling_neighbors):
        roadblock_id = f"nurec-rb:{min(component)}"
        members = sorted(component)
        roadblocks[roadblock_id] = {"lane_ids": members, "is_connector": False}
        lane_to_roadblock.update({lane: roadblock_id for lane in members})

    incoming: Dict[str, Set[str]] = defaultdict(set)
    outgoing: Dict[str, Set[str]] = defaultdict(set)
    for lane_id, next_lanes in associations.get("NEXT_LANE", {}).items():
        source = lane_to_roadblock.get(lane_id)
        if source is None:
            continue
        for next_lane in next_lanes:
            target = lane_to_roadblock.get(next_lane)
            if target is not None and source != target:
                outgoing[source].add(target)
                incoming[target].add(source)
    for roadblock_id, record in roadblocks.items():
        record["incoming"] = sorted(incoming[roadblock_id])
        record["outgoing"] = sorted(outgoing[roadblock_id])
    return roadblocks, lane_to_roadblock


def _lane_heading(centerline: LineString, point: Point) -> float:
    distance = centerline.project(point)
    epsilon = min(0.5, max(centerline.length / 20.0, 0.05))
    p0 = centerline.interpolate(max(0.0, distance - epsilon))
    p1 = centerline.interpolate(min(centerline.length, distance + epsilon))
    return math.atan2(p1.y - p0.y, p1.x - p0.x)


def _angle_error(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def _quaternion_yaw(rotation: Mapping[str, Any]) -> float:
    x, y, z, w = (float(rotation[key]) for key in ("x", "y", "z", "w"))
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _match_ego_route(
    ego_rows: Sequence[Mapping[str, Any]],
    lanes: Mapping[str, Mapping[str, Any]],
    lane_to_roadblock: Mapping[str, str],
    rig_bbox_centroid: Sequence[float] = (0.0, 0.0),
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Match the point used by AlpaSim DDC to the ClipGT route.

    Egomotion is expressed at the rig origin, whereas AlpaSim evaluates DDC at
    the center of ``rig_bbox``.  The center is typically about 1.3--1.5 m in
    front of the origin; matching the origin can therefore select the lane on
    the other side of a lane end or connector seam and make a valid trajectory
    look oncoming.  Use the same world-space point for route construction and
    scoring instead of compensating with a polygon tolerance.
    """
    centroid_x = float(rig_bbox_centroid[0]) if len(rig_bbox_centroid) > 0 else 0.0
    centroid_y = float(rig_bbox_centroid[1]) if len(rig_bbox_centroid) > 1 else 0.0
    lane_geometry = {
        lane_id: (
            Polygon(np.vstack([np.asarray(lane["left"])[:, :2], np.asarray(lane["right"])[::-1, :2]])).buffer(0),
            LineString(np.asarray(lane["center"])[:, :2]),
        )
        for lane_id, lane in lanes.items()
    }
    matched: List[Dict[str, Any]] = []
    previous_lane: Optional[str] = None
    for row in ego_rows:
        pose = row["egomotion_estimate"]
        location = pose["location"]
        yaw = _quaternion_yaw(pose["orientation"])
        point = Point(
            float(location["x"]) + centroid_x * math.cos(yaw) - centroid_y * math.sin(yaw),
            float(location["y"]) + centroid_x * math.sin(yaw) + centroid_y * math.cos(yaw),
        )
        candidates: List[Tuple[float, float, str]] = []
        for lane_id, (polygon, centerline) in lane_geometry.items():
            distance = polygon.distance(point)
            if distance > 12.0:
                continue
            heading_error = _angle_error(_lane_heading(centerline, point), yaw)
            continuity_bonus = -1.5 if lane_id == previous_lane else 0.0
            # Physical membership is the primary route observation.  Some
            # ClipGT lanes have centerline ordering opposite to the driven
            # carriageway; allowing heading to outweigh distance selected a
            # parallel road 4+ m away (e.g. 9dd86799) even while another lane
            # polygon covered the ego.  Heading remains the discriminator for
            # overlapping polygons and near-equal non-containing candidates.
            candidates.append((distance, heading_error + continuity_bonus, lane_id))
        containing = [candidate for candidate in candidates if candidate[0] <= 1e-6]
        if containing:
            lane_id = min(containing, key=lambda item: (item[1], item[2]))[2]
        elif candidates:
            lane_id = min(candidates, key=lambda item: (item[0], item[1], item[2]))[2]
        else:
            lane_id = None
        if lane_id is not None:
            previous_lane = lane_id
        matched.append(
            {
                "timestamp_us": int(row["key"]["timestamp_micros"]),
                "lane_id": lane_id,
                "roadblock_id": lane_to_roadblock.get(lane_id) if lane_id else None,
            }
        )

    route: List[str] = []
    for item in matched:
        roadblock_id = item["roadblock_id"]
        if roadblock_id and (not route or route[-1] != roadblock_id):
            route.append(roadblock_id)
    return matched, route


def extract_map_bundle(usdz_path: Path) -> Dict[str, Any]:
    """Read one NuRec USDZ and return a compact, serializable map bundle."""
    with zipfile.ZipFile(usdz_path) as archive:
        tables = {stem: _read_parquet(archive, stem, required=True) for stem in REQUIRED_PARQUETS}
        for stem in OPTIONAL_POLYGON_PARQUETS:
            tables[stem] = _read_parquet(archive, stem)
        tables["wait_line"] = _read_parquet(archive, "wait_line")
        tables["road_boundary"] = _read_parquet(archive, "road_boundary")
        if "map.xodr" not in archive.namelist() or "rig_trajectories.json" not in archive.namelist():
            raise NuRecMapFormatError(f"{usdz_path}: map.xodr and rig_trajectories.json are required")
        xodr_xml = archive.read("map.xodr").decode("utf-8")
        rig_trajectories_json = archive.read("rig_trajectories.json").decode("utf-8")

    if not tables["lane"]:
        raise NuRecMapFormatError(f"{usdz_path}: lane.parquet is empty")
    clip_id = str(tables["lane"][0]["key"]["clip_id"])
    associations = _association_index(tables["association"])

    lanes: Dict[str, Dict[str, Any]] = {}
    for row in tables["lane"]:
        lane_id = str(row["key"]["map_id"])
        value = row["lane"]
        left, right = _points(value.get("left_rail")), _points(value.get("right_rail"))
        if len(left) < 2 or len(right) < 2:
            continue
        count = min(len(left), len(right))
        left, right = left[:count], right[:count]
        center = ((np.asarray(left) + np.asarray(right)) / 2.0).tolist()
        speed = value.get("speed_limit")
        try:
            speed_mps = float(speed) if speed not in (None, "") else None
        except ValueError:
            speed_mps = None
        lanes[lane_id] = {
            "left": left,
            "right": right,
            "center": center,
            "speed_limit_mps": speed_mps,
            "next": sorted(associations.get("NEXT_LANE", {}).get(lane_id, set())),
            "previous": sorted(associations.get("PREVIOUS_LANE", {}).get(lane_id, set())),
            "left_lane": sorted(associations.get("LEFT_LANE", {}).get(lane_id, set())),
            "right_lane": sorted(associations.get("RIGHT_LANE", {}).get(lane_id, set())),
        }

    roadblocks, lane_to_roadblock = _build_roadblocks(set(lanes), associations)
    for lane_id, roadblock_id in lane_to_roadblock.items():
        lanes[lane_id]["roadblock_id"] = roadblock_id

    polygons: Dict[str, List[Dict[str, Any]]] = {}
    for stem in OPTIONAL_POLYGON_PARQUETS:
        records = []
        for row in tables[stem]:
            value = row.get(stem) or {}
            points = _points(value.get("location"))
            if _valid_polygon(points):
                records.append(
                    {"id": str(row["key"].get("map_id") or f"{stem}-{len(records)}"), "points": points}
                )
        polygons[stem] = records

    lane_shapes = []
    lane_shapes_by_id = {}
    for lane_id, lane in lanes.items():
        polygon = Polygon(
            np.vstack([np.asarray(lane["left"])[:, :2], np.asarray(lane["right"])[::-1, :2]])
        ).buffer(0)
        if not polygon.is_empty:
            lane_shapes.append(polygon)
            lane_shapes_by_id[lane_id] = polygon
    lane_surface = unary_union(lane_shapes).buffer(0)
    try:
        xodr_surface = xodr_driving_surface(xodr_xml, rig_trajectories_json)
    except Exception as error:
        raise NuRecMapFormatError(f"{usdz_path}: failed to build XODR driving surface: {error}") from error
    # EPDMS DAC consumes one area: ClipGT lanes, transformed OpenDRIVE driving
    # surface, and association-supported RoadEdge faces, followed by the
    # production 1.0 m DAC tolerance buffer. RoadEdge itself is never assigned
    # a standalone fixed-width buffer.
    drivable_records, drivable_stats = build_drivable_union(
        lane_surface, xodr_surface, tables["road_boundary"],
        association_rows=tables["association"], lane_polygons=lane_shapes_by_id,
    )
    if not drivable_records:
        raise NuRecMapFormatError(f"{usdz_path}: ClipGT lane union is empty")
    polygons["drivable_space"] = drivable_records

    road_boundaries = []
    for row in tables["road_boundary"]:
        points = _points((row.get("road_boundary") or {}).get("location"))
        if len(points) >= 2:
            road_boundaries.append({
                "id": str((row.get("key") or {}).get("map_id") or f"road-boundary-{len(road_boundaries)}"),
                "points": points,
            })

    rig_document = json.loads(rig_trajectories_json)
    rig_trajectories = rig_document.get("rig_trajectories") or []
    rig_bbox = (rig_trajectories[0].get("rig_bbox") or {}) if rig_trajectories else {}
    rig_bbox_centroid = rig_bbox.get("centroid") or (0.0, 0.0)
    matched_ego, route = _match_ego_route(
        tables["egomotion_estimate"], lanes, lane_to_roadblock, rig_bbox_centroid
    )
    return {
        "format_version": FORMAT_VERSION,
        "clip_id": clip_id,
        "map_name": f"nurec:{clip_id}",
        "source_usdz": str(usdz_path),
        # AlpaSim evaluates each clip with this scene-specific rig AABB.  Keep
        # it in the compact map bundle so metric caching can construct the same
        # ego footprint instead of silently falling back to Pacifica dimensions.
        "rig_bbox": {
            "dim": [float(value) for value in (rig_bbox.get("dim") or [])],
            "centroid": [float(value) for value in rig_bbox_centroid],
            "rot": [float(value) for value in (rig_bbox.get("rot") or (0.0, 0.0, 0.0))],
        },
        "lanes": lanes,
        "roadblocks": roadblocks,
        "polygons": polygons,
        "road_boundaries": road_boundaries,
        "matched_ego": matched_ego,
        "route_roadblock_ids": route,
        "stats": {
            "num_lanes": len(lanes),
            "num_roadblocks": sum(not item["is_connector"] for item in roadblocks.values()),
            "num_connectors": sum(item["is_connector"] for item in roadblocks.values()),
            "num_matched_ego": sum(item["lane_id"] is not None for item in matched_ego),
            "num_ego": len(matched_ego),
            "drivable_union": drivable_stats,
        },
    }


def save_map_bundle(bundle: Mapping[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        pickle.dump(dict(bundle), stream, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usdz", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    summaries = []
    for usdz_path in args.usdz:
        bundle = extract_map_bundle(usdz_path)
        output = args.output_root / f"{bundle['clip_id']}.pkl"
        save_map_bundle(bundle, output)
        summaries.append({"output": str(output), **bundle["stats"]})
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
