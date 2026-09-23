"""Build a conservative NuRec drivable surface from three vector sources.

The surface is the union of ClipGT lane polygons, transformed OpenDRIVE
driving-lane polygons, and only those closed road-boundary faces that overlap
the first two sources.  Open road-boundary polylines are preserved as evidence
but are never blindly buffered into drivable area.

The compact OpenDRIVE sampler supports the line and arc plan-view primitives
used by every map in the NuRec 26.04 release (1,607/1,607 audited clips).
Coordinate conversion follows NVIDIA trajdata's Apache-2.0 XODR utilities.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from shapely import intersection, make_valid, set_precision, union_all
from shapely.geometry import LineString, MultiPoint, Point, Polygon
from shapely.ops import nearest_points, polygonize, unary_union


DRIVABLE_LANE_TYPES = {"driving", "exit", "entry", "onramp", "offramp", "connectingramp", "parking"}
MAX_REASONABLE_LANE_WIDTH_M = 20.0
BOUNDARY_ENDPOINT_SNAP_M = 0.75
BOUNDARY_CUT_CAP_MAX_M = 40.0
BOUNDARY_CAP_SEED_BUFFER_M = 2.0
BOUNDARY_CAP_MIN_COVERAGE = 0.65
BOUNDARY_ASSOCIATION_CAP_MAX_M = 30.0
BOUNDARY_ASSOCIATION_EDGE_COVERAGE = 0.50
FINAL_DRIVABLE_BUFFER_M = 1.0


def _polygonal(geometry):
    """Drop zero-area line/point remnants produced by make_valid."""
    if geometry.is_empty:
        return Polygon()
    if geometry.geom_type == "Polygon":
        return geometry
    parts = []
    for item in getattr(geometry, "geoms", []):
        value = _polygonal(item)
        if not value.is_empty:
            parts.append(value)
    return unary_union(parts) if parts else Polygon()


def _fill_polygon_holes(geometry):
    """Remove every interior ring while preserving disconnected road components."""
    geometry = _polygonal(geometry)
    if geometry.is_empty:
        return Polygon()
    parts = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
    return _polygonal(make_valid(unary_union([Polygon(part.exterior) for part in parts])))


def _stable_polygon_union(geometries, grid_size: float = 0.01):
    """Union heterogeneous map polygons on one precision grid."""
    normalized = []
    for geometry in geometries:
        if geometry is None or geometry.is_empty:
            continue
        value = _polygonal(set_precision(make_valid(geometry), grid_size))
        if not value.is_empty:
            normalized.append(value)
    if not normalized:
        return Polygon()
    return _polygonal(make_valid(union_all(normalized, grid_size=grid_size)))


def _poly(coeff: Sequence[float], ds: np.ndarray) -> np.ndarray:
    a, b, c, d = coeff
    return a + b * ds + c * ds**2 + d * ds**3


def _piecewise(elements: Sequence[ET.Element], positions: np.ndarray, start_key: str) -> np.ndarray:
    if not elements:
        return np.zeros_like(positions)
    starts = np.asarray([float(item.attrib[start_key]) for item in elements])
    index = np.searchsorted(starts, positions, side="right") - 1
    index[index < 0] = 0
    result = np.zeros_like(positions)
    for idx, item in enumerate(elements):
        mask = index == idx
        result[mask] = _poly(tuple(float(item.attrib[k]) for k in "abcd"), positions[mask] - starts[idx])
    return result


def _sample_reference(road: ET.Element, resolution: float) -> Tuple[np.ndarray, np.ndarray]:
    points, headings, ss = [], [], []
    for geometry in road.findall("./planView/geometry"):
        length = float(geometry.attrib["length"])
        local_s = np.linspace(0.0, length, max(2, int(math.ceil(length / resolution)) + 1))
        x0, y0, heading0 = (float(geometry.attrib[k]) for k in ("x", "y", "hdg"))
        arc = geometry.find("arc")
        if geometry.find("line") is not None or (arc is not None and abs(float(arc.attrib["curvature"])) < 1e-9):
            x = x0 + np.cos(heading0) * local_s
            y = y0 + np.sin(heading0) * local_s
            heading = np.full_like(local_s, heading0)
        elif arc is not None:
            curvature = float(arc.attrib["curvature"])
            heading = heading0 + curvature * local_s
            x = x0 + (np.sin(heading) - math.sin(heading0)) / curvature
            y = y0 - (np.cos(heading) - math.cos(heading0)) / curvature
        else:
            raise ValueError(f"unsupported OpenDRIVE geometry in road {road.attrib.get('id')}")
        global_s = float(geometry.attrib.get("s", 0.0)) + local_s
        if points:
            x, y, heading, global_s = x[1:], y[1:], heading[1:], global_s[1:]
        points.extend(np.column_stack([x, y])); headings.extend(heading); ss.extend(global_s)
    return np.asarray(points), np.column_stack([np.asarray(headings), np.asarray(ss)])


def _lane_width_sections(lanes: ET.Element) -> Tuple[Dict[int, List[Tuple[float, Tuple[float, ...]]]], Dict[int, str]]:
    widths: Dict[int, List[Tuple[float, Tuple[float, ...]]]] = {}
    types: Dict[int, str] = {}
    for section in lanes.findall("laneSection"):
        section_s = float(section.attrib.get("s", 0.0))
        for side in ("left", "right"):
            side_node = section.find(side)
            if side_node is None:
                continue
            for lane in side_node.findall("lane"):
                lane_id = int(lane.attrib["id"])
                types.setdefault(lane_id, lane.attrib.get("type", "none").lower())
                for width in lane.findall("width"):
                    start = section_s + float(width.attrib.get("sOffset", 0.0))
                    coeff = tuple(float(width.attrib[k]) for k in "abcd")
                    widths.setdefault(lane_id, []).append((start, coeff))
    for values in widths.values():
        values.sort(key=lambda value: value[0])
    return widths, types


def _eval_width(sections: Sequence[Tuple[float, Tuple[float, ...]]], s: np.ndarray) -> np.ndarray:
    starts = np.asarray([item[0] for item in sections])
    index = np.searchsorted(starts, s, side="right") - 1
    index[index < 0] = 0
    value = np.zeros_like(s)
    for idx, (start, coeff) in enumerate(sections):
        mask = index == idx
        value[mask] = np.maximum(_poly(coeff, s[mask] - start), 0.0)
    return value


def _ecef_to_enu_transform(rig_ecef: np.ndarray, root: ET.Element) -> np.ndarray:
    node = root.find("./header/geoReference")
    if node is None or not node.text:
        return np.eye(4)
    values: Dict[str, float] = {}
    for token in node.text.strip().split():
        for key in ("lat_0", "lon_0", "alt_0", "h_0"):
            if token.startswith(f"+{key}="):
                values[key] = float(token.split("=", 1)[1])
    if "lat_0" not in values or "lon_0" not in values:
        return np.eye(4)
    lat, lon = math.radians(values["lat_0"]), math.radians(values["lon_0"])
    alt = values.get("alt_0", values.get("h_0", 0.0))
    a, flattening = 6378137.0, 1.0 / 298.257223563
    b = a * (1.0 - flattening); eccentricity = (a*a-b*b)/(a*a)
    normal = a / math.sqrt(1.0 - eccentricity * math.sin(lat)**2)
    ref = np.array([(normal+alt)*math.cos(lat)*math.cos(lon),
                    (normal+alt)*math.cos(lat)*math.sin(lon),
                    (normal*(b*b)/(a*a)+alt)*math.sin(lat)])
    rotation = np.array([[-math.sin(lon), math.cos(lon), 0.0],
                         [-math.sin(lat)*math.cos(lon), -math.sin(lat)*math.sin(lon), math.cos(lat)],
                         [math.cos(lat)*math.cos(lon), math.cos(lat)*math.sin(lon), math.sin(lat)]])
    ecef_to_enu = np.eye(4); ecef_to_enu[:3,:3] = rotation; ecef_to_enu[:3,3] = -rotation @ ref
    return ecef_to_enu @ rig_ecef


def _transform_xy(xy: np.ndarray, transform: np.ndarray, z: np.ndarray | None = None) -> np.ndarray:
    xyz1 = np.column_stack([xy, np.zeros(len(xy)) if z is None else z, np.ones(len(xy))])
    return (transform @ xyz1.T).T[:, :2]


def xodr_driving_surface(xodr_xml: str, rig_trajectories_json: str, resolution: float = 1.0):
    root = ET.fromstring(xodr_xml)
    rig = json.loads(rig_trajectories_json)
    transform = np.linalg.inv(_ecef_to_enu_transform(np.asarray(rig["T_world_base"], dtype=float), root))
    polygons = []
    for road in root.findall("road"):
        reference, heading_s = _sample_reference(road, resolution)
        if not len(reference):
            continue
        heading, s = heading_s[:, 0], heading_s[:, 1]
        elevations = road.findall("./elevationProfile/elevation")
        z = _piecewise(elevations, s, "s") if elevations else np.zeros_like(s)
        lanes = road.find("lanes")
        if lanes is None:
            continue
        offsets = lanes.findall("laneOffset")
        offset = _piecewise(offsets, s, "s") if offsets else np.zeros_like(s)
        reference[:, 0] -= offset * np.sin(heading); reference[:, 1] += offset * np.cos(heading)
        widths, lane_types = _lane_width_sections(lanes)
        sampled = {lane_id: _eval_width(values, s) for lane_id, values in widths.items()}
        for side in (1, -1):
            cumulative = np.zeros_like(s)
            for lane_id in sorted((value for value in sampled if value * side > 0), key=abs):
                width = sampled[lane_id]
                # One released clip contains a corrupt cubic yielding an
                # 18-km-wide lane over a 1-m road. Never let such metadata
                # create a giant metric-cache polygon.
                if not np.isfinite(width).all() or np.max(width) > MAX_REASONABLE_LANE_WIDTH_M:
                    continue
                inner, outer = cumulative.copy(), cumulative + width
                cumulative = outer
                if lane_types.get(lane_id, "none") not in DRIVABLE_LANE_TYPES:
                    continue
                inner_xy = np.column_stack([reference[:,0]-side*inner*np.sin(heading), reference[:,1]+side*inner*np.cos(heading)])
                outer_xy = np.column_stack([reference[:,0]-side*outer*np.sin(heading), reference[:,1]+side*outer*np.cos(heading)])
                ring = np.vstack([_transform_xy(inner_xy, transform, z), _transform_xy(outer_xy, transform, z)[::-1]])
                if not np.isfinite(ring).all() or len(np.unique(ring, axis=0)) < 3:
                    continue
                try:
                    poly = make_valid(Polygon(ring))
                except Exception:
                    continue
                if not poly.is_empty:
                    polygons.append(poly)
    return _polygonal(make_valid(unary_union(polygons))) if polygons else Polygon()


def _line_points(rows: Sequence[Mapping[str, Any]]) -> List[LineString]:
    result = []
    for row in rows:
        value = row.get("road_boundary") or {}
        points = [(float(p["x"]), float(p["y"])) for p in value.get("location") or []
                  if p.get("x") is not None and p.get("y") is not None]
        if len(points) >= 2:
            result.append(LineString(points))
    return result


def _boundary_closure_lines(boundary_rows: Sequence[Mapping[str, Any]], seed_surface) -> List[LineString]:
    """Close only endpoint gaps and CUT-to-CUT road cross-sections.

    No road-boundary line is buffered.  Short annotation gaps are joined, and
    clip-cut endpoints are paired only when the connecting segment is mostly
    covered by the lane/XODR seed.  Consequently the resulting polygon follows
    the measured boundary everywhere except at a genuine clip cut.
    """
    endpoints = []
    for row_index, row in enumerate(boundary_rows):
        value = row.get("road_boundary") or {}
        points = [(float(p["x"]), float(p["y"])) for p in value.get("location") or []
                  if p.get("x") is not None and p.get("y") is not None]
        if len(points) < 2:
            continue
        endpoints.append((row_index, 0, points[0], str(value.get("is_first_point_physical_end") or "").upper()))
        endpoints.append((row_index, 1, points[-1], str(value.get("is_last_point_physical_end") or "").upper()))

    candidates = []
    seed_gate = _polygonal(set_precision(
        make_valid(seed_surface.buffer(BOUNDARY_CAP_SEED_BUFFER_M)), 0.01
    ))
    for first in range(len(endpoints)):
        row_a, _, point_a, kind_a = endpoints[first]
        for second in range(first + 1, len(endpoints)):
            row_b, _, point_b, kind_b = endpoints[second]
            if row_a == row_b or kind_a == "TRUE" or kind_b == "TRUE":
                continue
            distance = math.dist(point_a, point_b)
            if distance <= 1e-6:
                continue
            is_short_gap = distance <= BOUNDARY_ENDPOINT_SNAP_M
            is_cut_cap = kind_a == "CUT" and kind_b == "CUT" and distance <= BOUNDARY_CUT_CAP_MAX_M
            if not is_short_gap and not is_cut_cap:
                continue
            connector = LineString([point_a, point_b])
            if is_short_gap:
                coverage = 1.0
            else:
                coverage = intersection(connector, seed_gate, grid_size=0.01).length / distance
            if is_short_gap or (is_cut_cap and coverage >= BOUNDARY_CAP_MIN_COVERAGE):
                # Short topological repairs precede longer clip caps.  Among
                # caps, prefer the cross-section best supported by road seed.
                score = (0 if is_short_gap else 1, distance if is_short_gap else -coverage, distance)
                candidates.append((score, first, second, connector))

    used, closures = set(), []
    for _, first, second, connector in sorted(candidates, key=lambda item: item[0]):
        if first in used or second in used:
            continue
        used.update((first, second)); closures.append(connector)
    return closures


def _association_index(association_rows: Sequence[Mapping[str, Any]]) -> Dict[str, set]:
    result: Dict[str, set] = {}
    for row in association_rows:
        key, value = row.get("key") or {}, row.get("association") or {}
        if key.get("kind") != "ROAD_BOUNDARY_TO_LANE":
            continue
        for boundary_id in value.get("subjects") or []:
            result.setdefault(str(boundary_id), set()).update(str(v) for v in value.get("objects") or [])
    return result


def _association_supported_faces(
    boundary_rows: Sequence[Mapping[str, Any]],
    association_rows: Sequence[Mapping[str, Any]],
    lane_polygons: Mapping[str, Any],
):
    """Close each RoadEdge only against the lanes explicitly associated with it."""
    boundary_to_lanes = _association_index(association_rows)
    accepted = []
    for row in boundary_rows:
        boundary_id = str((row.get("key") or {}).get("map_id") or "")
        related = [lane_polygons[lane_id] for lane_id in boundary_to_lanes.get(boundary_id, set())
                   if lane_id in lane_polygons and not lane_polygons[lane_id].is_empty]
        points = _line_points([row])
        if not related or not points:
            continue
        edge = points[0]
        lane_area = _polygonal(make_valid(unary_union(related)))
        if lane_area.is_empty:
            continue
        lane_edge = lane_area.boundary
        start_on_lane = nearest_points(Point(edge.coords[0]), lane_edge)[1]
        end_on_lane = nearest_points(Point(edge.coords[-1]), lane_edge)[1]
        start_cap = LineString([edge.coords[0], start_on_lane.coords[0]])
        end_cap = LineString([edge.coords[-1], end_on_lane.coords[0]])
        if max(start_cap.length, end_cap.length) > BOUNDARY_ASSOCIATION_CAP_MAX_M:
            continue
        network = unary_union([edge, lane_edge, start_cap, end_cap])
        for face in polygonize(network):
            if face.area < 0.25:
                continue
            shared_edge = face.boundary.intersection(edge.buffer(0.02)).length
            if shared_edge / max(edge.length, 1e-6) < BOUNDARY_ASSOCIATION_EDGE_COVERAGE:
                continue
            # The face must be the lane-adjacent corridor, not an unrelated
            # enclosure encountered elsewhere on a complex lane component.
            if face.distance(lane_area) <= 0.02:
                accepted.append(face)
    return _polygonal(make_valid(unary_union(accepted))) if accepted else Polygon()


def boundary_supported_faces(
    boundary_rows: Sequence[Mapping[str, Any]], seed_surface,
    association_rows: Sequence[Mapping[str, Any]] = (),
    lane_polygons: Mapping[str, Any] = None,
):
    """Return boundary faces closed only at verified gaps and clip cuts."""
    lines = _line_points(boundary_rows)
    closures = _boundary_closure_lines(boundary_rows, seed_surface)
    faces = []
    if association_rows and lane_polygons:
        faces.append(_association_supported_faces(boundary_rows, association_rows, lane_polygons))
    for face in polygonize(unary_union(lines + closures)):
        if face.area < 1.0:
            continue
        overlap = face.intersection(seed_surface).area
        if overlap / face.area >= 0.25:
            faces.append(face)
    if not faces:
        return Polygon(), lines
    return _stable_polygon_union(faces), lines


def _serialize_geometry(geometry, prefix: str) -> List[Dict[str, Any]]:
    parts = list(geometry.geoms) if hasattr(geometry, "geoms") else [geometry]
    records = []
    for part in parts:
        if part.geom_type != "Polygon" or part.area < 1e-3:
            continue
        records.append({
            "id": f"{prefix}-{len(records)}",
            "points": [[float(x), float(y), 0.0] for x, y in part.exterior.coords],
            "holes": [[[float(x), float(y), 0.0] for x, y in ring.coords] for ring in part.interiors],
        })
    return records


def build_drivable_union(
    lane_surface, xodr_surface, boundary_rows,
    association_rows: Sequence[Mapping[str, Any]] = (),
    lane_polygons: Mapping[str, Any] = None,
    buffer_m: float = FINAL_DRIVABLE_BUFFER_M,
):
    # A centimetre grid removes sub-millimetre non-noded intersections in raw
    # autolabel/XODR edges while remaining far below map and vehicle resolution.
    lane_surface = _polygonal(set_precision(make_valid(lane_surface), 0.01))
    xodr_surface = _polygonal(set_precision(make_valid(xodr_surface), 0.01))
    seed = _stable_polygon_union([lane_surface, xodr_surface])
    boundary_faces, lines = boundary_supported_faces(
        boundary_rows, seed, association_rows=association_rows, lane_polygons=lane_polygons
    )
    # Preserve interior rings. Whether islands/gore should be filled is a
    # metric-policy decision and is intentionally not baked into map geometry.
    raw_final = _stable_polygon_union([seed, boundary_faces])
    # EPDMS tests the complete vehicle footprint with a strict polygon covers
    # predicate. The audited 1.0 m map tolerance absorbs ordinary annotation /
    # scene-specific vehicle-outline disagreement while remaining far below
    # the expansion needed to repair genuinely missing map coverage.
    final = _stable_polygon_union([raw_final.buffer(buffer_m)]) if buffer_m > 0.0 else raw_final
    return _serialize_geometry(final, "nurec-drivable"), {
        "lane_area_m2": float(lane_surface.area),
        "xodr_area_m2": float(xodr_surface.area),
        "boundary_face_area_m2": float(boundary_faces.area),
        "raw_union_area_m2": float(raw_final.area),
        "final_buffer_m": float(buffer_m),
        "union_area_m2": float(final.area),
        "num_boundary_lines": len(lines),
        "lane_coverage": min(1.0, float(final.intersection(lane_surface).area / lane_surface.area)) if lane_surface.area else 1.0,
        "xodr_coverage": min(1.0, float(final.intersection(xodr_surface).area / xodr_surface.area)) if xodr_surface.area else 1.0,
    }
