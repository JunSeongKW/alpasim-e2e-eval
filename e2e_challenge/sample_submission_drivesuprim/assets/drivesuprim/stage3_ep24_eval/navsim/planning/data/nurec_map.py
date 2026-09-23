"""nuPlan ``AbstractMap`` adapter for compact NuRec ClipGT map bundles."""

from __future__ import annotations

import math
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.state_representation import Point2D, StateSE2
from nuplan.common.maps.abstract_map import AbstractMap, MapObject
from nuplan.common.maps.abstract_map_objects import (
    LaneGraphEdgeMapObject,
    PolygonMapObject,
    PolylineMapObject,
    RoadBlockGraphEdgeMapObject,
)
from nuplan.common.maps.maps_datatypes import RasterLayer, RasterMap, SemanticMapLayer
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import unary_union


NUREC_MAP_PREFIX = "nurec:"
NUREC_MAP_ROOT_ENV = "NUREC_MAP_ROOT"
# ClipGT lane rails sit locally about a metre off the driven line, so a body on
# a normal road often falls partly outside its own lane polygon.  This once
# carried a 1.0 m tolerance to absorb that, which worked for area membership but
# made adjacent lanes overlap by 2 m -- and lane identity, which decides the
# route centerline, then came down to a tie-break.  Drivable-area compliance no
# longer asks about containment at all (see PDMScorer.set_road_edges), so the
# tolerance has nothing left to buy.
NUREC_LANE_MEMBERSHIP_BUFFER_M = 0.0


class NuRecPolyline(PolylineMapObject):
    """Polyline implementation backed by a shapely LineString."""

    def __init__(self, object_id: str, points: Sequence[Sequence[float]]) -> None:
        super().__init__(object_id)
        array = np.asarray(points, dtype=np.float64)
        self._line = LineString(array[:, :2])
        # Planning is planar, but telling a flyover's road edge from the road
        # beneath it is not, so the heights are kept.  One median is not enough
        # on graded ground: a single boundary here spans up to 6.1 m end to end,
        # and its average describes neither end.  ``heights`` keeps the per-vertex
        # values, with the cumulative arc length they sit at, so a caller can ask
        # what this line's height is at a particular point rather than overall.
        self._elevation = float(np.median(array[:, 2])) if array.shape[1] > 2 else 0.0
        if array.shape[1] > 2 and len(array) >= 2:
            steps = np.linalg.norm(np.diff(array[:, :2], axis=0), axis=1)
            self._stations = np.concatenate([[0.0], np.cumsum(steps)])
            self._heights = array[:, 2].copy()
        else:
            self._stations = None
            self._heights = None

    @property
    def linestring(self) -> LineString:
        return self._line

    @property
    def elevation(self) -> float:
        """Representative height of this polyline."""
        return self._elevation

    def elevation_at(self, distance: float) -> float:
        """Height of this polyline at an arc length along it.

        Falls back to the representative height when the source carried no z,
        so callers need no special case for a 2D map.
        """
        if self._stations is None or self._heights is None:
            return self._elevation
        return float(np.interp(float(distance), self._stations, self._heights))

    @property
    def height_profile(self):
        """Arc lengths and heights of this polyline's own vertices, or None.

        The scorer resolves heights for tens of thousands of bodies a frame, so
        it needs the raw arrays rather than a call per point.
        """
        if self._stations is None or self._heights is None:
            return None
        return self._stations, self._heights

    @property
    def length(self) -> float:
        return float(self._line.length)

    @property
    def discrete_path(self) -> List[StateSE2]:
        count = max(2, int(math.ceil(self.length)) + 1)
        distances = np.linspace(0.0, self.length, count)
        result = []
        for idx, distance in enumerate(distances):
            point = self._line.interpolate(float(distance))
            before = self._line.interpolate(float(max(0.0, distance - 0.25)))
            after = self._line.interpolate(float(min(self.length, distance + 0.25)))
            heading = math.atan2(after.y - before.y, after.x - before.x)
            result.append(StateSE2(point.x, point.y, heading))
        return result

    def get_nearest_arc_length_from_position(self, point: Point2D) -> float:
        return float(self._line.project(Point(point.x, point.y)))

    def get_nearest_pose_from_position(self, point: Point2D) -> StateSE2:
        arc = self.get_nearest_arc_length_from_position(point)
        before = self._line.interpolate(max(0.0, arc - 0.25))
        after = self._line.interpolate(min(self.length, arc + 0.25))
        value = self._line.interpolate(arc)
        return StateSE2(value.x, value.y, math.atan2(after.y - before.y, after.x - before.x))

    def get_curvature_at_arc_length(self, arc_length: float) -> float:
        arc = min(max(float(arc_length), 0.0), self.length)
        epsilon = min(1.0, max(self.length / 20.0, 0.1))
        a = self._line.interpolate(max(0.0, arc - epsilon))
        b = self._line.interpolate(arc)
        c = self._line.interpolate(min(self.length, arc + epsilon))
        ab = np.array([b.x - a.x, b.y - a.y])
        bc = np.array([c.x - b.x, c.y - b.y])
        ac = np.array([c.x - a.x, c.y - a.y])
        denominator = np.linalg.norm(ab) * np.linalg.norm(bc) * np.linalg.norm(ac)
        if denominator < 1e-8:
            return 0.0
        return float(2.0 * np.cross(ab, bc) / denominator)


class NuRecBoundary:
    """RoadEdge object exposed through ``SemanticMapLayer.BOUNDARIES``."""

    def __init__(self, object_id: str, points: Sequence[Sequence[float]]) -> None:
        self._id = str(object_id)
        self._baseline_path = NuRecPolyline(object_id, points)

    @property
    def id(self) -> str:
        return self._id

    @property
    def baseline_path(self) -> NuRecPolyline:
        return self._baseline_path


class NuRecPolygon(PolygonMapObject):
    """Generic semantic polygon."""

    def __init__(self, object_id: str, polygon: Any) -> None:
        super().__init__(object_id)
        self._polygon = polygon

    @property
    def polygon(self) -> Any:
        return self._polygon


class NuRecLane(LaneGraphEdgeMapObject):
    """Lane or lane-connector object with graph connectivity."""

    def __init__(self, lane_id: str, record: Mapping[str, Any], map_api: "NuRecMap") -> None:
        super().__init__(lane_id)
        self._record = record
        self._map = map_api
        self._left = NuRecPolyline(f"{lane_id}:left", record["left"])
        self._right = NuRecPolyline(f"{lane_id}:right", record["right"])
        self._baseline = NuRecPolyline(f"{lane_id}:center", record["center"])
        left = np.asarray(record["left"], dtype=np.float64)[:, :2]
        right = np.asarray(record["right"], dtype=np.float64)[:, :2]
        self._raw_polygon = Polygon(np.vstack([left, right[::-1]])).buffer(0)
        self._polygon = self._raw_polygon.buffer(NUREC_LANE_MEMBERSHIP_BUFFER_M)

    @property
    def polygon(self) -> Any:
        """Lane surface plus the membership tolerance -- an area, not an identity.

        Adjacent lanes of one roadblock overlap by twice the buffer here, so a
        point inside this polygon does not say which lane it belongs to.  Use
        ``raw_polygon`` for that.
        """
        return self._polygon

    @property
    def raw_polygon(self) -> Any:
        """Lane surface exactly as the ClipGT rails draw it, with no tolerance."""
        return self._raw_polygon

    @property
    def baseline_path(self) -> NuRecPolyline:
        return self._baseline

    @property
    def left_boundary(self) -> NuRecPolyline:
        return self._left

    @property
    def right_boundary(self) -> NuRecPolyline:
        return self._right

    @property
    def speed_limit_mps(self) -> Optional[float]:
        return self._record.get("speed_limit_mps")

    @property
    def incoming_edges(self) -> List["NuRecLane"]:
        return self._map._lanes_by_ids(self._record.get("previous", []))

    @property
    def outgoing_edges(self) -> List["NuRecLane"]:
        return self._map._lanes_by_ids(self._record.get("next", []))

    @property
    def parallel_edges(self) -> List["NuRecLane"]:
        parent = self.parent()
        return parent.interior_edges if parent is not None else [self]

    def get_roadblock_id(self) -> str:
        return str(self._record["roadblock_id"])

    def parent(self) -> "NuRecRoadBlock":
        return self._map._roadblocks[self.get_roadblock_id()]

    def has_traffic_lights(self) -> bool:
        return False

    @property
    def stop_lines(self) -> List[Any]:
        return []

    @property
    def adjacent_edges(self) -> Tuple[Optional["NuRecLane"], Optional["NuRecLane"]]:
        left = self._map._lanes_by_ids(self._record.get("left_lane", []))
        right = self._map._lanes_by_ids(self._record.get("right_lane", []))
        return (left[0] if left else None, right[0] if right else None)

    def is_left_of(self, other: LaneGraphEdgeMapObject) -> bool:
        return self.adjacent_edges[1] is other

    def is_right_of(self, other: LaneGraphEdgeMapObject) -> bool:
        return self.adjacent_edges[0] is other

    def get_width_left_right(self, point: Point2D, include_outside: bool = False) -> Tuple[float, float]:
        query = Point(point.x, point.y)
        if not include_outside and not self._raw_polygon.covers(query):
            return math.inf, math.inf
        return float(self._left.linestring.distance(query)), float(self._right.linestring.distance(query))

    def oriented_distance(self, point: Point2D) -> float:
        pose = self._baseline.get_nearest_pose_from_position(point)
        dx, dy = point.x - pose.x, point.y - pose.y
        return float(-math.sin(pose.heading) * dx + math.cos(pose.heading) * dy)


class NuRecRoadBlock(RoadBlockGraphEdgeMapObject):
    """Roadblock grouping one or more NuRec lanes."""

    def __init__(self, object_id: str, record: Mapping[str, Any], map_api: "NuRecMap") -> None:
        super().__init__(object_id)
        self._record = record
        self._map = map_api

    @property
    def is_connector(self) -> bool:
        """Whether this block joins other blocks through an intersection.

        One class backs both kinds here, so callers cannot tell them apart by
        type the way nuPlan's separate classes allow.
        """
        return bool(self._record["is_connector"])

    @property
    def polygon(self) -> Any:
        return unary_union([lane.polygon for lane in self.interior_edges]).buffer(0)

    @property
    def incoming_edges(self) -> List["NuRecRoadBlock"]:
        return self._map._roadblocks_by_ids(self._record.get("incoming", []))

    @property
    def outgoing_edges(self) -> List["NuRecRoadBlock"]:
        return self._map._roadblocks_by_ids(self._record.get("outgoing", []))

    @property
    def parallel_edges(self) -> List["NuRecRoadBlock"]:
        return [self]

    @property
    def interior_edges(self) -> List[NuRecLane]:
        return self._map._lanes_by_ids(self._record["lane_ids"])

    @property
    def children_stop_lines(self) -> List[Any]:
        return []


class NuRecMap(AbstractMap):
    """In-memory vector map for one NuRec reconstruction clip."""

    def __init__(self, bundle: Mapping[str, Any]) -> None:
        self._bundle = dict(bundle)
        self._map_name = str(bundle["map_name"])
        self._lanes = {lane_id: NuRecLane(lane_id, record, self) for lane_id, record in bundle["lanes"].items()}
        self._roadblocks = {
            object_id: NuRecRoadBlock(object_id, record, self)
            for object_id, record in bundle["roadblocks"].items()
        }
        self._layers: Dict[SemanticMapLayer, Dict[str, Any]] = {
            layer: {} for layer in SemanticMapLayer
        }
        for lane_id, lane in self._lanes.items():
            block = self._roadblocks[lane.get_roadblock_id()]
            layer = SemanticMapLayer.LANE_CONNECTOR if block._record["is_connector"] else SemanticMapLayer.LANE
            self._layers[layer][lane_id] = lane
        for object_id, roadblock in self._roadblocks.items():
            layer = (
                SemanticMapLayer.ROADBLOCK_CONNECTOR
                if roadblock._record["is_connector"]
                else SemanticMapLayer.ROADBLOCK
            )
            self._layers[layer][object_id] = roadblock
        self._load_polygons(bundle.get("polygons", {}))
        for record in bundle.get("road_boundaries", []):
            boundary = NuRecBoundary(str(record["id"]), record["points"])
            self._layers[SemanticMapLayer.BOUNDARIES][boundary.id] = boundary

    @property
    def rig_bbox(self) -> Optional[Mapping[str, Any]]:
        """Scene-specific AlpaSim ego AABB, when supplied by NuRec extraction."""
        return self._bundle.get("rig_bbox")

    def __reduce__(self) -> Tuple[Any, Tuple[Mapping[str, Any]]]:
        return self.__class__, (self._bundle,)

    def _load_polygons(self, polygons: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
        mapping = {
            "intersection_area": SemanticMapLayer.INTERSECTION,
            "crosswalk": SemanticMapLayer.CROSSWALK,
            "drivable_space": SemanticMapLayer.DRIVABLE_AREA,
        }
        for source, layer in mapping.items():
            for record in polygons.get(source, []):
                shell = np.asarray(record["points"], dtype=np.float64)[:, :2]
                holes = [np.asarray(ring, dtype=np.float64)[:, :2] for ring in record.get("holes", [])]
                polygon = Polygon(shell, holes=holes).buffer(0)
                self._layers[layer][record["id"]] = NuRecPolygon(record["id"], polygon)

    @property
    def map_name(self) -> str:
        return self._map_name

    def _lanes_by_ids(self, ids: Sequence[str]) -> List[NuRecLane]:
        return [self._lanes[value] for value in ids if value in self._lanes]

    def _roadblocks_by_ids(self, ids: Sequence[str]) -> List[NuRecRoadBlock]:
        return [self._roadblocks[value] for value in ids if value in self._roadblocks]

    def _get_roadblock(self, object_id: str) -> Optional[NuRecRoadBlock]:
        block = self._roadblocks.get(str(object_id))
        return block if block is not None and not block._record["is_connector"] else None

    def _get_roadblock_connector(self, object_id: str) -> Optional[NuRecRoadBlock]:
        block = self._roadblocks.get(str(object_id))
        return block if block is not None and block._record["is_connector"] else None

    def get_available_map_objects(self) -> List[SemanticMapLayer]:
        return [layer for layer, objects in self._layers.items() if objects]

    def get_available_raster_layers(self) -> List[SemanticMapLayer]:
        return []

    def get_raster_map_layer(self, layer: SemanticMapLayer) -> RasterLayer:
        raise ValueError(f"NuRec vector maps do not provide raster layer {layer}")

    def get_raster_map(self, layers: List[SemanticMapLayer]) -> RasterMap:
        raise ValueError("NuRec vector maps do not provide raster layers")

    def get_all_map_objects(self, point: Point2D, layer: SemanticMapLayer) -> List[MapObject]:
        query = Point(point.x, point.y)
        return [obj for obj in self._layers.get(layer, {}).values() if obj.polygon.covers(query)]

    def get_one_map_object(self, point: Point2D, layer: SemanticMapLayer) -> Optional[MapObject]:
        objects = self.get_all_map_objects(point, layer)
        if len(objects) > 1:
            raise AssertionError(f"more than one {layer} object contains {point}")
        return objects[0] if objects else None

    def is_in_layer(self, point: Point2D, layer: SemanticMapLayer) -> bool:
        return bool(self.get_all_map_objects(point, layer))

    @staticmethod
    def _query_geometry(map_object: Any) -> Any:
        """Whatever geometry a layer's objects carry.

        Most layers hold areas, but BOUNDARIES holds road-edge polylines -- a
        road edge is a line the ego must not cross, and has no interior.
        """
        polygon = getattr(map_object, "polygon", None)
        if polygon is not None:
            return polygon
        return map_object.baseline_path.linestring

    def get_proximal_map_objects(
        self, point: Point2D, radius: float, layers: List[SemanticMapLayer]
    ) -> Dict[SemanticMapLayer, List[MapObject]]:
        query = Point(point.x, point.y).buffer(radius)
        return {
            layer: [
                obj
                for obj in self._layers.get(layer, {}).values()
                if self._query_geometry(obj).intersects(query)
            ]
            for layer in layers
        }

    def get_map_object(self, object_id: str, layer: SemanticMapLayer) -> Optional[MapObject]:
        return self._layers.get(layer, {}).get(str(object_id))

    def get_distance_to_nearest_map_object(
        self, point: Point2D, layer: SemanticMapLayer
    ) -> Tuple[Optional[str], Optional[float]]:
        query = Point(point.x, point.y)
        objects = self._layers.get(layer, {})
        if not objects:
            return None, np.nan
        object_id, obj = min(objects.items(), key=lambda item: item[1].polygon.distance(query))
        return object_id, float(obj.polygon.distance(query))

    def get_distance_to_nearest_raster_layer(self, point: Point2D, layer: SemanticMapLayer) -> float:
        raise ValueError("NuRec vector maps do not provide raster layers")

    def get_distances_matrix_to_nearest_map_object(
        self, points: List[Point2D], layer: SemanticMapLayer
    ) -> Optional[npt.NDArray[np.float64]]:
        return np.asarray([self.get_distance_to_nearest_map_object(point, layer)[1] for point in points])

    def initialize_all_layers(self) -> None:
        return None


def is_nurec_map_name(map_name: str) -> bool:
    return str(map_name).startswith(NUREC_MAP_PREFIX)


def load_nurec_map(map_name: str, map_root: Optional[Path] = None) -> NuRecMap:
    """Load ``<clip-id>.pkl`` from ``NUREC_MAP_ROOT``."""
    if not is_nurec_map_name(map_name):
        raise ValueError(f"not a NuRec map name: {map_name}")
    root_value = map_root or os.environ.get(NUREC_MAP_ROOT_ENV)
    if not root_value:
        raise RuntimeError(f"set {NUREC_MAP_ROOT_ENV} to the extracted NuRec map-bundle directory")
    root = Path(root_value)
    clip_id = map_name[len(NUREC_MAP_PREFIX) :]
    path = root / f"{clip_id}.pkl"
    if not path.is_file():
        raise FileNotFoundError(f"NuRec map bundle not found: {path}")
    with path.open("rb") as stream:
        bundle = pickle.load(stream)
    return NuRecMap(bundle)
