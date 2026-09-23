from abc import ABC
from typing import Dict, List, Optional, Tuple

import numpy as np
import numpy.typing as npt
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2
from nuplan.common.maps.abstract_map import AbstractMap
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject, RoadBlockGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer
from nuplan.planning.simulation.planner.abstract_planner import AbstractPlanner
from shapely.geometry import Point

from navsim.planning.simulation.planner.pdm_planner.observation.pdm_occupancy_map import PDMDrivableMap
from navsim.planning.simulation.planner.pdm_planner.utils.graph_search.dijkstra import Dijkstra
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import normalize_angle
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_path import PDMPath
from navsim.planning.simulation.planner.pdm_planner.utils.route_utils import route_roadblock_correction


class AbstractPDMPlanner(AbstractPlanner, ABC):
    """
    Interface for planners incorporating PDM-* variants.
    """

    def __init__(
        self,
        map_radius: float,
    ):
        """
        Constructor of AbstractPDMPlanner.
        :param map_radius: radius around ego to consider
        """

        self._map_radius: int = map_radius  # [m]
        self._iteration: int = 0

        # lazy loaded
        self._map_api: Optional[AbstractMap] = None
        self._route_roadblock_dict: Optional[Dict[str, RoadBlockGraphEdgeMapObject]] = None
        self._route_lane_dict: Optional[Dict[str, LaneGraphEdgeMapObject]] = None

        self._centerline: Optional[PDMPath] = None
        self._drivable_area_map: Optional[PDMDrivableMap] = None

    def _load_route_dicts(self, route_roadblock_ids: List[str]) -> None:
        """
        Loads roadblock and lane dictionaries of the target route from the map-api.
        :param route_roadblock_ids: ID's of on-route roadblocks
        """
        # remove repeated ids while remaining order in list
        route_roadblock_ids = list(dict.fromkeys(route_roadblock_ids))

        self._route_roadblock_dict = {}
        self._route_lane_dict = {}

        for id_ in route_roadblock_ids:
            block = self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or self._map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)

            self._route_roadblock_dict[block.id] = block

            for lane in block.interior_edges:
                self._route_lane_dict[lane.id] = lane

    def _route_roadblock_correction(self, ego_state: EgoState) -> None:
        """
        Corrects the roadblock route and reloads lane-graph dictionaries.
        :param ego_state: state of the ego vehicle.
        """
        route_roadblock_ids = route_roadblock_correction(ego_state.rear_axle, self._map_api, self._route_roadblock_dict)
        self._load_route_dicts(route_roadblock_ids)

    def _get_discrete_centerline(self, current_lane: LaneGraphEdgeMapObject, search_depth: int = 30) -> List[StateSE2]:
        """
        Applies a Dijkstra search on the lane-graph to retrieve discrete centerline.
        :param current_lane: lane object of starting lane.
        :param search_depth: depth of search (for runtime), defaults to 30
        :return: list of discrete states on centerline (x,y,θ)
        """

        roadblocks = list(self._route_roadblock_dict.values())
        roadblock_ids = list(self._route_roadblock_dict.keys())

        # find current roadblock index
        start_idx = np.argmax(np.array(roadblock_ids) == current_lane.get_roadblock_id())
        roadblock_window = roadblocks[start_idx : start_idx + search_depth]

        graph_search = Dijkstra(current_lane, list(self._route_lane_dict.keys()))
        route_plan, path_found = graph_search.search(roadblock_window[-1])

        centerline_discrete_path: List[StateSE2] = []
        for lane in route_plan:
            centerline_discrete_path.extend(lane.baseline_path.discrete_path)

        return centerline_discrete_path

    @property
    def _lane_polygons_are_buffered(self) -> bool:
        """Whether lane polygons carry a membership tolerance and so cannot identify a lane.

        NuRec buffers every lane by ``NUREC_LANE_MEMBERSHIP_BUFFER_M`` to absorb
        the ClipGT rail displacement.  NAVSIM's nuPlan lanes are exact, and take
        the original selection path unchanged.
        """
        return str(getattr(self._map_api, "map_name", "")).startswith("nurec:")

    def _get_starting_lane(self, ego_state: EgoState) -> LaneGraphEdgeMapObject:
        """
        Returns the most suitable starting lane, in ego's vicinity.
        :param ego_state: state of ego-vehicle
        :return: lane object (on-route)
        """
        starting_lane: LaneGraphEdgeMapObject = None
        on_route_lanes, heading_error, baseline_distance, inside_raw = self._get_intersecting_lanes(ego_state)

        if on_route_lanes:
            # 1. Option: find lanes from lane occupancy-map
            starting_lane = on_route_lanes[
                self._select_starting_lane(heading_error, baseline_distance, inside_raw)
            ]
            return starting_lane

        else:
            # 2. Option: find any intersecting or close lane on-route
            closest_distance = np.inf
            for edge in self._route_lane_dict.values():
                if edge.contains_point(ego_state.center):
                    starting_lane = edge
                    break

                distance = edge.polygon.distance(ego_state.car_footprint.geometry)
                if distance < closest_distance:
                    starting_lane = edge
                    closest_distance = distance

        return starting_lane

    def _select_starting_lane(
        self,
        heading_error: List[float],
        baseline_distance: List[float],
        inside_raw: List[bool],
    ) -> int:
        """Picks which candidate lane the ego is actually in.

        Upstream takes the lowest heading error.  That decides correctly only
        when the candidates point different ways.  NuRec lane polygons carry a
        1.0 m membership buffer, so adjacent lanes of the same roadblock overlap
        by 2 m and both contain the ego rear axle -- measured over 700 frames,
        48.3% had two or more candidates, against 8.3% on the unbuffered
        polygons.  Parallel lanes have near-identical heading error, so the
        winner came down to STRtree ordering, and the starting lane flipped to
        the neighbour.  That moves the whole centerline one lane sideways
        (median 3.12 m), which broke lane keeping on 4,973 frames and silently
        re-based ego progress, since both read ``self._centerline``.

        Containment in the unbuffered polygon answers "which lane", so it goes
        first.  Heading only rules out lanes pointing the other way.  What
        actually separates two same-direction neighbours is which baseline the
        ego is closer to.
        """
        heading_error_array = np.abs(np.asarray(heading_error, dtype=np.float64))
        if not self._lane_polygons_are_buffered:
            return int(np.argmin(heading_error_array))

        baseline_distance_array = np.asarray(baseline_distance, dtype=np.float64)
        candidates = np.flatnonzero(np.asarray(inside_raw, dtype=np.bool_))
        if len(candidates) == 0:
            # Ego sits in the tolerance band only -- every candidate stays in.
            candidates = np.arange(len(heading_error_array))

        same_direction = candidates[heading_error_array[candidates] < np.pi / 2.0]
        if len(same_direction) > 0:
            candidates = same_direction

        # np.lexsort orders by the last key first: nearest baseline decides,
        # heading error only breaks an exact tie.
        order = np.lexsort((heading_error_array[candidates], baseline_distance_array[candidates]))
        return int(candidates[order[0]])

    def _get_intersecting_lanes(
        self, ego_state: EgoState
    ) -> Tuple[List[LaneGraphEdgeMapObject], List[float], List[float], List[bool]]:
        """
        Returns on-route lanes where the ego-vehicle intersects, with the signals
        needed to tell them apart.
        :param ego_state: state of ego-vehicle
        :return: lane objects, heading errors [rad], baseline distances [m], and
            whether ego is inside each lane's unbuffered polygon.
        """
        assert self._drivable_area_map, "AbstractPDMPlanner: Drivable area map must be initialized first!"

        ego_position_array: npt.NDArray[np.float64] = ego_state.rear_axle.array
        ego_rear_axle_point: Point = Point(*ego_position_array)
        ego_heading: float = ego_state.rear_axle.heading

        intersecting_lanes = self._drivable_area_map.intersects(ego_rear_axle_point)

        on_route_lanes, on_route_heading_errors = [], []
        on_route_baseline_distances, on_route_inside_raw = [], []
        for lane_id in intersecting_lanes:
            if lane_id in self._route_lane_dict.keys():
                # collect baseline path as array
                lane_object = self._route_lane_dict[lane_id]
                lane_discrete_path: List[StateSE2] = lane_object.baseline_path.discrete_path
                lane_state_se2_array = np.array([state.array for state in lane_discrete_path], dtype=np.float64)
                # calculate nearest state on baseline
                lane_distances = (ego_position_array[None, ...] - lane_state_se2_array) ** 2
                lane_distances = lane_distances.sum(axis=-1) ** 0.5
                nearest_idx = np.argmin(lane_distances)

                # calculate heading error
                heading_error = lane_discrete_path[nearest_idx].heading - ego_heading
                heading_error = np.abs(normalize_angle(heading_error))

                # Maps whose lanes are exact have no unbuffered variant to ask;
                # they report True so the selection stays on the original path.
                raw_polygon = getattr(lane_object, "raw_polygon", None)
                inside_raw = True if raw_polygon is None else bool(raw_polygon.covers(ego_rear_axle_point))

                # add lane to candidates
                on_route_lanes.append(lane_object)
                on_route_heading_errors.append(heading_error)
                on_route_baseline_distances.append(float(lane_distances[nearest_idx]))
                on_route_inside_raw.append(inside_raw)

        return on_route_lanes, on_route_heading_errors, on_route_baseline_distances, on_route_inside_raw
