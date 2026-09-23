"""Builds the metric cache each EPDMS label is scored against.

Three NuRec-specific corrections live here, all measured over the 971 frames
where the human trajectory scores zero on no_at_fault_collisions.  The account
is ``docs/nurec/NC_HUMAN_FILTER.md``; in short:

* ``_interpolate_gt_observation`` no longer freezes a once-sampled track across
  the whole 4 s horizon.  That branch produced 85% of those 971 frames -- a car
  seen in one sample became a standing obstacle on road the ego was about to
  cover, and the ego drove into a vehicle the cameras show is not there.
* Boxes annotated on top of the ego's own body are dropped rather than held,
  and a box whose centre is inside the ego is dropped outright -- that is the
  ego re-detected as traffic, and it accounts for 15.5% of the surviving
  no_at_fault_collisions activations.
* ``compute_and_save_metric_cache`` skips clips whose annotations are known bad.

The ego footprint itself comes from the clip's own recording rig, not Pacifica;
that part lives in ``NavSimScenario._vehicle_parameters_from_scene``.

Applying all of this leaves 9.2% of the original no_at_fault_collisions
activations.  **Labels built before these changes disagree with labels built
after**, so the whole set has to be regenerated together.
"""

import logging
import pathlib
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from shapely.geometry import LineString
from nuplan.common.actor_state.agent import Agent
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D
from nuplan.common.actor_state.static_object import StaticObject
from nuplan.common.actor_state.tracked_objects import TrackedObjects
from nuplan.common.actor_state.tracked_objects_types import AGENT_TYPES
from nuplan.common.geometry.convert import absolute_to_relative_poses
from nuplan.common.maps.abstract_map_objects import LaneGraphEdgeMapObject, RoadBlockGraphEdgeMapObject
from nuplan.common.maps.maps_datatypes import SemanticMapLayer, TrafficLightStatusData
from nuplan.planning.scenario_builder.abstract_scenario import AbstractScenario
from nuplan.planning.simulation.history.simulation_history_buffer import SimulationHistoryBuffer
from nuplan.planning.simulation.observation.idm.utils import is_track_stopped
from nuplan.planning.simulation.observation.observation_type import DetectionsTracks
from nuplan.planning.simulation.planner.abstract_planner import PlannerInitialization, PlannerInput
from nuplan.planning.simulation.simulation_time_controller.simulation_iteration import SimulationIteration
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.training.experiments.cache_metadata_entry import CacheMetadataEntry

from navsim.common.dataclasses import Trajectory
from navsim.planning.data.nurec_excluded_clips import is_excluded
from navsim.common.enums import SceneFrameType
from navsim.planning.metric_caching.metric_cache import MapParameters, MetricCache
from navsim.planning.metric_caching.metric_caching_utils import StateInterpolator
from navsim.planning.scenario_builder.navsim_scenario import NavSimScenario
from navsim.planning.simulation.planner.pdm_planner.observation.pdm_observation import PDMObservation
from navsim.planning.simulation.planner.pdm_planner.pdm_closed_planner import PDMClosedPlanner
from navsim.planning.simulation.planner.pdm_planner.proposal.batch_idm_policy import BatchIDMPolicy


logger = logging.getLogger(__name__)

VEHICLE_SHRINK_FACTOR = 0.02
"""AlpaSim's ``vehicle_shrink_factor``, taken off both x and y.

It applies to every body in the scene, ego and actors alike, so that a
millimetre of annotation noise on either side does not read as contact.  The ego
side is handled where its parameters are built
(``NavSimScenario._vehicle_parameters_from_scene``); this constant is the actor
side, which used to keep its raw dimensions and so was scored against a
narrower ego than AlpaSim uses.
"""

ROAD_EDGE_ELEVATION_GATE_M = 3.0  # [m]
"""Height difference above which a road edge belongs to another road level.

AlpaSim uses this to keep a flyover's edge from marking the road beneath it as
off-road.  It is not a lateral margin -- the lateral test is 1 mm.
"""

SINGLE_SAMPLE_HOLD_RANGE_M = 30.0  # [m]
"""How near a once-sampled stopped track must be for the hold to be believable.

The speed threshold below is what decides whether such a track is treated as
parked, and its margin is thin: it was set at 4e-02 against an observed 0.0403,
so a clip reporting 0.039 walks through.  Range is the sturdier half of the
test.  Of the 278 single-sample stopped tracks measured, the median sits 74.6 m
from the ego and only 20.1% inside 50 m; a genuinely parked car that close is
seen again as the ego closes on it, and so never reaches this branch at all.
Beyond this range the box stands only for the instant it was seen.
"""

SINGLE_SAMPLE_STOPPED_SPEED = 4e-02  # [m/s]
"""Speed under which a once-sampled track is held for the whole horizon.

Below nuPlan's own 5e-02 because that value admitted a class of NuRec artefact:
a box detected in a single frame, far out and often metres off the ground,
carrying an annotated speed just under the threshold. Held for four seconds it
becomes a standing obstacle on road the ego is about to cover. The annotated
speed itself is sound -- against measured displacement it is within 0.05-0.11 m/s
at every range -- but a box seen once cannot be checked that way at all, so the
margin is deliberately tight. Tracks seen twice or more never reach this branch.
"""


def _shrink_detection_tracks(tracks: List[DetectionsTracks]) -> List[DetectionsTracks]:
    """Rebuilds every actor at AlpaSim's evaluation size.

    Both bodies lose 2% of length and width before AlpaSim tests them, and we
    already do that for the ego.  Leaving the actors at full size scored the ego
    against traffic 2% larger than the evaluator sees, which shows up as
    collisions that AlpaSim does not record.  Heading, velocity and metadata are
    untouched -- only the box shrinks, about its own centre.
    """
    scale = 1.0 - VEHICLE_SHRINK_FACTOR
    shrunk_tracks: List[DetectionsTracks] = []
    for detections in tracks:
        objects = []
        for tracked_object in detections.tracked_objects:
            box = tracked_object.box
            oriented_box = OrientedBox(
                box.center, box.length * scale, box.width * scale, box.height
            )
            if isinstance(tracked_object, Agent):
                objects.append(
                    Agent(
                        tracked_object_type=tracked_object.tracked_object_type,
                        oriented_box=oriented_box,
                        velocity=tracked_object.velocity,
                        metadata=tracked_object.metadata,
                    )
                )
            else:
                objects.append(
                    StaticObject(
                        tracked_object_type=tracked_object.tracked_object_type,
                        oriented_box=oriented_box,
                        metadata=tracked_object.metadata,
                    )
                )
        shrunk_tracks.append(DetectionsTracks(TrackedObjects(objects)))
    return shrunk_tracks


class MetricCacheProcessor:
    """Class for creating metric cache in NAVSIM."""

    def __init__(
        self,
        cache_path: Optional[str],
        force_feature_computation: bool,
        proposal_sampling: TrajectorySampling,
    ):
        """
        Initialize class.
        :param cache_path: Whether to cache features.
        :param force_feature_computation: If true, even if cache exists, it will be overwritten.
        """
        self._cache_path = pathlib.Path(cache_path) if cache_path else None
        self._force_feature_computation = force_feature_computation

        # 1s additional observation for ttc metric
        future_poses = proposal_sampling.num_poses + int(1.0 / proposal_sampling.interval_length)
        future_sampling = TrajectorySampling(num_poses=future_poses, interval_length=proposal_sampling.interval_length)
        self._proposal_sampling = proposal_sampling
        # NuRec runs to motorway speed: over the 4 s horizon the ego covers a
        # median 37 m but 10.7% of frames pass 100 m, out to 141 m.  At the old
        # 100 m the far half of those horizons had no map at all, so drivable
        # area quietly passed and the centerline search ran on a truncated
        # route.  200 m covers every measured frame with margin, and costs
        # little -- the proximal query goes from 11.8 ms to 13.9 ms.
        self._map_radius = 200

        self._pdm_closed = PDMClosedPlanner(
            trajectory_sampling=future_sampling,
            proposal_sampling=self._proposal_sampling,
            idm_policies=BatchIDMPolicy(
                speed_limit_fraction=[0.2, 0.4, 0.6, 0.8, 1.0],
                fallback_target_velocity=15.0,
                min_gap_to_lead_agent=1.0,
                headway_time=1.5,
                accel_max=1.5,
                decel_max=3.0,
            ),
            lateral_offsets=[-1.0, 1.0],
            map_radius=self._map_radius,
        )
        # The reference trajectory this planner produces is ranked by the same
        # scorer, so lane keeping and driving direction would be paid for on
        # every cached frame too -- four fifths of the scoring time, for terms
        # the PDM score no longer uses.
        self._pdm_closed._scorer.set_score_unused_metrics(False)

    def _get_planner_inputs(self, scenario: AbstractScenario) -> Tuple[PlannerInput, PlannerInitialization]:
        """
        Creates planner input arguments from scenario object.
        :param scenario: scenario object of nuPlan
        :return: tuple of planner input and initialization objects
        """

        # Initialize Planner
        planner_initialization = PlannerInitialization(
            route_roadblock_ids=scenario.get_route_roadblock_ids(),
            mission_goal=scenario.get_mission_goal(),
            map_api=scenario.map_api,
        )

        history = SimulationHistoryBuffer.initialize_from_list(
            buffer_size=1,
            ego_states=[scenario.initial_ego_state],
            observations=[scenario.initial_tracked_objects],
        )

        planner_input = PlannerInput(
            iteration=SimulationIteration(index=0, time_point=scenario.start_time),
            history=history,
            traffic_light_data=list(scenario.get_traffic_light_status_at_iteration(0)),
        )

        return planner_input, planner_initialization

    def _interpolate_gt_observation(self, scenario: NavSimScenario) -> List[DetectionsTracks]:
        """
        Helper function to interpolate detections tracks to higher temporal resolution.
        :param scenario: scenario interface of nuPlan framework
        :return: interpolated detection tracks
        """

        # TODO: add to config
        state_size = 6  # (time, x, y, heading, velo_x, velo_y)

        time_horizon = self._proposal_sampling.time_horizon  # [s]
        resolution_step = 0.5  # [s]
        interpolate_step = self._proposal_sampling.interval_length  # [s]

        scenario_step = scenario.database_interval  # [s]

        # sample detection tracks a 2Hz
        relative_time_s = np.arange(0, (time_horizon * 1 / resolution_step) + 1, 1, dtype=float) * resolution_step

        gt_indices = np.arange(
            0,
            int(time_horizon / scenario_step) + 1,
            int(resolution_step / scenario_step),
        )
        gt_detection_tracks = [
            scenario.get_tracked_objects_at_iteration(iteration=iteration) for iteration in gt_indices
        ]
        gt_ego_states = [scenario.get_ego_state_at_iteration(iteration=iteration) for iteration in gt_indices]

        # Boxes annotated where the ego's own body is. A car cannot be inside the
        # ego, so such a detection is wrong at the sample itself, not merely
        # unknown between samples -- the support window below cannot help. They
        # are dropped outright, but only for tracks seen once: a properly
        # tracked vehicle that really does hit the ego must still be scored.
        ego_overlapping_tokens = set()
        # A track whose *centre* sits inside the ego body is the ego itself,
        # re-detected.  Measured across all 1,603 trainval logs this fires on 18
        # tracks in 14 clips (83 of 65,671 frames).  Every one of them appears
        # out of nothing already covering ~85% of the ego footprint and then
        # travels at the ego's own speed -- it never approaches, which is what a
        # real collision does.  Overlap alone is too loose to drop a
        # multi-sample track, but centre containment is not: two distinct
        # vehicles cannot share a centre.
        ego_containing_tokens = set()
        for ego_state, detection_track in zip(gt_ego_states, gt_detection_tracks):
            ego_polygon = ego_state.car_footprint.oriented_box.geometry
            for tracked_object in detection_track.tracked_objects:
                if tracked_object.box.geometry.intersects(ego_polygon):
                    ego_overlapping_tokens.add(tracked_object.track_token)
                if ego_polygon.contains(tracked_object.box.geometry.centroid):
                    ego_containing_tokens.add(tracked_object.track_token)

        ego_positions = np.array(
            [[state.center.x, state.center.y] for state in gt_ego_states], dtype=np.float64
        )

        detection_tracks_states: Dict[str, Any] = {}
        unique_detection_tracks: Dict[str, Any] = {}

        for time_s, detection_track in zip(relative_time_s, gt_detection_tracks):

            for tracked_object in detection_track.tracked_objects:
                # log detection track
                token = tracked_object.track_token
                if token in ego_containing_tokens:
                    continue

                # extract states for dynamic and static objects
                tracked_state = np.zeros(state_size, dtype=np.float64)
                tracked_state[:4] = (
                    time_s,
                    tracked_object.center.x,
                    tracked_object.center.y,
                    tracked_object.center.heading,
                )

                if tracked_object.tracked_object_type in AGENT_TYPES:
                    # extract additional states for dynamic objects
                    tracked_state[4:] = (
                        tracked_object.velocity.x,
                        tracked_object.velocity.y,
                    )

                # found new object
                if token not in detection_tracks_states.keys():
                    detection_tracks_states[token] = [tracked_state]
                    unique_detection_tracks[token] = tracked_object

                # object already existed
                else:
                    detection_tracks_states[token].append(tracked_state)

        # create time interpolators
        detection_interpolators: Dict[str, StateInterpolator] = {}
        for token, states_list in detection_tracks_states.items():
            states = np.array(states_list, dtype=np.float64)
            detection_interpolators[token] = StateInterpolator(states)

        # interpolate at 10Hz
        interpolated_time_s = np.arange(0, int(time_horizon / interpolate_step) + 1, 1, dtype=float) * interpolate_step

        interpolated_detection_tracks = []
        for time_s in interpolated_time_s:
            interpolated_tracks = []
            for token, interpolator in detection_interpolators.items():
                initial_detection_track = unique_detection_tracks[token]
                interpolated_state = interpolator.interpolate(time_s)

                if interpolator.start_time == interpolator.end_time:
                    if token in ego_overlapping_tokens:
                        continue
                    # A track sampled once in the window says where the object
                    # was at that instant and nothing about any other instant.
                    # Holding the box over the whole horizon -- what this branch
                    # used to do -- asserts a standing obstacle on road the ego
                    # is about to cover, which is how 85% of the NuRec human
                    # trajectory's at-fault collisions arise. Keep it only over
                    # the interval its one sample stands for. A stopped object
                    # is still held throughout: standing still is exactly what
                    # its sample says, and dropping it would lose parked cars.
                    centre = initial_detection_track.center
                    ego_range = float(
                        np.min(
                            np.hypot(
                                ego_positions[:, 0] - centre.x, ego_positions[:, 1] - centre.y
                            )
                        )
                    )
                    holds_still = (
                        is_track_stopped(initial_detection_track, SINGLE_SAMPLE_STOPPED_SPEED)
                        and ego_range <= SINGLE_SAMPLE_HOLD_RANGE_M
                    )
                    if holds_still or abs(time_s - interpolator.start_time) <= resolution_step / 2.0:
                        interpolated_tracks.append(initial_detection_track)

                elif interpolated_state is not None:

                    tracked_type = initial_detection_track.tracked_object_type
                    metadata = initial_detection_track.metadata  # copied since time stamp is ignored

                    oriented_box = OrientedBox(
                        StateSE2(*interpolated_state[:3]),
                        initial_detection_track.box.length,
                        initial_detection_track.box.width,
                        initial_detection_track.box.height,
                    )

                    if tracked_type in AGENT_TYPES:
                        velocity = StateVector2D(*interpolated_state[3:])

                        detection_track = Agent(
                            tracked_object_type=tracked_type,
                            oriented_box=oriented_box,
                            velocity=velocity,
                            metadata=initial_detection_track.metadata,  # simply copy
                        )
                    else:
                        detection_track = StaticObject(
                            tracked_object_type=tracked_type,
                            oriented_box=oriented_box,
                            metadata=metadata,
                        )

                    interpolated_tracks.append(detection_track)
            interpolated_detection_tracks.append(DetectionsTracks(TrackedObjects(interpolated_tracks)))
        return interpolated_detection_tracks

    def _build_pdm_observation(
        self,
        interpolated_detection_tracks: List[DetectionsTracks],
        interpolated_traffic_light_data: List[List[TrafficLightStatusData]],
        route_lane_dict: Dict[str, LaneGraphEdgeMapObject],
    ):
        # convert to pdm observation
        pdm_observation = PDMObservation(
            self._proposal_sampling,
            self._proposal_sampling,
            self._map_radius,
            observation_sample_res=1,
            extend_observation_for_ttc=False,
        )
        pdm_observation.update_detections_tracks(
            interpolated_detection_tracks,
            interpolated_traffic_light_data,
            route_lane_dict,
            compute_traffic_light_data=True,
        )
        return pdm_observation

    def _interpolate_traffic_light_status(self, scenario: NavSimScenario) -> List[List[TrafficLightStatusData]]:

        time_horizon = self._proposal_sampling.time_horizon  # [s]
        interpolate_step = self._proposal_sampling.interval_length  # [s]

        scenario_step = scenario.database_interval  # [s]
        gt_indices = np.arange(0, int(time_horizon / scenario_step) + 1, 1, dtype=int)

        traffic_light_status = []
        for iteration in gt_indices:
            current_status_list = list(scenario.get_traffic_light_status_at_iteration(iteration=iteration))
            for _ in range(int(scenario_step / interpolate_step)):
                traffic_light_status.append(current_status_list)

        if scenario_step == interpolate_step:
            return traffic_light_status
        else:
            return traffic_light_status[: -int(scenario_step / interpolate_step) + 1]

    def _load_route_dicts(
        self, scenario: NavSimScenario, route_roadblock_ids: List[str]
    ) -> Tuple[Dict[str, RoadBlockGraphEdgeMapObject], Dict[str, LaneGraphEdgeMapObject]]:
        route_roadblock_ids = list(dict.fromkeys(route_roadblock_ids))

        route_roadblock_dict = {}
        route_lane_dict = {}

        for id_ in route_roadblock_ids:
            block = scenario.map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK)
            block = block or scenario.map_api.get_map_object(id_, SemanticMapLayer.ROADBLOCK_CONNECTOR)

            route_roadblock_dict[block.id] = block

            for lane in block.interior_edges:
                route_lane_dict[lane.id] = lane

        return route_roadblock_dict, route_lane_dict

    def _build_file_path(self, scenario: NavSimScenario) -> pathlib.Path:
        return (
            (self._cache_path / scenario.log_name / scenario.scenario_type / scenario.token / "metric_cache.pkl")
            if self._cache_path
            else None
        )

    def _road_edges_for(self, scenario: NavSimScenario) -> Dict[str, Any]:
        """Road-edge lines near the ego, for the AlpaSim offroad rule.

        AlpaSim never asks whether the ego is inside a drivable area; it asks
        whether the body reached the edge of the road.  That question needs the
        boundary lines, which no NAVSIM map publishes -- so this returns nothing
        for those and the nuPlan containment rule stands.

        The lines do not close into rings: of 298,939 across the NuRec release,
        11 have coincident endpoints and the median gap is 16.6 m.  Nothing here
        tries to close them, which is the point -- a distance test needs no ring.
        """
        map_api = scenario.map_api
        boundaries = map_api.get_proximal_map_objects(
            scenario.initial_ego_state.center.point,
            self._map_radius,
            [SemanticMapLayer.BOUNDARIES],
        ).get(SemanticMapLayer.BOUNDARIES, [])
        if not boundaries:
            return {}

        # The flyover gate is applied here rather than at scoring time: the ego
        # covers at most a few tens of metres in the horizon, over which road
        # height barely moves, so one filter at the frame's own position stands
        # in for a per-pose one and keeps the scorer free of elevation data.
        ego_elevation = self._road_surface_elevation(scenario)
        lines, elevations = [], []
        for boundary in boundaries:
            polyline = boundary.baseline_path
            if polyline.linestring.is_empty:
                continue
            if abs(polyline.elevation - ego_elevation) > ROAD_EDGE_ELEVATION_GATE_M:
                continue
            lines.append(polyline.linestring)
            elevations.append(polyline.elevation)
        if not lines:
            return {}
        return {
            "road_edges": lines,
            "road_edge_elevations": np.asarray(elevations, dtype=np.float64),
        }

    @staticmethod
    def _road_surface_elevation(scenario: NavSimScenario) -> float:
        """Height of the road under the ego, read off the nearest lane centre.

        ``EgoState`` is planar, and the lanes and boundaries come out of one
        ClipGT extraction, so the lane graph is the available vertical frame.
        """
        position = scenario.initial_ego_state.center.point
        lanes = scenario.map_api.get_proximal_map_objects(
            position, 25.0, [SemanticMapLayer.LANE]
        ).get(SemanticMapLayer.LANE, [])
        elevations = [
            lane.baseline_path.elevation
            for lane in lanes
            if getattr(lane.baseline_path, "elevation", None) is not None
        ]
        return float(np.median(elevations)) if elevations else 0.0

    def compute_and_save_metric_cache(self, scenario: NavSimScenario) -> Optional[CacheMetadataEntry]:
        # Clips whose annotations place vehicles where the cameras show empty
        # road. Nothing in the box geometry marks them, so they are dropped
        # whole -- see nurec_excluded_clips and docs/nurec/NC_HUMAN_FILTER.md
        # section 9. Returning None makes the caller count it as a failure,
        # which is honest: no cache is written for these tokens.
        if is_excluded(scenario.log_name):
            logger.info(f"Skipping excluded NuRec clip {scenario.log_name}")
            return None

        file_name = self._build_file_path(scenario)
        assert file_name is not None, "Cache path can not be None for saving cache."
        if file_name.exists() and not self._force_feature_computation:
            return CacheMetadataEntry(file_name)
        metric_cache = self.compute_metric_cache(scenario)
        metric_cache.dump()
        return CacheMetadataEntry(metric_cache.file_path)

    def _extract_ego_future_trajectory(self, scenario: NavSimScenario) -> Trajectory:
        ego_trajectory_sampling = TrajectorySampling(
            time_horizon=self._proposal_sampling.time_horizon,
            interval_length=scenario.database_interval,
        )
        future_ego_states = list(
            scenario.get_ego_future_trajectory(
                iteration=0,
                time_horizon=ego_trajectory_sampling.time_horizon,
                num_samples=ego_trajectory_sampling.num_poses,
            )
        )
        initial_ego_state = scenario.get_ego_state_at_iteration(0)
        if future_ego_states[0].time_point != initial_ego_state.time_point:
            # nuPlan does not return the initial state while navsim does
            # make sure to add the initial state before transforming to relative poses
            future_ego_states = [initial_ego_state] + future_ego_states

        future_ego_poses = [state.rear_axle for state in future_ego_states]
        relative_future_states = absolute_to_relative_poses(future_ego_poses)[1:]
        return Trajectory(
            poses=np.array([[pose.x, pose.y, pose.heading] for pose in relative_future_states]),
            trajectory_sampling=ego_trajectory_sampling,
        )

    def compute_metric_cache(self, scenario: NavSimScenario) -> MetricCache:
        file_name = self._build_file_path(scenario)

        # TODO: we should infer this from the scene metadata
        is_synthetic_scene = len(scenario.token) == 17

        # init and run PDM-Closed
        planner_input, planner_initialization = self._get_planner_inputs(scenario)
        self._pdm_closed.initialize(planner_initialization)
        pdm_closed_trajectory = self._pdm_closed.compute_planner_trajectory(planner_input)

        route_roadblock_dict, route_lane_dict = self._load_route_dicts(
            scenario, planner_initialization.route_roadblock_ids
        )

        interpolated_detection_tracks = _shrink_detection_tracks(
            self._interpolate_gt_observation(scenario)
        )
        interpolated_traffic_light_status = self._interpolate_traffic_light_status(scenario)

        observation = self._build_pdm_observation(
            interpolated_detection_tracks=interpolated_detection_tracks,
            interpolated_traffic_light_data=interpolated_traffic_light_status,
            route_lane_dict=route_lane_dict,
        )
        future_tracked_objects = interpolated_detection_tracks[1:]

        past_human_trajectory = InterpolatedTrajectory(
            [ego_state for ego_state in scenario.get_ego_past_trajectory(0, 1.5)]
        )

        if not is_synthetic_scene:
            human_trajectory = self._extract_ego_future_trajectory(scenario)
        else:
            human_trajectory = None

        # save and dump features
        return MetricCache(
            file_path=file_name,
            log_name=scenario.log_name,
            scene_type=SceneFrameType.SYNTHETIC if is_synthetic_scene else SceneFrameType.ORIGINAL,
            timepoint=scenario.start_time,
            trajectory=pdm_closed_trajectory,
            human_trajectory=human_trajectory,
            past_human_trajectory=past_human_trajectory,
            ego_state=scenario.initial_ego_state,
            observation=observation,
            centerline=self._pdm_closed._centerline,
            route_lane_ids=list(self._pdm_closed._route_lane_dict.keys()),
            drivable_area_map=self._pdm_closed._drivable_area_map,
            past_detections_tracks=[
                dt for dt in scenario.get_past_tracked_objects(iteration=0, time_horizon=1.5, num_samples=3)
            ][:-1],
            current_tracked_objects=[scenario.initial_tracked_objects],
            future_tracked_objects=future_tracked_objects,
            map_parameters=MapParameters(
                map_root=scenario.map_root,
                map_version=scenario.map_version,
                map_name=scenario.map_api.map_name,
            ),
            **self._road_edges_for(scenario),
        )
