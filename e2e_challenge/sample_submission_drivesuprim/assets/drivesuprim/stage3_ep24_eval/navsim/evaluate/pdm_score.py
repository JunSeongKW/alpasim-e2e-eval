import logging
import time
from typing import Dict, List, Optional

import numpy as np
from shapely.geometry import LineString
import numpy.typing as npt
import pandas as pd
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint
from nuplan.common.geometry.convert import relative_to_absolute_poses
from nuplan.planning.simulation.planner.ml_planner.transform_utils import (
    _get_fixed_timesteps,
    _se2_vel_acc_to_ego_state,
)
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import Trajectory
from navsim.common.enums import SceneFrameType
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_state_to_state_array,
    ego_states_to_state_array,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex,
    WeightedMetricIndex,
)

logger = logging.getLogger(__name__)


def _gt_path(metric_cache, initial_ego_state) -> Optional[LineString]:
    """The recorded run in world coordinates, for drift and progress.

    NuRec scenes carry one; NAVSIM scenes may not, and without it both terms
    stay at their nuPlan behaviour.
    """
    trajectory = getattr(metric_cache, "human_trajectory", None)
    poses = getattr(trajectory, "poses", None)
    if poses is None:
        return None
    relative = np.asarray(poses, dtype=np.float64)
    if relative.ndim != 2 or relative.shape[0] < 1 or relative.shape[1] < 2:
        return None
    origin = initial_ego_state.rear_axle
    cos, sin = np.cos(origin.heading), np.sin(origin.heading)
    world = np.stack(
        [
            origin.x + relative[:, 0] * cos - relative[:, 1] * sin,
            origin.y + relative[:, 0] * sin + relative[:, 1] * cos,
        ],
        axis=-1,
    )
    return LineString(np.vstack([[origin.x, origin.y], world]))


def aggregate_epdms_labels(gt: Dict[str, np.ndarray], scorer_config) -> np.ndarray:
    """Aggregates exported metric labels into the score training reads.

    This is NAVSIM's v1 PDM score, not v2 EPDMS.  The two differ in what they
    multiply and what they weigh:

        here  = NC x DAC x GT x EP
        EPDMS = NC x DAC x DDC x TL x (5 EP + 5 TTC + 2 LK + 2 HC + 2 2FC) / 16

    The terms EPDMS adds are the ones NuRec cannot support.  Lane keeping fires
    on 27.1% of frames because the ClipGT lane annotation sits about a metre off
    the driven line -- measured over 47,949 frames, the recorded human leaves the
    lane surface on 28% of timesteps.  Driving direction fires on 2.6%, and
    inspection of those frames found no wrong-way driving at all: the metric
    tests route membership rather than heading, and the frames it flags have the
    human aligned with its lane to within 2.1 degrees.  Traffic light is constant
    at 1.0 because NuRec publishes no signal state.  Between them they neutralised
    a quarter of every label while measuring map defects rather than driving.

    Neither has a counterpart in AlpaSim's scene score, which is what these
    labels are ultimately judged against, so dropping them moves the label toward
    the evaluator rather than away from it.  Both metrics are still computed and
    exported -- they remain useful for diagnosing the map -- they just no longer
    enter the score.

    Comfort was dropped for the same reason as the rest.  It applies jerk and
    yaw-rate limits to an MPC rollout of the recorded run, and the recorded run
    fails them on 8.0% of frames -- the controller's tracking ripple rather than
    the driving.  AlpaSim has no comfort term.

    Time-to-collision is out as well.  AlpaSim has no such term, and keeping it
    was the one deliberate departure from the evaluator; the prototype drops it
    so the label is exactly AlpaSim's scene score.  Two thirds of the frames it
    zeroed were already zeroed by collision, so it was contributing an
    independent signal on roughly 0.2% of frames.  It is still computed and
    exported for diagnosis.

    There is no drift term.  AlpaSim has none; the 4 m corridor enters through
    progress, which stops accruing once a body leaves it.  Carried separately it
    only fired the human penalty filter -- the trigger on 297 of 300 frames
    measured, since the recorded run cannot hold its own corridor either, and a
    fired filter overwrites every term of that frame with ones.
    """
    terms = [gt["no_at_fault_collisions"], gt["drivable_area_compliance"]]
    multiplicative = np.prod(np.stack(terms).astype(np.float64), axis=0)

    # Progress is the only weighted term left, so the weighted sum is progress
    # itself.  scorer_config.progress_weight is read anyway to keep the failure
    # loud if the config ever stops carrying it.
    assert scorer_config.progress_weight > 0.0
    score = multiplicative * np.asarray(gt["ego_progress"], dtype=np.float64)
    return score.astype(np.float16)


ALPASIM_SCORE_TERMS = ("no_at_fault_collisions", "drivable_area_compliance", "ego_progress")
"""Columns whose product is the scene score.  Same terms, same order, as
``aggregate_epdms_labels`` builds the training target from."""


ALPASIM_RAW_PREFIX = "raw_"
"""Prefix of the pre-human-filter snapshot of each term."""


def alpasim_scene_score(df, prefix: str = "") -> np.ndarray:
    """Scene score for evaluated rows: NC x DAC x GT x EP.

    Read from the per-metric *columns*, not from ``weighted_metrics``.  That is
    the opposite of what ``score``/``pdms`` do, and deliberately so: the human
    penalty filter rewrites the scalar columns to 1.0 and leaves the arrays
    alone, and the training labels are built from filtered per-metric values.
    Aggregating from the arrays here would score the model against a target it
    was never trained on -- on exactly the frames where the recorded human
    itself fails.

    A row missing any term (a scenario that raised, so ``get_empty_results``)
    yields NaN and is skipped by the mean, same as every other column.
    """
    terms = []
    for name in ALPASIM_SCORE_TERMS:
        column = f"{prefix}{name}"
        if column not in df.columns:
            raise KeyError(f"cannot aggregate the scene score without column {column!r}")
        terms.append(np.asarray(df[column], dtype=np.float64))
    return np.prod(np.stack(terms), axis=0)


def transform_trajectory(pred_trajectory: Trajectory, initial_ego_state: EgoState) -> InterpolatedTrajectory:
    """
    Transform trajectory in global frame and return as InterpolatedTrajectory
    :param pred_trajectory: trajectory dataclass in ego frame
    :param initial_ego_state: nuPlan's ego state object
    :return: nuPlan's InterpolatedTrajectory
    """

    future_sampling = pred_trajectory.trajectory_sampling
    timesteps = _get_fixed_timesteps(initial_ego_state, future_sampling.time_horizon, future_sampling.interval_length)

    relative_poses = np.array(pred_trajectory.poses, dtype=np.float64)
    relative_states = [StateSE2.deserialize(pose) for pose in relative_poses]
    absolute_states = relative_to_absolute_poses(initial_ego_state.rear_axle, relative_states)

    # NOTE: velocity and acceleration ignored by LQR + bicycle model
    agent_states = [
        _se2_vel_acc_to_ego_state(
            state,
            [0.0, 0.0],
            [0.0, 0.0],
            timestep,
            initial_ego_state.car_footprint.vehicle_parameters,
        )
        for state, timestep in zip(absolute_states, timesteps)
    ]

    # NOTE: maybe make addition of initial_ego_state optional
    return InterpolatedTrajectory([initial_ego_state] + agent_states)


def get_trajectory_as_array(
    trajectory: InterpolatedTrajectory,
    future_sampling: TrajectorySampling,
    start_time: TimePoint,
) -> npt.NDArray[np.float64]:
    """
    Interpolated trajectory and return as numpy array
    :param trajectory: nuPlan's InterpolatedTrajectory object
    :param future_sampling: Sampling parameters for interpolation
    :param start_time: TimePoint object of start
    :return: Array of interpolated trajectory states.
    """

    times_s = np.arange(
        0.0,
        future_sampling.time_horizon + future_sampling.interval_length,
        future_sampling.interval_length,
    )
    times_s += start_time.time_s
    times_us = [int(time_s * 1e6) for time_s in times_s]
    times_us = np.clip(times_us, trajectory.start_time.time_us, trajectory.end_time.time_us)
    time_points = [TimePoint(time_us) for time_us in times_us]

    trajectory_ego_states: List[EgoState] = trajectory.get_state_at_times(time_points)

    return ego_states_to_state_array(trajectory_ego_states)


def pdm_score(
    metric_cache: MetricCache,
    model_trajectory: Trajectory,
    future_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    traffic_agents_policy: AbstractTrafficAgentsPolicy,
) -> pd.DataFrame:
    """
    FIXME: Output type hints and refactoring/debugging. Inconsistent with some evaluation scripts.
    Runs PDM-Score and saves results in the corresponding dataclass.
    :param metric_cache: Metric cache dataclass of the sample.
    :param model_trajectory: Predicted trajectory in ego frame.
    :param future_sampling: Sampling configuration of the model trajectory.
    :param simulator: Simulator applied on the model trajectory.
    :param scorer: Scoring object to retrieve the sub-scores
    :param traffic_agents_policy: background traffic used during simulation/scoring.
    :return: Dataclass of PDM sub-scores.
    """

    pred_trajectory = transform_trajectory(model_trajectory, metric_cache.ego_state)

    return pdm_score_from_interpolated_trajectory(
        metric_cache=metric_cache,
        pred_trajectory=pred_trajectory,
        future_sampling=future_sampling,
        simulator=simulator,
        scorer=scorer,
        traffic_agents_policy=traffic_agents_policy,
    )


def pdm_score_from_interpolated_trajectory(
    metric_cache: MetricCache,
    pred_trajectory: InterpolatedTrajectory,
    future_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    traffic_agents_policy: AbstractTrafficAgentsPolicy,
):
    """
    FIXME: Output type hints and refactoring/debugging. Inconsistent with some evaluation scripts.
    Computes PDM-Score from interpolated trajectory of an agent.
    :param metric_cache: Metric cache dataclass of the sample.
    :param pred_trajectory: Predicted (interpolated) trajectory in global frame.
    :param future_sampling: Sampling configuration of the trajectory.
    :param simulator: Simulator applied on the trajectory.
    :param scorer: Scoring object to retrieve the sub-scores.
    :param traffic_agents_policy: background traffic used during simulation/scoring.
    :return: Dataclass of PDM sub-scores.
    """

    initial_ego_state = metric_cache.ego_state
    # Score against the body that recorded the clip, not Pacifica.
    scorer.set_vehicle_parameters(initial_ego_state.car_footprint.vehicle_parameters)
    scorer.set_road_edges(getattr(metric_cache, "road_edges", None))
    scorer.set_road_levels(
        getattr(metric_cache, "road_edge_profiles", None),
        getattr(metric_cache, "road_surfaces", None),
        getattr(metric_cache, "road_surface_profiles", None),
    )
    # A scene carrying road edges is a NuRec scene, and NuRec labels are
    # meant to agree with AlpaSim, so at-fault collision follows its bumper
    # rule there.  NAVSIM scenes publish no boundaries and keep nuPlan's,
    # which is what their benchmark numbers mean.
    scorer.set_alpasim_collision(getattr(metric_cache, "road_edges", None) is not None)
    scorer.set_gt_path(_gt_path(metric_cache, initial_ego_state))
    # Lane keeping and driving direction are four fifths of scoring time and no
    # longer enter the score; see PDMScorer.set_score_unused_metrics.  Off by
    # default, on when a run asks for the old weighted EPDMS as well.
    score_unused = bool(getattr(scorer._config, "score_unused_metrics", False))
    scorer.set_score_unused_metrics(score_unused)
    pdm_trajectory = metric_cache.trajectory

    pdm_states, pred_states = (
        get_trajectory_as_array(pdm_trajectory, future_sampling, initial_ego_state.time_point),
        get_trajectory_as_array(pred_trajectory, future_sampling, initial_ego_state.time_point),
    )
    trajectory_states = np.concatenate([pdm_states[None, ...], pred_states[None, ...]], axis=0)

    simulated_states = simulator.simulate_proposals(trajectory_states, initial_ego_state)

    # infer traffic agents policy and update future observation
    simulated_agent_detections_tracks = traffic_agents_policy.simulate_environment(simulated_states[1], metric_cache)

    assert (
        len(simulated_agent_detections_tracks) == trajectory_states.shape[1]
    ), f"""
            Traffic agents policy returned trajectories of invalid length:
            Traffic agents trajectories must be of length ego_trajectory_length = {trajectory_states.shape[1]},
            but got {len(simulated_agent_detections_tracks)}
        """

    pred_idx = 1  # index of predicted trajectory in trajectory_states and simulated_states
    pdm_result = scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        metric_cache.map_parameters,
        simulated_agent_detections_tracks,
        metric_cache.past_human_trajectory,
    )[pred_idx]

    # Snapshot the four terms as measured, before the human penalty filter can
    # rewrite any of them to 1.0.  Both readings are wanted and they answer
    # different questions: the filtered one is the target the model was trained
    # against, the raw one is what AlpaSim -- which has no such filter -- would
    # hand this trajectory.
    for _term in ALPASIM_SCORE_TERMS:
        pdm_result[f"{ALPASIM_RAW_PREFIX}{_term}"] = pdm_result[_term].iloc[0]

    if scorer._config.human_penalty_filter and metric_cache.scene_type == SceneFrameType.ORIGINAL:
        # human_penalty_filter

        human_states = human_states_to_simulate(metric_cache, future_sampling)

        human_simulated_states = simulator.simulate_proposals(human_states, initial_ego_state)

        human_simulated_agent_detections_tracks = traffic_agents_policy.simulate_environment(
            human_simulated_states[0], metric_cache
        )

        human_pdm_result = scorer.score_proposals(
            human_simulated_states,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
            metric_cache.map_parameters,
            human_simulated_agent_detections_tracks,
        )[0]

        for column in human_pdm_result.columns:
            if column in [
                "multiplicative_metrics_prod",
                "weighted_metrics",
                "weighted_metrics_array",
                "v1_terms",
            ]:
                continue
            if human_pdm_result[column].iloc[0] == 0:
                pdm_result.at[0, column] = 1

    # A scene carrying road edges is a NuRec scene, and NuRec labels are
    # AlpaSim's scene score, so the checkpoint has to be judged by that same
    # score rather than by the weighted EPDMS sum.  Recorded here, per row,
    # because aggregation happens later in a function that only sees the frame.
    pdm_result["alpasim_mode"] = getattr(metric_cache, "road_edges", None) is not None

    # Whether the terms only the old weighted scores need were measured at all.
    # When they were not they sit at 1.0, and 1.0 there means "not computed",
    # not "passed" -- so the aggregation drops those scores rather than report a
    # number built from held constants.
    pdm_result["unused_metrics_scored"] = score_unused

    return pdm_result, simulated_states[pred_idx]

def vocab_to_state_arrays(
    vocab_trajectories: npt.NDArray[np.float64],
    initial_ego_state: EgoState,
) -> npt.NDArray[np.float64]:
    """Vocabulary poses as simulator input arrays, without the object round-trip.

    Building one ``InterpolatedTrajectory`` of ``EgoState`` objects per candidate
    and sampling it straight back onto the grid it was built from costs about a
    third of the scoring time -- roughly 170k short-lived EgoStates per token --
    and every field it fills beyond the pose is zero, because the poses carry no
    velocity or acceleration. The transform is therefore done directly: rotate
    the ego-frame poses into the global frame and prepend the real initial state.

    :param vocab_trajectories: ``[K, num_poses, 3]`` ego-frame (x, y, heading)
    :return: ``[K, num_poses + 1, StateIndex.size()]``
    """
    poses = np.asarray(vocab_trajectories, dtype=np.float64)
    count, num_poses, _ = poses.shape

    origin = initial_ego_state.rear_axle
    cos_h, sin_h = np.cos(origin.heading), np.sin(origin.heading)

    states = np.zeros((count, num_poses + 1, StateIndex.size()), dtype=np.float64)
    states[:, 0] = ego_state_to_state_array(initial_ego_state)
    states[:, 1:, StateIndex.X] = origin.x + poses[..., 0] * cos_h - poses[..., 1] * sin_h
    states[:, 1:, StateIndex.Y] = origin.y + poses[..., 0] * sin_h + poses[..., 1] * cos_h
    states[:, 1:, StateIndex.HEADING] = poses[..., 2] + origin.heading
    return states


def proposal_states_to_simulate(
    metric_cache: MetricCache,
    vocab_trajectories: Trajectory,
    future_sampling: TrajectorySampling,
) -> npt.NDArray[np.float64]:
    """The desired states ``pdm_score_full_v2`` rolls out for one token.

    Exposed so a caller can build several tokens' worth and simulate them in one
    call -- one token's proposals do not fill a modern GPU, and batching tokens
    is worth several-fold. Keeping the construction here means the batched path
    and the plain path cannot drift apart.

    :return: ``[1 + len(vocab), num_poses + 1, StateIndex.size()]``, the PDM
        trajectory first and the vocabulary after it
    """
    initial_ego_state = metric_cache.ego_state
    pdm_states = get_trajectory_as_array(
        metric_cache.trajectory,
        future_sampling,
        initial_ego_state.time_point
    )[None]
    # pdm, vocab-0, vocab-1, ..., vocab-n
    return np.concatenate(
        [pdm_states, vocab_to_state_arrays(vocab_trajectories, initial_ego_state)], axis=0
    )


def human_states_to_simulate(
    metric_cache: MetricCache,
    future_sampling: TrajectorySampling,
) -> npt.NDArray[np.float64]:
    """The human trajectory the penalty filter rolls out, shaped ``[1, T, T]``."""
    initial_ego_state = metric_cache.ego_state
    human_trajectory = transform_trajectory(metric_cache.human_trajectory, initial_ego_state)
    return get_trajectory_as_array(
        human_trajectory, future_sampling, initial_ego_state.time_point
    )[None, ...]


def pdm_score_full_v2(
    metric_cache: MetricCache,
    vocab_trajectories: Trajectory,
    future_sampling: TrajectorySampling,
    simulator: PDMSimulator,
    scorer: PDMScorer,
    traffic_agents_policy: AbstractTrafficAgentsPolicy,
):
    start = time.time()
    initial_ego_state = metric_cache.ego_state
    # Score against the body that recorded the clip, not Pacifica.
    scorer.set_vehicle_parameters(initial_ego_state.car_footprint.vehicle_parameters)
    scorer.set_road_edges(getattr(metric_cache, "road_edges", None))
    scorer.set_road_levels(
        getattr(metric_cache, "road_edge_profiles", None),
        getattr(metric_cache, "road_surfaces", None),
        getattr(metric_cache, "road_surface_profiles", None),
    )
    # A scene carrying road edges is a NuRec scene, and NuRec labels are
    # meant to agree with AlpaSim, so at-fault collision follows its bumper
    # rule there.  NAVSIM scenes publish no boundaries and keep nuPlan's,
    # which is what their benchmark numbers mean.
    scorer.set_alpasim_collision(getattr(metric_cache, "road_edges", None) is not None)
    scorer.set_gt_path(_gt_path(metric_cache, initial_ego_state))
    # Lane keeping and driving direction are four fifths of scoring time and
    # no longer enter the score; see PDMScorer.set_score_unused_metrics.
    scorer.set_score_unused_metrics(bool(getattr(scorer._config, "score_unused_metrics", False)))
    all_states = proposal_states_to_simulate(metric_cache, vocab_trajectories, future_sampling)

    simulated_states = simulator.simulate_proposals(all_states, initial_ego_state)

    # infer traffic agents policy and update future observation
    simulated_agent_detections_tracks = traffic_agents_policy.simulate_environment(simulated_states[1], metric_cache)

    assert (
            len(simulated_agent_detections_tracks) == all_states.shape[1]
    ), f"""
                Traffic agents policy returned trajectories of invalid length:
                Traffic agents trajectories must be of length ego_trajectory_length = {all_states.shape[1]},
                but got {len(simulated_agent_detections_tracks)}
            """

    pdm_result, pdms = scorer.score_proposals(
        simulated_states,
        metric_cache.observation,
        metric_cache.centerline,
        metric_cache.route_lane_ids,
        metric_cache.drivable_area_map,
        metric_cache.map_parameters,
        simulated_agent_detections_tracks,
        metric_cache.past_human_trajectory,
        if_return_pdms=True
    )
    gt = {
        'no_at_fault_collisions': scorer._multi_metrics[MultiMetricIndex.NO_COLLISION].astype(np.float16)[1:],
        'drivable_area_compliance': scorer._multi_metrics[MultiMetricIndex.DRIVABLE_AREA].astype(np.float16)[1:],
        'driving_direction_compliance': scorer._multi_metrics[MultiMetricIndex.DRIVING_DIRECTION].astype(np.float16)[1:],
        'traffic_light_compliance': scorer._multi_metrics[MultiMetricIndex.TRAFFIC_LIGHT_COMPLIANCE].astype(np.float16)[1:],
        'comfort': np.asarray(scorer._comfort).astype(np.float16)[1:],
        'ego_progress': scorer._weighted_metrics[WeightedMetricIndex.PROGRESS].astype(np.float16)[1:],
        'time_to_collision_within_bound': scorer._weighted_metrics[WeightedMetricIndex.TTC].astype(np.float16)[1:],
        'lane_keeping': scorer._weighted_metrics[WeightedMetricIndex.LANE_KEEPING].astype(np.float16)[1:],
        'history_comfort': scorer._weighted_metrics[WeightedMetricIndex.HISTORY_COMFORT].astype(np.float16)[1:],
        'pdm_score': np.array(pdms).astype(np.float16)[1:]
    }

    if scorer._config.human_penalty_filter and metric_cache.scene_type == SceneFrameType.ORIGINAL:
        # human_penalty_filter

        human_trajectory = transform_trajectory(metric_cache.human_trajectory, initial_ego_state)

        human_states = get_trajectory_as_array(human_trajectory, future_sampling, initial_ego_state.time_point)

        human_simulated_states = simulator.simulate_proposals(human_states[None, ...], initial_ego_state)

        human_simulated_agent_detections_tracks = traffic_agents_policy.simulate_environment(
            human_simulated_states[0], metric_cache
        )

        human_pdm_result = scorer.score_proposals(
            human_simulated_states,
            metric_cache.observation,
            metric_cache.centerline,
            metric_cache.route_lane_ids,
            metric_cache.drivable_area_map,
            metric_cache.map_parameters,
            human_simulated_agent_detections_tracks,
        )[0]

        for column in human_pdm_result.columns:
            if column in [
                "multiplicative_metrics_prod",
                "weighted_metrics",
                "weighted_metrics_array",
                "v1_terms",
                "pdm_score",
            ]:
                continue
            if human_pdm_result[column].iloc[0] == 0:
                gt[column] = np.ones_like(gt[column])

    # The human filter runs after PDMScorer's original aggregation. Rebuild the
    # final score so it matches the neutralized components. This is essential
    # for NuRec where traffic-light compliance is intentionally fixed at one.
    gt["pdm_score"] = aggregate_epdms_labels(gt, scorer._config)

    logger.debug(f"pdm_score_full_v2 took {time.time() - start:.1f}s")
    return gt
