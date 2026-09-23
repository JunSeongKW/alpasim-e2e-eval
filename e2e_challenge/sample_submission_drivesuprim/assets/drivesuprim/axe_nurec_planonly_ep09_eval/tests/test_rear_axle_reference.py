"""The recorded run must score full marks against itself.

Every trajectory in this pipeline -- the 4,096-entry vocabulary, the cached
human trajectory, and the GT path built from it -- is referenced to the ego's
**rear axle**.  ``PDMScorer._ego_coords[..., CENTER]`` is not: the body centre
sits about 1.47 m ahead of the axle.  Measuring a rear-axle path with the body
centre once cost EP a 0.046 floor for a proposal that never moved, and charged
the recorded human run a median 1.39 m of drift against a 4 m budget it had not
touched.  Neither showed up in the human filter, because the human clamps to 1.0
at the end of its own path.

These tests fail if either term goes back to the body centre.
"""

import numpy as np
import pytest
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely.geometry import LineString

from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    GT_LATERAL_LIMIT_M,
    PDMScorer,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    state_array_to_coords_array,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import (
    MultiMetricIndex,
    StateIndex,
    WeightedMetricIndex,
)

NUM_POSES = 40
SAMPLING = TrajectorySampling(num_poses=NUM_POSES, interval_length=0.1)


def _arc_states(radius: float = 40.0, speed: float = 10.0) -> np.ndarray:
    """A curved rear-axle run.  A curve is what separates the two references:
    on a straight line the centre projects onto the same path."""
    steps = NUM_POSES + 1
    arc = speed * SAMPLING.interval_length * np.arange(steps) / radius
    states = np.zeros((1, steps, StateIndex.size()), dtype=np.float64)
    states[0, :, StateIndex.X] = radius * np.sin(arc)
    states[0, :, StateIndex.Y] = radius * (1.0 - np.cos(arc))
    states[0, :, StateIndex.HEADING] = arc
    states[0, :, StateIndex.VELOCITY_X] = speed
    return states


def _prime(scorer: PDMScorer, states: np.ndarray, gt_path: LineString) -> None:
    """Sets up just enough of the scorer to run the two GT-referenced terms."""
    scorer.set_vehicle_parameters(get_pacifica_parameters())
    scorer.set_gt_path(gt_path)
    scorer._num_proposals = states.shape[0]
    scorer._states = states
    scorer._ego_coords = state_array_to_coords_array(states, scorer._vehicle_parameters)
    scorer._multi_metrics = np.zeros((len(MultiMetricIndex), states.shape[0]), dtype=np.float64)
    scorer._weighted_metrics = np.zeros((len(WeightedMetricIndex), states.shape[0]), dtype=np.float64)
    scorer._contact_time_idcs = np.full(states.shape[0], np.inf, dtype=np.float64)


def _gt_path_of(states: np.ndarray) -> LineString:
    return LineString(states[0, :, [StateIndex.X, StateIndex.Y]].T)


def test_recorded_run_keeps_full_gt_compliance():
    states = _arc_states()
    scorer = PDMScorer(SAMPLING)
    _prime(scorer, states, _gt_path_of(states))

    scorer._calculate_gt_compliance()

    assert scorer._multi_metrics[MultiMetricIndex.GT_COMPLIANCE, 0] == 1.0


def test_recorded_run_scores_full_progress():
    states = _arc_states()
    scorer = PDMScorer(SAMPLING)
    _prime(scorer, states, _gt_path_of(states))

    scorer._calculate_progress()

    assert scorer._progress_raw[0] == pytest.approx(1.0, abs=1e-9)


def test_a_proposal_that_never_moves_makes_no_progress():
    states = _arc_states()
    gt_path = _gt_path_of(states)
    parked = np.repeat(states[:, :1], NUM_POSES + 1, axis=1)
    scorer = PDMScorer(SAMPLING)
    _prime(scorer, parked, gt_path)

    scorer._calculate_progress()

    assert scorer._progress_raw[0] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize(
    "offset, expected", [(0.0, 1.0), (3.5, 1.0), (GT_LATERAL_LIMIT_M - 0.1, 1.0), (GT_LATERAL_LIMIT_M + 0.1, 0.0)]
)
def test_gt_drift_cuts_at_the_limit(offset: float, expected: float):
    states = _arc_states()
    gt_path = _gt_path_of(states)
    shifted = states.copy()
    heading = shifted[0, :, StateIndex.HEADING]
    shifted[0, :, StateIndex.X] += -np.sin(heading) * offset
    shifted[0, :, StateIndex.Y] += np.cos(heading) * offset
    scorer = PDMScorer(SAMPLING)
    _prime(scorer, shifted, gt_path)

    scorer._calculate_gt_compliance()

    assert scorer._multi_metrics[MultiMetricIndex.GT_COMPLIANCE, 0] == expected
