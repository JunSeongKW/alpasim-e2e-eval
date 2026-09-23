import numpy as np
import pytest
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.state_representation import StateSE2, StateVector2D, TimePoint
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.planning.simulation.planner.nurec_controller.nurec_simulator import NuRecSimulator
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex

SAMPLING = TrajectorySampling(num_poses=40, interval_length=0.1)


def _ego_state(speed=12.0):
    return EgoState.build_from_rear_axle(
        rear_axle_pose=StateSE2(0.0, 0.0, 0.0),
        rear_axle_velocity_2d=StateVector2D(speed, 0.0),
        rear_axle_acceleration_2d=StateVector2D(0.0, 0.0),
        tire_steering_angle=0.0,
        time_point=TimePoint(0),
        vehicle_parameters=get_pacifica_parameters(),
    )


def _desired(tracks):
    """Pack ``[B, T, 3]`` (x, y, yaw) into the state-array layout."""
    states = np.zeros((tracks.shape[0], SAMPLING.num_poses + 1, StateIndex.size()))
    states[:, :, StateIndex.X] = tracks[:, :, 0]
    states[:, :, StateIndex.Y] = tracks[:, :, 1]
    states[:, :, StateIndex.HEADING] = tracks[:, :, 2]
    return states


def _straight(speed):
    x = np.arange(SAMPLING.num_poses + 1) * speed * SAMPLING.interval_length
    return np.stack([x, np.zeros_like(x), np.zeros_like(x)], axis=1)


def _lane_change(speed, lateral):
    x = np.arange(SAMPLING.num_poses + 1) * speed * SAMPLING.interval_length
    y = lateral * np.clip(np.linspace(0.0, 1.6, len(x)), 0.0, 1.0)
    return np.stack([x, y, np.arctan2(np.gradient(y), np.gradient(x))], axis=1)


def test_shape_and_first_row_match_the_input():
    simulator = NuRecSimulator(SAMPLING)
    states = _desired(_straight(12.0)[None])
    ego = _ego_state()

    simulated = simulator.simulate_proposals(states, ego)

    assert simulated.shape == states.shape
    assert np.isfinite(simulated).all()
    np.testing.assert_allclose(simulated[0, 0, StateIndex.X], 0.0, atol=1e-12)
    np.testing.assert_allclose(simulated[0, 0, StateIndex.VELOCITY_X], 12.0, atol=1e-12)


def test_a_straight_proposal_at_the_current_speed_is_tracked_closely():
    simulator = NuRecSimulator(SAMPLING)
    track = _straight(12.0)
    simulated = simulator.simulate_proposals(_desired(track[None]), _ego_state(12.0))

    lateral = np.abs(simulated[0, :, StateIndex.Y])
    assert lateral.max() < 0.05, f"drifted {lateral.max():.3f} m off a straight line"
    longitudinal = abs(simulated[0, -1, StateIndex.X] - track[-1, 0])
    assert longitudinal < 0.5, f"fell {longitudinal:.2f} m behind"


def test_a_lane_change_is_followed_in_the_right_direction():
    simulator = NuRecSimulator(SAMPLING)
    tracks = np.stack([_lane_change(12.0, 3.5), _lane_change(12.0, -3.5)])
    simulated = simulator.simulate_proposals(_desired(tracks), _ego_state(12.0))

    assert simulated[0, -1, StateIndex.Y] > 1.0
    assert simulated[1, -1, StateIndex.Y] < -1.0
    # Mirror-image references must give mirror-image tracks.
    np.testing.assert_allclose(simulated[0, :, StateIndex.Y], -simulated[1, :, StateIndex.Y],
                               rtol=0, atol=1e-6)


def test_the_horizon_is_fed_past_the_end_of_the_proposal():
    # The controller looks 2 s ahead while a proposal is only 4 s long, so its
    # tail would otherwise be tracked against a clamped final pose and every
    # proposal would brake through it.
    simulator = NuRecSimulator(SAMPLING)
    simulated = simulator.simulate_proposals(_desired(_straight(12.0)[None]), _ego_state(12.0))

    speed = simulated[0, :, StateIndex.VELOCITY_X]
    assert speed.min() > 11.5, f"slowed to {speed.min():.2f} m/s on a constant-speed reference"


def test_proposals_in_a_batch_do_not_influence_each_other():
    simulator = NuRecSimulator(SAMPLING)
    tracks = np.stack([_straight(12.0), _lane_change(12.0, 3.5), _lane_change(12.0, -2.0)])
    together = simulator.simulate_proposals(_desired(tracks), _ego_state(12.0))

    for i in range(len(tracks)):
        alone = NuRecSimulator(SAMPLING).simulate_proposals(
            _desired(tracks[i][None]), _ego_state(12.0))
        # Not bit-identical: the batched ADMM stops when every item has
        # converged, so an item run alongside others takes a few more
        # iterations. The gap has to stay far below anything the metrics can
        # see -- sub-millimetre, not sub-metre.
        np.testing.assert_allclose(together[i], alone[0], rtol=0, atol=1e-3,
                                   err_msg=f"proposal {i} depends on its batch mates")


def test_the_dynamic_states_the_comfort_metrics_read_are_populated():
    simulator = NuRecSimulator(SAMPLING)
    simulated = simulator.simulate_proposals(_desired(_lane_change(14.0, 3.0)[None]), _ego_state(14.0))
    row = simulated[0, 1:]

    for index in (StateIndex.ACCELERATION_X, StateIndex.ACCELERATION_Y,
                  StateIndex.STEERING_ANGLE, StateIndex.STEERING_RATE,
                  StateIndex.ANGULAR_VELOCITY, StateIndex.ANGULAR_ACCELERATION):
        assert np.isfinite(row[:, index]).all()
    assert np.abs(row[:, StateIndex.ANGULAR_VELOCITY]).max() > 1e-3, "a lane change must yaw"
