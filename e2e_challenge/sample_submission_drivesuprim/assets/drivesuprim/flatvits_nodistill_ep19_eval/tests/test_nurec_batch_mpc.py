import numpy as np
import pytest
import torch

from navsim.planning.simulation.planner.nurec_controller.batch_linear_mpc import (
    BatchLinearMPC,
    interpolate_reference,
    wrap_to_pi,
)
from navsim.planning.simulation.planner.nurec_controller.geometry import Pose, Trajectory
from navsim.planning.simulation.planner.nurec_controller.reference.linear_mpc import LinearMPC
from navsim.planning.simulation.planner.nurec_controller.reference.mpc_controller import ControllerInput

T0 = 1_000_000
DT = 0.1
N_POSES = 41


def _track(speed, lateral, curvature):
    x = np.cumsum(np.full(N_POSES, speed * DT))
    y = lateral * np.linspace(0.0, 1.0, N_POSES) + curvature * x ** 2 / 2.0
    yaw = np.arctan2(np.gradient(y), np.gradient(x))
    return np.stack([x, y, yaw], axis=1)


CASES = [
    (12.0, 0.0, 0.0),
    (12.0, 3.0, 0.0),
    (12.0, -3.0, 0.0),
    (4.0, 1.0, 0.01),      # kinematic branch
    (20.0, -2.0, -0.008),
    (8.0, 0.0, 0.02),
]


def _times():
    return np.array([T0 + int(k * DT * 1e6) for k in range(N_POSES)], dtype=np.int64)


def _reference_control(state, track, timestamp_us=T0):
    trajectory = Trajectory.from_poses(
        _times().tolist(), [Pose.from_xy_yaw(*pose) for pose in track]
    )
    return LinearMPC().compute_control(
        ControllerInput(state=state, reference_trajectory=trajectory, timestamp_us=timestamp_us)
    ).control


def test_interpolation_matches_the_reference_path():
    track = _track(11.0, 2.0, 0.004)
    trajectory = Trajectory.from_poses(_times().tolist(), [Pose.from_xy_yaw(*p) for p in track])
    # Deliberately runs past the end of the trajectory, where the reference
    # clamps into its half-open range instead of extrapolating.
    targets = np.array([T0 + int(k * DT * 1e6) for k in range(30, 51)], dtype=np.int64)
    clamped = np.clip(targets, trajectory.time_range_us.start, trajectory.time_range_us.stop - 1)
    expected = trajectory.interpolate(clamped.astype(np.uint64))

    actual = interpolate_reference(torch.as_tensor(track[None]), torch.as_tensor(_times()), torch.as_tensor(targets))[0].numpy()

    for k in range(len(targets)):
        pose = expected.get_pose(k)
        np.testing.assert_allclose(actual[k, :2], pose.vec3[:2], rtol=0, atol=1e-9)
        assert abs(float(wrap_to_pi(torch.tensor(actual[k, 2] - pose.yaw())))) < 1e-9


@pytest.mark.parametrize("case", CASES)
def test_one_proposal_matches_the_reference_controller(case):
    track = _track(*case)
    state = np.array([0.0, 0.0, 0.0, case[0], 0.0, 0.0, 0.0, 0.0])
    expected = _reference_control(state, track)

    targets = np.array([T0 + int(k * DT * 1e6) for k in range(21)], dtype=np.int64)
    reference = interpolate_reference(torch.as_tensor(track[None]), torch.as_tensor(_times()), torch.as_tensor(targets))
    actual = BatchLinearMPC(device='cpu').compute_control(state[None], reference)[0].numpy()

    # Both solve the same QP to a 1e-4 residual, so the controls agree well
    # inside that; an outright formulation mismatch shows up far larger.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-3)


def test_a_mixed_batch_matches_the_reference_run_one_at_a_time():
    tracks = np.stack([_track(*case) for case in CASES])
    states = np.array([[0.0, 0.0, 0.0, case[0], 0.0, 0.0, 0.0, 0.0] for case in CASES])
    targets = np.array([T0 + int(k * DT * 1e6) for k in range(21)], dtype=np.int64)

    actual = BatchLinearMPC(device='cpu').compute_control(
        states, interpolate_reference(torch.as_tensor(tracks), torch.as_tensor(_times()), torch.as_tensor(targets))).numpy()

    for i, case in enumerate(CASES):
        expected = _reference_control(states[i], tracks[i])
        np.testing.assert_allclose(actual[i], expected, rtol=0, atol=1e-3,
                                   err_msg=f"proposal {i} diverged: {case}")


def test_steering_is_mirrored_for_mirrored_references():
    # A sign slip anywhere in the rig-frame maths would break this symmetry.
    targets = np.array([T0 + int(k * DT * 1e6) for k in range(21)], dtype=np.int64)
    left, right = _track(12.0, 3.0, 0.0), _track(12.0, -3.0, 0.0)
    states = np.tile([0.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 0.0], (2, 1))

    controls = BatchLinearMPC(device='cpu').compute_control(
        states, interpolate_reference(torch.as_tensor(np.stack([left, right])),
                                      torch.as_tensor(_times()), torch.as_tensor(targets))
    ).numpy()
    assert controls[0, 0] > 0.05
    np.testing.assert_allclose(controls[0, 0], -controls[1, 0], rtol=0, atol=1e-4)


def test_commands_respect_the_input_bounds():
    # An unreachable reference must saturate, not run away.
    track = _track(12.0, 60.0, 0.0)
    targets = np.array([T0 + int(k * DT * 1e6) for k in range(21)], dtype=np.int64)
    control = BatchLinearMPC(device='cpu').compute_control(
        np.array([[0.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 0.0]]),
        interpolate_reference(torch.as_tensor(track[None]), torch.as_tensor(_times()),
                              torch.as_tensor(targets)),
    )[0].numpy()
    assert -2.0 - 1e-3 <= control[0] <= 2.0 + 1e-3
    assert -9.0 - 1e-3 <= control[1] <= 6.0 + 1e-3


def test_an_unreachable_reference_still_yields_a_command():
    # Asking for 40 m/s from 12 m/s cannot be tracked -- the reference's own
    # state bound is 35 m/s -- and the QP does not converge on it. OSQP calls
    # that "solved inaccurate" and the reference uses the answer anyway; a zero
    # command here would silently mean "coast", which is a different trajectory.
    track = _track(40.0, 0.0, 0.0)
    targets = np.array([T0 + int(k * DT * 1e6) for k in range(21)], dtype=np.int64)
    control = BatchLinearMPC(device="cpu").compute_control(
        np.array([[0.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 0.0]]),
        interpolate_reference(torch.as_tensor(track[None]), torch.as_tensor(_times()),
                              torch.as_tensor(targets)),
    )[0].numpy()

    assert np.isfinite(control).all()
    assert control[1] > 0.5, f"an unreachable speed target must still accelerate, got {control[1]}"
    assert -9.0 - 1e-3 <= control[1] <= 6.0 + 1e-3
