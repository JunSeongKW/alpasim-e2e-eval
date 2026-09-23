import numpy as np
import pytest
import torch

from navsim.planning.simulation.planner.nurec_controller.batch_vehicle_model import BatchVehicleModel
from navsim.planning.simulation.planner.nurec_controller.reference.vehicle_model import VehicleModel

DT = 0.1


def _reference_rollout(velocity, yaw_rate, commands, steps):
    model = VehicleModel(np.asarray(velocity, dtype=np.float64), float(yaw_rate))
    for _ in range(steps):
        model.advance(np.asarray(commands, dtype=np.float64), DT)
    return model.state.copy()


def _batch_rollout(velocities, yaw_rates, commands, steps):
    model = BatchVehicleModel(len(velocities), np.asarray(velocities), np.asarray(yaw_rates),
                              device="cpu")
    for _ in range(steps):
        model.advance(np.asarray(commands, dtype=np.float64), DT)
    return model.state.cpu().numpy()


CASES = [
    # (vx, vy, yaw_rate, steering_cmd, accel_cmd) -- spans both the low-speed
    # kinematic branch and the dynamic branch, and the switch between them.
    (12.0, 0.0, 0.0, 0.0, 0.0),
    (12.0, 0.0, 0.0, 0.15, 0.0),
    (12.0, 0.0, 0.0, -0.30, 1.5),
    (2.0, 0.0, 0.0, 0.25, 2.0),        # starts kinematic, accelerates across 5 m/s
    (8.0, 0.2, 0.1, 0.05, -3.0),       # decelerates down across the threshold
    (0.0, 0.0, 0.0, 0.4, 3.0),         # from rest
    (25.0, -0.3, -0.05, 0.02, -1.0),
]


@pytest.mark.parametrize("case", CASES)
def test_one_proposal_matches_the_reference(case):
    vx, vy, yaw_rate, steer, accel = case
    expected = _reference_rollout([vx, vy], yaw_rate, [steer, accel], steps=40)
    actual = _batch_rollout([[vx, vy]], [yaw_rate], [[steer, accel]], steps=40)[0]
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_a_mixed_batch_matches_the_references_run_one_at_a_time():
    # The vectorised _derivs evaluates both branches for every element; this is
    # what catches a mask that leaks the wrong branch's value.
    velocities = [[c[0], c[1]] for c in CASES]
    yaw_rates = [c[2] for c in CASES]
    commands = [[c[3], c[4]] for c in CASES]

    actual = _batch_rollout(velocities, yaw_rates, commands, steps=40)
    for i, case in enumerate(CASES):
        expected = _reference_rollout([case[0], case[1]], case[2], [case[3], case[4]], steps=40)
        np.testing.assert_allclose(actual[i], expected, rtol=0, atol=1e-12,
                                   err_msg=f"proposal {i} diverged: {case}")


def test_the_dynamic_branch_never_leaks_a_division_by_zero():
    # v_x = 0 makes the dynamic coefficients infinite; they are computed for the
    # whole batch and only then masked away, so they must stay finite.
    model = BatchVehicleModel(3, np.zeros((3, 2)), np.zeros(3), device="cpu")
    derivs = model._derivs(model.state, torch.zeros((3, 2), dtype=torch.float64))
    assert torch.isfinite(derivs).all()


def test_accelerations_match_the_reference():
    reference = VehicleModel(np.array([14.0, 0.0]), 0.0)
    batch = BatchVehicleModel(1, np.array([[14.0, 0.0]]), np.array([0.0]), device="cpu")
    for _ in range(5):
        reference.advance(np.array([0.12, 1.0]), DT)
        batch.advance(np.array([[0.12, 1.0]]), DT)
    np.testing.assert_allclose(batch.accelerations[0].cpu().numpy(), reference.accelerations,
                               rtol=0, atol=1e-12)


def test_reset_origin_and_set_velocity_behave_like_the_reference():
    batch = BatchVehicleModel(2, np.array([[9.0, 0.0], [4.0, 0.0]]), np.zeros(2), device="cpu")
    batch.advance(np.array([[0.1, 0.0], [0.1, 0.0]]), DT)
    batch.reset_origin()
    assert torch.count_nonzero(batch.state[:, :3]) == 0
    batch.set_velocity(np.array([3.0, 7.0]), np.array([0.1, -0.1]))
    np.testing.assert_allclose(batch.state[:, 3].cpu().numpy(), [3.0, 7.0])
    np.testing.assert_allclose(batch.state[:, 4].cpu().numpy(), [0.1, -0.1])
