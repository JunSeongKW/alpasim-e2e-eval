import numpy as np
import pytest
import torch

from navsim.planning.simulation.planner.nurec_controller.batch_linearize import linearize
from navsim.planning.simulation.planner.nurec_controller.reference.linear_mpc import LinearMPC

DT = 0.1

STATES = [
    [0.0, 0.0, 0.0, 12.0, 0.0, 0.0, 0.0, 0.0],
    [3.0, -1.0, 0.4, 18.0, 0.2, 0.05, 0.1, 1.0],
    [0.0, 0.0, -0.3, 2.0, 0.0, 0.0, 0.2, 0.0],      # kinematic branch
    [0.0, 0.0, 0.0, 4.999, 0.0, 0.0, 0.0, 0.0],     # just below the switch
    [0.0, 0.0, 0.0, 5.001, 0.0, 0.0, 0.0, 0.0],     # just above it
    [0.0, 0.0, 1.2, 30.0, -0.4, -0.1, -0.2, -2.0],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],       # at rest
]


@pytest.mark.parametrize("state", STATES)
def test_matches_the_reference_linearisation(state):
    reference = LinearMPC()
    expected_a, expected_b = reference._linearize_dynamics(np.asarray(state))
    actual_a, actual_b = linearize(torch.as_tensor([state], dtype=torch.float64),
                                   reference._vehicle_params, DT)

    np.testing.assert_allclose(actual_a[0].numpy(), expected_a, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(actual_b[0].numpy(), expected_b, rtol=1e-11, atol=1e-13)


def test_a_mixed_batch_keeps_each_proposal_on_its_own_branch():
    # Both branches are built for the whole batch and then selected; a stale
    # entry from the unselected branch would only show up in a mixed batch.
    reference = LinearMPC()
    actual_a, actual_b = linearize(torch.as_tensor(STATES, dtype=torch.float64),
                                   reference._vehicle_params, DT)
    for i, state in enumerate(STATES):
        expected_a, expected_b = reference._linearize_dynamics(np.asarray(state))
        np.testing.assert_allclose(actual_a[i].numpy(), expected_a, rtol=1e-11, atol=1e-13,
                                   err_msg=f"state {i} A mismatch")
        np.testing.assert_allclose(actual_b[i].numpy(), expected_b, rtol=1e-11, atol=1e-13,
                                   err_msg=f"state {i} B mismatch")


def test_a_resting_proposal_stays_finite():
    reference = LinearMPC()
    a, b = linearize(torch.zeros((4, 8), dtype=torch.float64), reference._vehicle_params, DT)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
