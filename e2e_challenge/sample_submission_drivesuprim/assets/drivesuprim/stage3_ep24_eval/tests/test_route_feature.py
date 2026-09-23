import numpy as np
import pytest
import torch

from navsim.agents.drivesuprim.drivesuprim_config import DriveSuprimConfig
from navsim.agents.drivesuprim.drivesuprim_features import DriveSuprimFeatureBuilder
from navsim.common.dataclasses import AgentInput, EgoStatus
from navsim.planning.data.nurec_route import ROUTE_SLOTS, ROUTE_SPACING_M


def _builder(**overrides):
    config = DriveSuprimConfig(training=False, use_route=True, **overrides)
    return DriveSuprimFeatureBuilder(config), config


def _agent_input(route):
    status = EgoStatus(
        ego_pose=np.zeros(3),
        ego_velocity=np.zeros(2, dtype=np.float32),
        ego_acceleration=np.zeros(2, dtype=np.float32),
        driving_command=np.array([0, 1, 0, 0]),
        route_waypoints=np.asarray(route, dtype=np.float32),
    )
    return AgentInput(ego_statuses=[status], cameras=[], lidars=[])


def _straight_route(lateral=0.0):
    route = np.full((ROUTE_SLOTS, 3), np.nan, dtype=np.float32)
    route[:10, 0] = np.arange(10, 20) * ROUTE_SPACING_M
    route[:10, 1] = lateral
    route[:10, 2] = 0.0
    return route


def test_padding_never_reaches_the_tensor_as_nan():
    # One NaN through the encoder's Linear would spread across the batch.
    builder, config = _builder()
    features = builder._get_route_feature(_agent_input(_straight_route()), "token", rotation_num=0)

    assert torch.isfinite(features["route_feature"]).all()
    assert features["route_feature"].shape == (ROUTE_SLOTS, 2)
    assert features["route_mask"].tolist() == [1.0] * 10 + [0.0] * 10
    # Padded slots are zeroed, not left as whatever NaN cast to.
    assert torch.count_nonzero(features["route_feature"][10:]) == 0


def test_waypoints_are_normalised_by_the_horizon():
    builder, config = _builder()
    features = builder._get_route_feature(_agent_input(_straight_route()), "token", rotation_num=0)

    expected = np.arange(10, 20) * ROUTE_SPACING_M / config.route_norm_m
    np.testing.assert_allclose(features["route_feature"][:10, 0].numpy(), expected, atol=1e-5)
    assert features["route_feature"][:10, 0].max() <= 1.0 + 1e-5


def test_rotation_augmentation_turns_the_route_with_the_frame():
    # rotated_trajectories rotate by -rot; the route must match or the augmented
    # sample points its BEV one way and its intent another.
    builder, config = _builder()
    builder.aug_info = {"token": [{"rot": 90.0}]}
    features = builder._get_route_feature(_agent_input(_straight_route()), "token", rotation_num=1)

    rotated = features["route_feature_rotated"][0]
    assert len(features["route_feature_rotated"]) == 1
    assert torch.isfinite(rotated).all()
    # Straight ahead, turned by -90 degrees, points along -y.
    original_x = features["route_feature"][:10, 0]
    np.testing.assert_allclose(rotated[:10, 0].numpy(), 0.0, atol=1e-5)
    np.testing.assert_allclose(rotated[:10, 1].numpy(), -original_x.numpy(), atol=1e-5)


@pytest.mark.parametrize("use_route, expected", [(True, [0.0] * 4), (False, [0.0, 1.0, 0.0, 0.0])])
def test_the_driving_command_is_zeroed_only_when_the_route_replaces_it(use_route, expected):
    config = DriveSuprimConfig(training=False, use_route=use_route, seq_len=1)
    builder = DriveSuprimFeatureBuilder(config)

    status = builder._get_status_feature(_agent_input(_straight_route()))[0]

    # The four command slots survive either way so the checkpoint still loads.
    assert status.shape == (8,)
    assert status[:4].tolist() == expected


def test_route_is_absent_when_the_flag_is_off():
    config = DriveSuprimConfig(training=False, use_route=False)
    assert not config.use_route


def _curved_route(lateral):
    """A route that starts 42 m ahead, as NuRec routes do, and bends by `lateral`."""
    route = np.full((ROUTE_SLOTS, 3), np.nan, dtype=np.float32)
    route[:10, 0] = np.arange(10, 20) * ROUTE_SPACING_M
    route[:10, 1] = np.linspace(0.0, lateral, 10)
    route[:10, 2] = 0.0
    return route


@pytest.mark.parametrize(
    "lateral, expected",
    [(0.0, [0.0, 1.0, 0.0, 0.0]), (9.0, [1.0, 0.0, 0.0, 0.0]), (-9.0, [0.0, 0.0, 1.0, 0.0])],
)
def test_the_command_is_rebuilt_from_the_route_for_a_command_only_checkpoint(lateral, expected):
    """Legacy checkpoints steer off the command; NuRec's logged one under-signals.

    The logged command here says straight on every case; the route says
    otherwise on two of them, and the route is what reaches the model.
    """
    config = DriveSuprimConfig(training=False, use_route=False, seq_len=1)
    builder = DriveSuprimFeatureBuilder(config)

    status = builder._get_status_feature(_agent_input(_curved_route(lateral)))[0]

    assert status[:4].tolist() == expected


def test_the_logged_command_stands_when_the_frame_has_no_route():
    config = DriveSuprimConfig(training=False, use_route=False, seq_len=1)
    builder = DriveSuprimFeatureBuilder(config)
    empty = np.full((ROUTE_SLOTS, 3), np.nan, dtype=np.float32)

    status = builder._get_status_feature(_agent_input(empty))[0]

    assert status[:4].tolist() == [0.0, 1.0, 0.0, 0.0]  # the logged 'straight'


def test_deriving_can_be_turned_off():
    config = DriveSuprimConfig(
        training=False, use_route=False, seq_len=1, derive_command_from_route=False
    )
    builder = DriveSuprimFeatureBuilder(config)

    status = builder._get_status_feature(_agent_input(_curved_route(9.0)))[0]

    assert status[:4].tolist() == [0.0, 1.0, 0.0, 0.0]  # the logged value, unchanged


def test_the_route_still_replaces_the_command_when_it_is_on():
    """With the route encoder in play the command slots stay zeroed regardless."""
    config = DriveSuprimConfig(training=False, use_route=True, seq_len=1)
    builder = DriveSuprimFeatureBuilder(config)

    status = builder._get_status_feature(_agent_input(_curved_route(9.0)))[0]

    assert status[:4].tolist() == [0.0] * 4
