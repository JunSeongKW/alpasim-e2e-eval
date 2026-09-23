"""The four-way driving command, rebuilt from the NuRec route.

The original DriveSuprim checkpoints (r34 / V2-99 / ViT-L) have no route
encoder: their navigation intent arrives as NAVSIM's one-hot
``[left, straight, right, unknown]``. NuRec logs such a field, but it reads
straight on 85% of frames where NAVSIM reads 62%, so a NAVSIM-trained model sees
far less intent than it was trained on. The route carries the same information
and, measured the way NAVSIM measures it, in the same proportions.

The measurement detail that decides the answer: the NuRec route does not start
under the car. Its first waypoint sits about 42 m ahead and the polyline runs
roughly 38 m from there. Lateral offset taken in ego coordinates therefore
includes 42 m of lever arm from road the ego has not reached, and reads a turn
into almost every bend -- 19% / 56% / 24% against NAVSIM's 16 / 62 / 22. Taken
from the route's own first waypoint it gives 16 / 64 / 20.
"""

import numpy as np
import pytest

from navsim.common.driving_command import (
    DEFAULT_LATERAL_M,
    LEFT,
    RIGHT,
    STRAIGHT,
    UNKNOWN,
    driving_command_from_route,
)


def _route(start_x: float, points):
    """A route polyline starting at ``start_x`` metres ahead of the ego."""
    return np.asarray([[start_x + x, y] for x, y in points], dtype=np.float32)


def _straight_route(start_x=42.0, length=38.0, n=10, lateral=0.0, offset=0.0):
    xs = np.linspace(0.0, length, n)
    ys = np.linspace(0.0, lateral, n) + offset
    return _route(start_x, list(zip(xs, ys)))


def test_a_straight_route_is_straight():
    np.testing.assert_array_equal(driving_command_from_route(_straight_route()), STRAIGHT)


def test_a_route_bending_left_is_left():
    route = _straight_route(lateral=+6.0)
    np.testing.assert_array_equal(driving_command_from_route(route), LEFT)


def test_a_route_bending_right_is_right():
    route = _straight_route(lateral=-6.0)
    np.testing.assert_array_equal(driving_command_from_route(route), RIGHT)


def test_a_lane_change_counts_as_a_turn():
    """NAVSIM's left/right buckets include lane changes, not only junctions."""
    route = _straight_route(lateral=-3.5)
    assert 3.5 > DEFAULT_LATERAL_M
    np.testing.assert_array_equal(driving_command_from_route(route), RIGHT)


def test_a_route_that_starts_offset_but_runs_straight_is_still_straight():
    """The reason for measuring from the first waypoint.

    The ego sits 8 m to the right of a route that then runs dead straight. In ego
    coordinates the endpoint is 8 m to the left, which the lateral test would
    call a left turn; relative to the route's own start there is no manoeuvre.
    """
    route = _straight_route(lateral=0.0, offset=8.0)
    assert route[-1, 1] > DEFAULT_LATERAL_M  # would be LEFT measured from the ego
    np.testing.assert_array_equal(driving_command_from_route(route), STRAIGHT)


def test_a_tight_turn_is_caught_by_heading_even_when_it_covers_little_ground():
    """A route curving through 40 degrees over a short distance."""
    angles = np.radians(np.linspace(0.0, 40.0, 12))
    radius = 4.0
    points = [(radius * np.sin(a), radius * (1.0 - np.cos(a))) for a in angles]
    assert abs(points[-1][1]) < DEFAULT_LATERAL_M  # lateral alone would say straight
    np.testing.assert_array_equal(driving_command_from_route(_route(42.0, points)), LEFT)


def test_missing_and_degenerate_routes_are_unknown():
    for route in (None, np.zeros((0, 2)), np.full((10, 2), np.nan), np.zeros((1, 2))):
        np.testing.assert_array_equal(driving_command_from_route(route), UNKNOWN)


def test_nan_padding_is_dropped_not_treated_as_a_waypoint():
    """EgoStatus.route_waypoints is a fixed 20 slots, NaN-padded."""
    route = np.full((20, 2), np.nan, dtype=np.float32)
    route[:10] = _straight_route(lateral=-6.0)
    np.testing.assert_array_equal(driving_command_from_route(route), RIGHT)


def test_the_command_is_one_hot():
    for route in (_straight_route(), _straight_route(lateral=9.0), None):
        command = driving_command_from_route(route)
        assert command.shape == (4,)
        assert command.sum() == 1


@pytest.mark.parametrize("lateral,expected", [(+9.0, LEFT), (-9.0, RIGHT), (0.0, STRAIGHT)])
def test_extra_columns_beyond_xy_are_ignored(lateral, expected):
    """Waypoints may carry more than x and y; only the first two are read."""
    route = _straight_route(lateral=lateral)
    padded = np.concatenate([route, np.full((len(route), 1), 7.0, np.float32)], axis=1)
    np.testing.assert_array_equal(driving_command_from_route(padded), expected)
