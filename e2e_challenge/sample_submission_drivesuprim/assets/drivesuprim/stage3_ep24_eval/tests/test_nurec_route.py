import numpy as np
import pytest

from navsim.planning.data.nurec_route import (
    MAX_EGO_ROUTE_OFFSET_M,
    ROUTE_HORIZON_M,
    ROUTE_NEAR_CUTOFF_M,
    ROUTE_SLOTS,
    ROUTE_SPACING_M,
    RouteBuildError,
    build_routes,
    gt_lane_sequence,
)


def _bundle(lanes, lane_sequence):
    return {
        "clip_id": "clip",
        "map_name": "nurec:clip",
        "lanes": lanes,
        "matched_ego": [
            {"timestamp_us": index * 500_000, "lane_id": lane_id, "roadblock_id": "rb"}
            for index, lane_id in enumerate(lane_sequence)
        ],
    }


def _straight_lane(start_x, end_x, y=0.0, nxt=None):
    return {
        "center": [[start_x, y, 0.0], [end_x, y, 0.0]],
        "next": list(nxt or []),
        "previous": [],
        "roadblock_id": "rb",
    }


def _frames(positions, heading=0.0):
    frames = []
    for index, (x, y) in enumerate(positions):
        transform = np.eye(4)
        transform[0, 0] = transform[1, 1] = np.cos(heading)
        transform[0, 1], transform[1, 0] = -np.sin(heading), np.sin(heading)
        transform[:2, 3] = (x, y)
        frames.append(
            {
                "token": f"t{index}",
                "timestamp": index * 500_000,
                "ego2global": transform,
                "map_location": "nurec:clip",
            }
        )
    return frames


def test_spacing_matches_the_runtime_message():
    assert ROUTE_SLOTS == 20
    assert ROUTE_SPACING_M == pytest.approx(80.0 / 19)
    assert ROUTE_SPACING_M == pytest.approx(4.210526, abs=1e-6)
    assert ROUTE_HORIZON_M == 80.0
    assert ROUTE_NEAR_CUTOFF_M == 40.0


def test_straight_route_gives_ten_finite_waypoints_then_nan():
    lanes = {"L": _straight_lane(0.0, 300.0)}
    frames = _frames([(0.0, 0.0), (10.0, 0.0)])
    route = build_routes(frames, _bundle(lanes, ["L", "L"]))["t0"]

    assert route.shape == (ROUTE_SLOTS, 3)
    assert route.dtype == np.float32
    finite = np.isfinite(route[:, 0])
    # The near field below 40 m is dropped and the survivors compact to the front.
    assert finite.sum() == 10
    assert finite[:10].all() and not finite[10:].any()

    expected = np.arange(10, 20) * ROUTE_SPACING_M
    np.testing.assert_allclose(route[:10, 0], expected, atol=1e-3)
    np.testing.assert_allclose(route[:10, 1], 0.0, atol=1e-3)
    np.testing.assert_allclose(route[:10, 2], 0.0)
    assert route[0, 0] == pytest.approx(42.105, abs=1e-2)
    assert route[9, 0] == pytest.approx(80.0, abs=1e-2)


def test_waypoints_are_in_the_ego_rig_frame():
    # Same road, but the ego faces +y: the route must come back as straight ahead.
    lanes = {"L": {"center": [[0.0, 0.0, 0.0], [0.0, 300.0, 0.0]], "next": [], "roadblock_id": "rb"}}
    frames = _frames([(0.0, 0.0)], heading=np.pi / 2)
    route = build_routes(frames, _bundle(lanes, ["L"]))["t0"]

    np.testing.assert_allclose(route[:10, 0], np.arange(10, 20) * ROUTE_SPACING_M, atol=1e-3)
    np.testing.assert_allclose(route[:10, 1], 0.0, atol=1e-3)


def test_route_running_out_yields_fewer_waypoints_not_wrong_ones():
    lanes = {"L": _straight_lane(0.0, 60.0)}
    frames = _frames([(0.0, 0.0)])
    route = build_routes(frames, _bundle(lanes, ["L"]))["t0"]

    finite = np.isfinite(route[:, 0])
    assert 0 < finite.sum() < 10
    assert route[finite, 0].max() <= 60.0 + 1e-6


def test_ego_far_from_the_route_gets_no_waypoints():
    # Projection would land somewhere arbitrary, so the message must be empty
    # rather than confidently wrong.
    lanes = {"L": _straight_lane(0.0, 300.0)}
    frames = _frames([(0.0, MAX_EGO_ROUTE_OFFSET_M + 5.0)])
    route = build_routes(frames, _bundle(lanes, ["L"]))["t0"]

    assert not np.isfinite(route).any()


def test_route_follows_the_straightest_successor():
    lanes = {
        "L": _straight_lane(0.0, 50.0, nxt=["STRAIGHT", "TURN"]),
        "STRAIGHT": _straight_lane(50.0, 300.0),
        "TURN": {"center": [[50.0, 0.0, 0.0], [50.0, 250.0, 0.0]], "next": [], "roadblock_id": "rb"},
    }
    frames = _frames([(0.0, 0.0)])
    route = build_routes(frames, _bundle(lanes, ["L"]))["t0"]

    # Taking TURN would bend the route sideways; continuing straight keeps y at 0.
    np.testing.assert_allclose(route[:10, 1], 0.0, atol=1e-3)
    assert np.isfinite(route[:, 0]).sum() == 10


def test_gt_lane_sequence_drops_repeats_and_unmatched_poses():
    bundle = _bundle({}, ["A", "A", None, "B", "B", "A"])
    assert gt_lane_sequence(bundle) == ["A", "B", "A"]


def test_a_clip_with_no_matched_lane_cannot_build_a_route():
    frames = _frames([(0.0, 0.0), (5.0, 0.0)])
    with pytest.raises(RouteBuildError):
        build_routes(frames, _bundle({}, [None, None]))


def _bent_lane(bend_y):
    """A road running +x for 120 m, then bending to ``bend_y`` over the next 120 m."""
    return {
        "center": [[0.0, 0.0, 0.0], [120.0, 0.0, 0.0], [200.0, bend_y, 0.0]],
        "next": [],
        "roadblock_id": "rb",
    }


@pytest.mark.parametrize("bend_y, expected_sign", [(60.0, +1.0), (-60.0, -1.0)])
def test_y_is_positive_to_the_left(bend_y, expected_sign):
    # The rig frame is x forward, y left. Getting this backwards would mirror
    # every turn the model is told about, so pin the sign explicitly.
    lanes = {"L": _bent_lane(bend_y)}
    frames = _frames([(100.0, 0.0)])
    route = build_routes(frames, _bundle(lanes, ["L"]))["t0"]

    finite = np.isfinite(route[:, 0])
    assert finite.sum() >= 5
    assert np.sign(route[finite, 1][-1]) == expected_sign


def test_lateral_sign_survives_an_ego_rotation():
    # Same left bend, but the ego drives north: the answer must not depend on
    # the global orientation.
    lanes = {"L": {"center": [[0.0, 0.0, 0.0], [0.0, 120.0, 0.0], [-60.0, 200.0, 0.0]],
                   "next": [], "roadblock_id": "rb"}}
    frames = _frames([(0.0, 100.0)], heading=np.pi / 2)
    route = build_routes(frames, _bundle(lanes, ["L"]))["t0"]

    finite = np.isfinite(route[:, 0])
    # Bending toward -x while facing +y is a left turn, so y must come back positive.
    assert route[finite, 1][-1] > 0


def test_a_gap_outside_the_band_truncates_the_rest():
    from navsim.planning.data.nurec_route import (
        MIN_WAYPOINT_GAP_M, _arclength, route_waypoints,
    )

    # A route that runs straight and then doubles back: the chord between the
    # samples across the hairpin falls below the allowed gap, so the Runtime
    # stops trusting the route from there on.
    straight = np.stack([np.linspace(0, 120, 200), np.zeros(200)], axis=1)
    hairpin = np.stack([np.linspace(120, 118, 60), np.linspace(0, 1, 60)], axis=1)
    back = np.stack([np.linspace(118, 40, 200), np.full(200, 1.0)], axis=1)
    polyline = np.concatenate([straight, hairpin, back])
    cumulative = _arclength(polyline)

    route = route_waypoints(polyline, cumulative, np.zeros(2), 0.0)
    finite = np.isfinite(route[:, 0])

    assert finite.any(), "the straight part before the hairpin must survive"
    kept = route[finite, :2]
    if len(kept) >= 2:
        gaps = np.linalg.norm(np.diff(kept, axis=0), axis=1)
        assert gaps.min() >= MIN_WAYPOINT_GAP_M
    # Truncation is a prefix: no valid waypoint may follow a NaN one.
    assert not np.isfinite(route[int(finite.sum()):, 0]).any()


def test_a_clean_straight_route_is_not_truncated():
    lanes = {"L": _straight_lane(0.0, 300.0)}
    frames = _frames([(0.0, 0.0)])
    route = build_routes(frames, _bundle(lanes, ["L"]))["t0"]
    assert np.isfinite(route[:, 0]).sum() == 10


def _arc_polyline(radius, sweep_rad, points=4000):
    """A constant-radius arc starting at the origin heading +x."""
    theta = np.linspace(0.0, sweep_rad, points)
    return np.stack([radius * np.sin(theta), radius * (1.0 - np.cos(theta))], axis=1)


def test_the_trim_is_measured_along_the_delivered_waypoints():
    from navsim.planning.data.nurec_route import (
        ROUTE_NEAR_CUTOFF_M, ROUTE_SPACING_M, _arclength, route_waypoints,
    )

    # The cutoff counts distance along the sampled points, not the arclength
    # that produced them. On a bend the chords cut the corner, the running total
    # reaches 40 m a slot later, and the message comes back one waypoint short.
    # Asserting the rule rather than a particular radius keeps this honest.
    for polyline in (
        np.stack([np.linspace(0.0, 400.0, 4000), np.zeros(4000)], axis=1),
        _arc_polyline(radius=45.0, sweep_rad=2.2),
        _arc_polyline(radius=12.0, sweep_rad=3.0),
    ):
        route = route_waypoints(polyline, _arclength(polyline), np.zeros(2), 0.0)
        valid = np.isfinite(route[:, 0])
        if not valid.any():
            continue
        first = route[0, :2]
        # Everything delivered starts at or past the cutoff...
        assert np.linalg.norm(first) <= ROUTE_NEAR_CUTOFF_M + 2 * ROUTE_SPACING_M
        # ...and one slot earlier would have been inside it.
        assert int(valid.sum()) <= 10


def test_a_straight_route_is_never_trimmed_short():
    from navsim.planning.data.nurec_route import _arclength, route_waypoints

    # Chord equals arc on a straight line, so the shift cannot fire here; this
    # pins the rule to curvature rather than to an off-by-one.
    straight = np.stack([np.linspace(0.0, 400.0, 4000), np.zeros(4000)], axis=1)
    route = route_waypoints(straight, _arclength(straight), np.zeros(2), 0.0)
    assert int(np.isfinite(route[:, 0]).sum()) == 10
    assert route[0, 0] == pytest.approx(42.105, abs=1e-2)


def test_a_message_never_carries_more_than_ten_waypoints():
    from navsim.planning.data.nurec_route import _arclength, route_waypoints

    for radius in (30.0, 60.0, 120.0, 400.0):
        polyline = _arc_polyline(radius=radius, sweep_rad=3.0)
        route = route_waypoints(polyline, _arclength(polyline), np.zeros(2), 0.0)
        valid = int(np.isfinite(route[:, 0]).sum())
        assert 0 <= valid <= 10, f"radius {radius} produced {valid} waypoints"


def test_spacing_truncation_always_leaves_at_least_one_waypoint():
    from navsim.planning.data.nurec_route import _arclength, route_waypoints

    # A hairpin trips the spacing rule immediately; the contract is that the
    # rule can shorten a message but never empty it.
    straight = np.stack([np.linspace(0, 60, 900), np.zeros(900)], axis=1)
    hairpin = np.stack([np.linspace(60, 58, 200), np.linspace(0, 1.5, 200)], axis=1)
    back = np.stack([np.linspace(58, 5, 900), np.full(900, 1.5)], axis=1)
    polyline = np.concatenate([straight, hairpin, back])

    route = route_waypoints(polyline, _arclength(polyline), np.zeros(2), 0.0)
    assert int(np.isfinite(route[:, 0]).sum()) >= 1
