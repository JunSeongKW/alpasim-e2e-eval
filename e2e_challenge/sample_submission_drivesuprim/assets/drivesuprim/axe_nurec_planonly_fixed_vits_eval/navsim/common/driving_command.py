"""NAVSIM's four-way driving command, derived from the NuRec route.

DriveSuprim never computes this field -- across DriveSuprim, DriveSuprim_YJ and
its bundled navsimv1.1 every reference only reads it. The definition it relies
on is NAVSIM's, stated in DriveSuprim's own docs/agents.md:

    "we provide a discrete driving command, indicating whether the intended
     route is towards the left, straight or right direction ... the driving
     command in NAVSIM is based solely on the desired route ... the left and
     right driving commands cover turns, lane changes and sharp curves."

The numeric rule lives upstream in the nuPlan-to-OpenScene conversion, which is
not in either repo, so this implements the definition rather than porting code.

NuRec logs a driving_command of their own, but it reads straight on 85% of
frames, so a NAVSIM-trained checkpoint sees far less intent than it was trained
on. The route says the same thing and is present on every frame.

**Measure from the route's first waypoint, not from the ego.** The NuRec route
does not begin under the car: its first waypoint sits about 42 m ahead and the
polyline runs roughly 38 m from there. Lateral offset taken in ego coordinates
therefore charges the route with 42 m of lever arm belonging to road the ego has
yet to reach, and turns nearly every gentle bend into a turn -- 19% left / 56%
straight / 24% right, against 16 / 64 / 20 measured from the route's own start.

**Only the reference point changes; the decision is unchanged.** Thresholds
(3 m lateral, 12 degrees), the OR between them, and the rule that a
lateral/heading disagreement falls back to straight are all as they stand in
SNU's EADv1.1_nvidia implementation of the same NAVSIM definition. Three metres
is just under a lane width, so a lane change counts, which is what the
definition asks for.

One detail is forced rather than chosen: that implementation reads the heading
straight off the pose column, and ``route_waypoints`` has no such column -- it
is ``(20, 3)`` of x, y, z with z identically zero on every frame. The heading
change is therefore taken from the polyline's first and last segments, which is
how ``nurec_route`` itself reads a heading off a route.
Deliberately not tuned to reproduce NAVSIM's label distribution: NuRec drives
far more curved road than NAVSIM does, and matching the histogram would mean
calling real turns straight. Measured over the recorded 4 s of driving,

    lateral offset      NAVSIM      NuRec
    median              0.08 m      0.56 m
    p90                 2.47 m      9.36 m
    over 3 m             8.6%       25.6%
    command says L/R    15.5%       35.8%  (this rule)

NAVSIM's own command fires 1.8x as often as its roads exceed three metres; this
rule fires 1.4x as often as NuRec's do. It is the more conservative of the two
relative to the road it is reading, which is the right side to err on when the
command is steering a model that has never seen this domain.
"""
from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

LEFT = np.array([1, 0, 0, 0], dtype=np.int64)
STRAIGHT = np.array([0, 1, 0, 0], dtype=np.int64)
RIGHT = np.array([0, 0, 1, 0], dtype=np.int64)
UNKNOWN = np.array([0, 0, 0, 1], dtype=np.int64)

DEFAULT_LATERAL_M = 3.0
"""Lateral offset over the route that counts as a turn, lane change or curve."""

DEFAULT_YAW_DEG = 12.0
"""Heading change that counts on its own, for turns too tight to travel far."""


def driving_command_from_route(
    waypoints,
    lateral_threshold_m: float = DEFAULT_LATERAL_M,
    yaw_threshold_deg: float = DEFAULT_YAW_DEG,
) -> npt.NDArray[np.int64]:
    """One-hot ``[left, straight, right, unknown]`` for a route polyline.

    :param waypoints: ``[N, >=2]`` route waypoints in the ego rig frame
        (x forward, y left), NaN-padded as ``EgoStatus.route_waypoints`` are.
    :param lateral_threshold_m: offset from the route's own start, at its end,
        above which the manoeuvre is a left or a right.
    :param yaw_threshold_deg: heading change over the route that also decides it,
        OR-ed with the lateral test.
    :return: the one-hot command; ``unknown`` when fewer than two waypoints
        survive the NaN mask, which is what an absent route looks like.
    """
    if waypoints is None:
        return UNKNOWN.copy()
    route = np.asarray(waypoints, dtype=np.float64)
    if route.ndim != 2 or route.shape[0] == 0 or route.shape[1] < 2:
        return UNKNOWN.copy()
    route = route[np.isfinite(route[:, :2]).all(axis=1)][:, :2]
    if len(route) < 2:
        return UNKNOWN.copy()

    lateral = float(route[-1, 1] - route[0, 1])
    # Heading change across the route, start segment to end segment. Reading the
    # final segment alone would measure where the route points rather than how
    # far it has turned, and the route starts well ahead of the car.
    start = route[1] - route[0]
    end = route[-1] - route[-2]
    yaw = math.atan2(end[1], end[0]) - math.atan2(start[1], start[0])
    yaw = (yaw + math.pi) % (2.0 * math.pi) - math.pi
    threshold = math.radians(yaw_threshold_deg)

    left = lateral >= lateral_threshold_m or yaw >= threshold
    right = lateral <= -lateral_threshold_m or yaw <= -threshold

    if left and not right:
        return LEFT.copy()
    if right and not left:
        return RIGHT.copy()
    # Straight, or the two tests disagree -- which is not a manoeuvre either.
    return STRAIGHT.copy()
