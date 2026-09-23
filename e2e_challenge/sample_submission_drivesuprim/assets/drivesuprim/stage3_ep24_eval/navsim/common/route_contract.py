"""Shape of the PAI-Track route message.

Kept dependency-free and separate from ``nurec_route`` so that core dataclasses
do not have to import the NuRec data-preparation module: that module is edited
while long scoring jobs are running, and those jobs import the dataclasses.
"""

ROUTE_SLOTS = 20
ROUTE_HORIZON_M = 80.0
# The Runtime spaces the slots evenly over the horizon: 80 / 19 m apart along
# the route arclength.
ROUTE_SPACING_M = ROUTE_HORIZON_M / (ROUTE_SLOTS - 1)
# Slots nearer than this are dropped; the survivors compact to the front.
# Measured along the sampled waypoints, not along the arclength that produced
# them: on a curve the chord accumulates more slowly, the cutoff lands one slot
# later, and the message carries nine points instead of ten.
ROUTE_NEAR_CUTOFF_M = 40.0
# A resampled gap outside this band means the route is no longer trustworthy,
# and every waypoint after it is dropped. The upper bound cannot trip on an
# arclength resample (a chord never exceeds its arc) but is part of the
# contract, so it is checked rather than assumed.
MIN_WAYPOINT_GAP_M = 3.5
MAX_WAYPOINT_GAP_M = 4.5
