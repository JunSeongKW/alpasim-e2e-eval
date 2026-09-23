import numpy as np
import pytest
import torch

from navsim.agents.drivesuprim.drivesuprim_model import RouteEncoder
from navsim.common.route_contract import ROUTE_SLOTS


def _route(num_valid: int, slots: int = ROUTE_SLOTS, value: float = 0.5):
    route = torch.zeros(1, slots, 2)
    mask = torch.zeros(1, slots)
    route[0, :num_valid, 0] = value
    mask[0, :num_valid] = 1.0
    return route, mask


def test_route_length_changes_the_encoding():
    # The encoder flattens rather than pools, so this is the OPPOSITE of what the
    # pooled version guaranteed. The pooled version normalised by the valid count
    # so that ten waypoints and three encoded alike; that threw away how far the
    # route is known, which is real information -- a short route means the near
    # cutoff ate most of it or the Runtime lost the rest. route_contract compacts
    # survivors to the front, so the count is exactly the prefix length.
    encoder = RouteEncoder(d_model=16, d_hidden=8).eval()
    with torch.no_grad():
        ten = encoder(*_route(10))
        three = encoder(*_route(3))
    assert not torch.allclose(ten, three)


def test_waypoint_order_changes_the_encoding():
    # The point of flattening: a route that goes straight then turns left is not
    # the same route as one that turns left then goes straight. A mean/max pool
    # over a shared per-point MLP is permutation invariant and cannot tell them
    # apart from the pooled statistics alone.
    torch.manual_seed(0)
    encoder = RouteEncoder(d_model=16, d_hidden=8).eval()
    mask = torch.zeros(1, ROUTE_SLOTS)
    mask[0, :10] = 1.0
    forward = torch.arange(10, 20, dtype=torch.float32) * (80.0 / 19) / 80.0

    ordered = torch.zeros(1, ROUTE_SLOTS, 2)
    ordered[0, :10, 0] = forward
    ordered[0, :10, 1] = torch.linspace(0.0, 0.3, 10)   # bends left with distance

    reversed_ = ordered.clone()
    reversed_[0, :10] = ordered[0, :10].flip(0)          # same point set, other order

    with torch.no_grad():
        assert not torch.allclose(encoder(ordered, mask), encoder(reversed_, mask))


def test_a_route_with_no_valid_waypoint_stays_finite():
    # About 0.5% of NuRec frames have no route at all. Coordinates and mask are
    # both all-zero there, which the flatten path handles without a special case
    # -- unlike the max-pool it replaced, which produced -inf over an empty set.
    encoder = RouteEncoder(d_model=16, d_hidden=8).eval()
    with torch.no_grad():
        encoded = encoder(*_route(0))
    assert torch.isfinite(encoded).all()


def test_no_route_is_distinguishable_from_a_route_through_the_origin():
    # Padding is zeroed upstream, so an absent waypoint and a waypoint at the ego
    # origin have identical coordinates. Only the mask separates them.
    encoder = RouteEncoder(d_model=16, d_hidden=8).eval()
    zeros_valid = (torch.zeros(1, ROUTE_SLOTS, 2), torch.ones(1, ROUTE_SLOTS))
    with torch.no_grad():
        assert not torch.allclose(encoder(*zeros_valid), encoder(*_route(0)))


def test_gradients_stay_finite_for_a_mixed_batch():
    encoder = RouteEncoder(d_model=16, d_hidden=8)
    route = torch.zeros(3, ROUTE_SLOTS, 2, requires_grad=True)
    mask = torch.zeros(3, ROUTE_SLOTS)
    mask[0, :10] = 1.0      # full route
    mask[1, :4] = 1.0       # partial
    mask[2] = 0.0           # none

    encoder(route, mask).sum().backward()

    assert torch.isfinite(route.grad).all()
    for parameter in encoder.parameters():
        assert torch.isfinite(parameter.grad).all()


def test_route_shape_changes_the_encoding():
    # A left-bending and a right-bending route must not collapse to the same
    # token, or the intent signal carries nothing. Seeded because a narrow
    # d_hidden can be initialised with every unit dead for this y-range, which
    # says nothing about the encoder and makes the test flaky.
    torch.manual_seed(0)
    encoder = RouteEncoder(d_model=16, d_hidden=8).eval()
    mask = torch.zeros(1, ROUTE_SLOTS)
    mask[0, :10] = 1.0
    forward = torch.arange(10, 20, dtype=torch.float32) * (80.0 / 19) / 80.0
    left, right = torch.zeros(1, ROUTE_SLOTS, 2), torch.zeros(1, ROUTE_SLOTS, 2)
    left[0, :10, 0] = right[0, :10, 0] = forward
    left[0, :10, 1] = 0.3
    right[0, :10, 1] = -0.3

    with torch.no_grad():
        assert not torch.allclose(encoder(left, mask), encoder(right, mask))


def test_a_slot_count_that_does_not_match_the_contract_is_rejected():
    # Flattening binds the slot count into the weights. A Runtime that changed
    # ROUTE_SLOTS would otherwise misalign every slot silently.
    encoder = RouteEncoder(d_model=16, d_hidden=8).eval()
    with pytest.raises(ValueError, match="20 slots"):
        encoder(torch.zeros(1, 19, 2), torch.zeros(1, 19))
