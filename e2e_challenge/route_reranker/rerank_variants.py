"""Two variants of the bundle's mean-L2 route reranker, for a 2x2 comparison.

`route_reranker.py` is the 2026-09-23 bundle, byte for byte, and stays that way:
it measures from the trajectory POSE and averages the distance over the four
seconds. Two of its choices are worth testing against the thing actually being
scored, and both are one line each, so they live here rather than as edits to
the bundle.

* **Where the point is.** A vocab pose is the rig origin -- the mid bottom rear
  edge of the box -- while the scorer's `left_corridor_laterally` measures the
  BODY CENTRE, about 1.47 m ahead along the heading. On a straight route the
  offset slides the projection along the route and barely moves the lateral
  distance; through a turn it does not.
* **How time is aggregated.** The scorer fails a rollout if the corridor is left
  at ANY moment, so its aggregation is a maximum. A mean lets a trajectory swing
  wide and come back for almost no penalty, which is precisely the failure the
  gate exists to catch.

Nothing else differs: the same interior-projection rule, the same 8 m span gate
on both the incumbent and the challenger, the same `score - gamma * cost`, and
the same "strictly better, ties keep the incumbent" replacement.
"""

from __future__ import annotations

import torch

# The bundle's file is mounted inside the navsim package, not next to this one,
# so it has to be imported by its package path. Importing it as a bare module
# works in the source tree and fails in the container, which is where it ran.
from navsim.agents.drivesuprim.route_reranker import select_route_candidate


def shift_to_body_centre(trajectories: torch.Tensor, dx: float) -> torch.Tensor:
    """Slide each pose `dx` forward along its own heading.

    `trajectories` is [B, K, T, >=3] with heading in channel 2, which is what the
    refinement stage carries. Returns a copy; the caller still hands the ORIGINAL
    trajectory to the controller, because only the measurement moves, not the
    plan.
    """
    if not dx:
        return trajectories
    out = trajectories.clone()
    heading = trajectories[..., 2]
    out[..., 0] = trajectories[..., 0] + dx * torch.cos(heading)
    out[..., 1] = trajectories[..., 1] + dx * torch.sin(heading)
    return out


@torch.no_grad()
def select_route_candidate_max(scores, trajectories, route, route_mask, feasibility,
                               scale=80.0, weight=0.0005, min_overlap=8.0, **legacy):
    """`select_route_candidate` with the cost as a maximum instead of a mean.

    Line for line the bundle's function; the only change is the two lines that
    aggregate `nearest` over time, marked below. Kept as a copy rather than a
    flag inside the bundle file so that file stays byte-identical to the archive.
    """
    base = scores.argmax(dim=1)
    comparable = torch.zeros_like(scores, dtype=torch.bool)
    costs = torch.zeros_like(scores, dtype=torch.float32)
    if route is None or route_mask is None or route.shape[1] < 2 or weight == 0:
        return base, comparable, costs
    if weight < 0 or min_overlap <= 0:
        raise ValueError("Invalid route reranker thresholds")

    points = route.float() * scale
    valid = route_mask.bool() & torch.isfinite(points).all(dim=-1)
    points = torch.nan_to_num(points)
    a, delta = points[:, :-1], points[:, 1:] - points[:, :-1]
    length = delta.norm(dim=-1)
    segment_valid = valid[:, :-1] & valid[:, 1:] & (length > 1e-4)
    xy = trajectories[..., :2].float()
    offset = xy.unsqueeze(-2) - a[:, None, None]
    u = (offset * delta[:, None, None]).sum(-1) / length.square().clamp_min(1e-8)[:, None, None]
    distance = (offset - u.unsqueeze(-1) * delta[:, None, None]).norm(dim=-1)
    interior = (u >= 0) & (u <= 1) & segment_valid[:, None, None]
    nearest, index = distance.masked_fill(~interior, float('inf')).min(-1)
    cumulative = torch.cat([length.new_zeros(length.shape[0], 1), length.cumsum(-1)], -1)[:, :-1]
    progress = cumulative[:, None, None] + u * length[:, None, None]
    progress = progress.gather(-1, index.unsqueeze(-1)).squeeze(-1)
    supported = torch.isfinite(nearest) & torch.isfinite(xy).all(-1)
    first = progress.masked_fill(~supported, float('inf')).amin(-1)
    last = progress.masked_fill(~supported, -float('inf')).amax(-1)
    span = torch.where(supported.any(-1), last - first, torch.zeros_like(first))
    required = torch.full_like(span, min_overlap)
    comparable = (span >= required) & torch.isfinite(scores)
    # --- the only change from the bundle: worst moment, not the average one ---
    costs = torch.where(supported, nearest, torch.zeros_like(nearest)).amax(-1)
    # -------------------------------------------------------------------------
    rows = torch.arange(scores.shape[0], device=scores.device)
    adjusted = scores.float() - weight * costs
    selected = adjusted.masked_fill(~comparable, -float('inf')).argmax(-1)
    improve = adjusted[rows, selected] > adjusted[rows, base]
    change = comparable[rows, base] & torch.isfinite(scores).all(-1) & improve
    return torch.where(change, selected, base), comparable, costs


def select(aggregate: str, centre_dx: float, scores, trajectories, route, route_mask,
           *, scale, weight, min_overlap, **legacy):
    """Dispatch: the bundle's function unless a variant is asked for."""
    trajectories = shift_to_body_centre(trajectories, centre_dx)
    if aggregate == "max":
        return select_route_candidate_max(
            scores, trajectories, route, route_mask, None,
            scale=scale, weight=weight, min_overlap=min_overlap)
    return select_route_candidate(
        scores, trajectories, route, route_mask, None,
        scale=scale, weight=weight, min_overlap=min_overlap, **legacy)
