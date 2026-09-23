"""Conservative inference-only selection among existing trajectory candidates."""

import torch


@torch.no_grad()
def select_route_candidate(scores, trajectories, route, route_mask, feasibility,
                           scale=80.0, weight=0.0005, margin=0.02,
                           min_overlap=8.0, deadband=1.0, max_distance=8.0,
                           adaptive=False, near_distance=40.0, near_gain=2.0,
                           min_travel=2.0, diagnostics=None):
    # Only scale, weight and min_overlap affect selection; other options are
    # accepted for compatibility with existing configs/checkpoint callers.
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
    # Span of all interior projections, independent of order or lateral distance.
    first = progress.masked_fill(~supported, float('inf')).amin(-1)
    last = progress.masked_fill(~supported, -float('inf')).amax(-1)
    span = torch.where(supported.any(-1), last - first, torch.zeros_like(first))
    step_length = (xy[:, :, 1:] - xy[:, :, :-1]).norm(dim=-1)
    travel = step_length.sum(-1)
    coverage = (span / travel.clamp_min(1e-6)).clamp(0, 1)
    # Actual segment distance, not a fixed 40 m assumption or elapsed time.
    origin_u = (-a * delta).sum(-1) / length.square().clamp_min(1e-8)
    origin_distance = (a + origin_u.clamp(0, 1).unsqueeze(-1) * delta).norm(dim=-1)
    route_start = origin_distance.masked_fill(~segment_valid, float('inf')).min(-1).values
    required = torch.full_like(span, min_overlap)
    comparable = (span >= required) & torch.isfinite(scores)
    # Arithmetic mean Euclidean distance in meters over interior projections.
    costs = torch.where(supported, nearest, torch.zeros_like(nearest)).sum(-1)
    costs = costs / supported.sum(-1).clamp_min(1)
    rows = torch.arange(scores.shape[0], device=scores.device)
    eligible = comparable.clone()
    adjusted = scores.float() - weight * costs
    if diagnostics is not None:
        diagnostics.update(route_start_m=route_start, coverage=coverage,
                           gain=torch.ones_like(route_start),
                           overlap_m=span, required_overlap_m=required)
    selected = adjusted.masked_fill(~eligible, -float('inf')).argmax(-1)
    # Keep the original choice on ties and on unsupported or non-finite rows.
    improve = adjusted[rows, selected] > adjusted[rows, base]
    change = comparable[rows, base] & torch.isfinite(scores).all(-1) & improve
    return torch.where(change, selected, base), comparable, costs
