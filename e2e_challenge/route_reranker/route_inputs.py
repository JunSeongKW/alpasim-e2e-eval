"""Keep the trained far-route encoder separate from full reranker geometry."""
import numpy as np

from navsim.common.route_contract import ROUTE_SLOTS, ROUTE_SPACING_M


def split_route_inputs(waypoints, capacity=64):
    """Return fixed encoder slots plus all supplied geometry, without gap filling.

    Input points are ordered in current ego coordinates. The first-point distance
    anchors arclength when the provided polyline does not yet reach the ego.
    The reranker always receives the original points, not resampled geometry.
    """
    xy = np.asarray(waypoints, dtype=np.float32)
    if xy.ndim != 2 or xy.shape[1] != 2 or len(xy) > capacity:
        raise ValueError(f'Expected ordered route [N,2] with N <= {capacity}')
    valid = np.isfinite(xy).all(-1)
    full = np.full((capacity, 2), np.nan, dtype=np.float32)
    full[:len(xy)] = xy
    # A 20-slot message already satisfies the trained encoder contract. Do not
    # guess whether a curved route is "near" from Euclidean radius and rewrite it.
    if len(xy) == ROUTE_SLOTS:
        return xy.copy(), full
    encoder = np.full((ROUTE_SLOTS, 2), np.nan, dtype=np.float32)
    if not valid.any():
        return encoder, full
    start = int(np.flatnonzero(valid)[0])
    end = start
    while end < len(xy) and valid[end]:
        end += 1
    points = xy[start:end]
    if len(points) < 2:
        return encoder, full
    distances = np.linalg.norm(np.diff(points, axis=0), axis=-1)
    # Degenerate/disconnected sections are not interpolated into fake evidence.
    bad = np.flatnonzero((distances < 1e-4) | (distances > 10.0))
    if len(bad):
        points = points[:bad[0] + 1]
    if len(points) < 2:
        return encoder, full
    arc = np.r_[0., np.linalg.norm(np.diff(points, axis=0), axis=-1).cumsum()]
    # Locate ego on the available polyline; no extrapolation before its start.
    delta = np.diff(points, axis=0)
    u = np.clip((-points[:-1] * delta).sum(-1) / (delta * delta).sum(-1), 0, 1)
    projected = points[:-1] + u[:, None] * delta
    i = int(np.argmin(np.linalg.norm(projected, axis=-1)))
    if arc[i] + u[i] * np.linalg.norm(delta[i]) > 1e-4 and np.linalg.norm(projected[i]) < 5:
        arc = arc - (arc[i] + u[i] * np.linalg.norm(delta[i]))
    else:
        arc = arc + np.linalg.norm(points[0])
    targets = np.arange(10, 20) * ROUTE_SPACING_M
    inside = (targets >= arc[0]) & (targets <= arc[-1] + 1e-4)
    for k, t in enumerate(targets):
        if inside[k]:
            encoder[k] = [np.interp(t, arc, points[:, axis]) for axis in range(2)]
    return encoder, full
