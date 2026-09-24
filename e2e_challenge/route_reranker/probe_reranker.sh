#!/usr/bin/env bash
# Prove the reranker code path runs inside a started driver container.
#
# Executes one synthetic forward of the selection function through the SAME
# python, mounts and environment the driver uses, so an import that only fails
# in the container (the /app vs package-path mistake) surfaces here and not two
# hours later in a log nobody was reading. Exit 0 only if a decision is made.
set -euo pipefail
name="${1:?driver container name}"
docker exec "$name" python -c '
import os, sys, torch
sys.path.insert(0, "/app")
import rerank_variants as V
from navsim.agents.drivesuprim.drivesuprim_config import DriveSuprimConfig
cfg = DriveSuprimConfig()
agg = os.environ.get("DRIVESUPRIM_ROUTE_RERANK_AGG", "mean")
dx = float(os.environ.get("DRIVESUPRIM_ROUTE_RERANK_CENTRE_DX", "0"))
w = float(os.environ.get("DRIVESUPRIM_ROUTE_RERANK_WEIGHT", "0.0005"))
# a near-field route (cache-like) and three candidates; something must qualify
s = torch.arange(-20.0, 90.0, 80.0 / 19.0)
route = torch.stack([s, torch.zeros_like(s)], -1)
R = torch.zeros(1, 64, 2); R[0, :len(s)] = route / 80.0
M = torch.zeros(1, 64); M[0, :len(s)] = 1
t = torch.linspace(0.1, 4.0, 40)
trajs = torch.stack([torch.stack([6*t, 0*t, 0*t], -1), torch.stack([6*t, 2+0*t, 0*t], -1), torch.stack([6*t, 5+0*t, 0*t], -1)])[None]
scores = torch.tensor([[0.990, 1.000, 0.995]])
# gamma == 0 is the reference arm and switches the selector off inside the
# archived selector, so probe the machinery at a non-zero weight and report
# the value the run will use. Probing at 0 would assert on an empty result.
probe_w = w if w > 0 else 0.0005
sel, comp, cost = V.select(agg, dx, scores, trajs, R, M, scale=80.0, weight=probe_w, min_overlap=8.0)
assert int(comp.sum()) == 3, f"eligibility broken: {comp}"

# The accumulated route is spliced once per step, so its near field carries a
# point every half metre and the polyline outgrows the 64-slot channel after
# about a minute of driving. That threw on every frame of a whole sweep while
# the driver quietly kept the previous plan, so the probe drives the real
# adapter with a route of that density and length, not just a short one.
if os.environ.get("DRIVESUPRIM_ROUTE_RERANK_CACHE") == "1":
    import numpy as np
    from drivesuprim_challenge.policy import _resample_route
    from navsim.agents.drivesuprim.route_inputs import split_route_inputs
    dense = np.stack([np.arange(-60.0, 90.0, 0.4),
                      np.zeros(len(np.arange(-60.0, 90.0, 0.4)))], 1).astype(np.float32)
    thin = _resample_route(dense)
    _, full = split_route_inputs(thin)          # raises if the adapter would throw
    print(f"CACHE PROBE OK dense={len(dense)} -> resampled={len(thin)} slots={len(full)}")

print(f"RERANK PROBE OK agg={agg} dx={dx} gamma={w} probed_at={probe_w} sel={int(sel)} costs={[round(float(c),2) for c in cost[0]]}")
'
