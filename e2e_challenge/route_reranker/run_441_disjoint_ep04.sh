#!/usr/bin/env bash
# stage3_disjoint_ep04 on the 441 set, at gamma=0.02, under the axe-v9 contract.
#
# This is run_441_aug_ep04.sh with the weights swapped, so the two ep04
# checkpoints are measured the same way and can be read against each other.
# Fixed: the runtime image is the axe-v9 submission base with ONLY its weights
# replaced (Dockerfile.checkpoint_overlay, image built by chain_ckpt_queue.sh and
# already label-verified), dev preset / 441 curated_val / one rollout / 16
# drivers / 16 renderers / 16 workers, gains lat 1.0, lon 0.25, idx 3, no video.
# Varied: the checkpoint. The reranker is on at gamma 0.02, the value the tuning
# sweep chose, which is also what the axe-v9 and aug-ep04 reranker runs used.
#
# Comparisons this makes possible:
#   runs/leaderboard-disjoint-ep04                 same weights, NO reranker
#   runs/rr441-g0p02-augep04-cache-centre-max      aug ep04, gamma 0.02
#   runs/rr441-g0p02-cache-centre-max              axe-v9 weights, gamma 0.02
#   runs/leaderboard-merged-route-ep30             axe-v9 weights, no reranker
#
# The first of those is the paired one that matters: same checkpoint, same
# contract, same worker count, reranker the only difference. Concurrency moves
# gRPC timing enough to flip clip verdicts, so only a 16-worker run is a fair
# reference for a 16-worker run.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/e2e_challenge/route_reranker"
cd "$ROOT"
GAMMA=0.02 \
IMG=alpasim-e2e-axe-v9:disjoint-ep04 \
CKPT_SHA=3ce4eb8815ccaef163194a244a4afec8881ef56ea85e69aa66c12edc4c1f8ee4 \
TAG_SUFFIX=-disjointep04 \
    "$HERE/run_441_reranker.sh"
