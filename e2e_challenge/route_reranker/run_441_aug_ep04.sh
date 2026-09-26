#!/usr/bin/env bash
# stage3_aug_ep04 on the 441 set, at gamma=0.02, under the axe-v9 contract.
#
# Two things are being held fixed and one is being varied. Fixed: the runtime
# image is the axe-v9 submission base with ONLY its weights replaced
# (Dockerfile.checkpoint_overlay), the driver gets only the four variables the
# official evaluation hands a contestant container (OFFICIAL_ENV_ONLY), and the
# wizard runs dev preset / 441 curated_val / one rollout / 16 drivers / 16
# renderers / 16 workers with gains lat 1.0, lon 0.25, idx 3. Varied: the
# checkpoint. The reranker is on at the gamma that won the tuning sweep, the
# same as the axe-v9 441 run it will be read against.
#
# Comparisons this makes possible:
#   runs/rr441-g0p02-cache-centre-max         axe-v9 weights, gamma 0.02
#   runs/rr441-g0p02-augep04-cache-centre-max stage3_aug_ep04, gamma 0.02  <- this
#   runs/leaderboard-merged-route-ep30        axe-v9 weights, no reranker
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/e2e_challenge/route_reranker"
cd "$ROOT"
GAMMA=0.02 \
IMG=alpasim-e2e-axe-v9:stage3-aug-ep04 \
CKPT_SHA=7bab880de228960a87e2ae1d4027fdbe7bcae86d6ef7b96dc0ba8d0c9c9fd63b \
TAG_SUFFIX=-augep04 \
    "$HERE/run_441_reranker.sh"
