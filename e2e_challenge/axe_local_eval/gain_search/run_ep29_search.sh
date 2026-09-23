#!/usr/bin/env bash
# Continue the controller-gain search against stage3_epzero_ep29, the checkpoint
# that leads the local board, optimising average scene score.
#
# Three things differ from the ep24 search this reuses.
#
# The anchor moves to lat 3.0 / lon 0.25 / idx 3, ep29's current best. The screen
# phase measures one-factor deviations *from the anchor*, so leaving it at the
# stock 6.0 / 0.5 / 3 would spend the first lap re-walking ground already
# covered on the other checkpoint.
#
# The results file is separate. search.py derives the incumbent, the screen plan
# and the explore seed from that one file, so mixing two checkpoints' runs would
# have each proposing points chosen by the other's landscape.
#
# The preset is dev, not the ec2 the ep24 search used. That costs roughly 2.8x
# in wall clock (1.9 vs 5.4 rollout/min), and it is bought deliberately: the
# number being optimised is the average scene score on the local leaderboard,
# and every subject there was scored under dev. Tuning under ec2 would optimise
# a surface that is measured nowhere.
#
# Stop with: touch <this dir>/STOP    (finishes the run in flight first)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../../.." && pwd)"
AXE="$ROOT/e2e_challenge/axe_local_eval"
IMG=alpasim-e2e-drivesuprim-stage3:epzero-ep29
OUT="$HERE/ep29"
mkdir -p "$OUT"
cd "$ROOT"

log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$OUT/loop.log"; }

rm -f "$HERE/STOP"

log "=== starting 16 drivers (ep29) on GPUs 4-7 ==="
addrs="$(IMAGE="$IMG" \
    EXPECTED_CHECKPOINT_SHA256=07ab5a40027c9b0960765bd2fc9a6297190fd41038c652ebc031c8e52aaa94b4 \
    OFFICIAL_ENV_ONLY=1 \
    CONTAINER_PREFIX="axe-gs-ep29" \
    BASE_PORT=6900 \
    GPU_INDICES_CSV=4,5,6,7 REPLICAS_PER_GPU=4 \
    "$AXE/start_drivers.sh" 2>>"$OUT/loop.log" | tail -1)"
[[ "$addrs" == \[* ]] || { log "drivers failed to start"; exit 1; }
printf '%s\n' "$addrs" > "$OUT/driver_addresses.txt"
log "drivers up: $addrs"

log "=== search: anchor lat=3.0 lon=0.25 idx=3, objective = avg scene score ==="
# A cold renderer cache has taken just over the default 15-minute service
# deadline on this host.  The evaluation itself is healthy once the renderers
# are serving, so allow cold starts to finish instead of recording a gain set
# as a scoring failure before its first clip.
GAIN_RESULTS="$OUT/results.jsonl" \
GAIN_LOG="$OUT/loop.log" \
GAIN_ANCHOR='{"lat_position_weight": 3.0, "long_position_weight": 0.25, "idx_start_penalty": 3}' \
DRIVER_ADDRESSES="$addrs" \
CONTESTANT_IMAGE="$IMG" \
PRESET=dev \
SUBSET=search \
N_ROLLOUTS=1 \
RENDER_GPUS_CSV=4,5,6,7 \
SERVICE_STARTUP_TIMEOUT_SEC=1800 \
RUNS_ROOT="$ROOT/runs/gain-search-ep29" \
MIN_FREE_GB=200 \
    "$HERE/loop.sh"
rc=$?
log "loop exited rc=$rc"

docker ps -aq --filter 'name=axe-gs-ep29' | xargs -r docker rm -f >/dev/null 2>&1 || true
log "drivers torn down"
