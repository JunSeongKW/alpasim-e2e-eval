#!/usr/bin/env bash
# Evaluate stage3_epzero_ep29 in the best configuration found so far.
#
# Two things make it "best", and they come from different experiments. The ego
# footprint comes from the image, which is ep30's with only the weights swapped:
# DRIVESUPRIM_EGO_FOOTPRINT_FROM_API plus the rig->box centre offset, the pair
# that put ep30 and axe-v8 at the top of the local board. The gains come from
# the controller search, which scored 27 combinations on a stratified subset and
# put lat 3.0 / lon 0.25 / idx 3 at 0.7081 against 0.6635 for the 6.0 / 0.5 / 3
# that every earlier leaderboard run used.
#
# That gain change is worth stating plainly, because it makes this subject the
# only one on the board not scored under 6.0 / 0.5 / 3. A gap against ep30 will
# therefore mix two causes -- the checkpoint and the gains -- and the control
# that separates them is ep30 rerun on these gains, not this run alone.
#
# Everything else is held to the ep30 recipe exactly: dev preset, 441 curated_val
# scenes x 1 rollout, 16 drivers and 16 renderers on GPUs 4-7, official env only.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
AXE="$ROOT/e2e_challenge/axe_local_eval"
LOG="$AXE/epzero_ep29.log"
RUN_DIR="$ROOT/runs/leaderboard-epzero-ep29"
IMG=alpasim-e2e-drivesuprim-stage3:epzero-ep29
log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== epzero-ep29: starting 16 drivers on GPUs 4-7 ==="
addrs="$(IMAGE="$IMG" \
    EXPECTED_CHECKPOINT_SHA256=07ab5a40027c9b0960765bd2fc9a6297190fd41038c652ebc031c8e52aaa94b4 \
    OFFICIAL_ENV_ONLY=1 \
    CONTAINER_PREFIX="axe-lb-ep29" \
    BASE_PORT=6900 \
    GPU_INDICES_CSV=4,5,6,7 REPLICAS_PER_GPU=4 \
    "$AXE/start_drivers.sh" 2>>"$LOG" | tail -1)"
[[ "$addrs" == \[* ]] || { log "drivers failed"; exit 1; }

log "=== epzero-ep29: 441 scenes x 1 rollout (dev), gains lat=3.0 lon=0.25 idx=3 ==="
RUN_DIR="$RUN_DIR" \
PRESET=dev \
CONTESTANT_IMAGE="$IMG" \
DRIVER_ADDRESSES="$addrs" \
N_ROLLOUTS=1 \
ROLLOUT_WORKERS=16 \
RENDER_GPUS_CSV=4,5,6,7 \
RENDERER_REPLICAS_PER_GPU=4 \
NRE_CACHE_SIZE=1 \
ENABLE_AUTORESUME=true \
MPC_OVERRIDES="controller.gains.long_position_weight=0.25 controller.gains.lat_position_weight=3.0 controller.gains.idx_start_penalty=3" \
    "$AXE/run_curated_val.sh" >> "$RUN_DIR.wizard.log" 2>&1
log "epzero-ep29: wizard exit=$?"

docker ps -aq --filter "name=axe-lb-ep29" | xargs -r docker rm -f >/dev/null 2>&1 || true

if [[ -f "$RUN_DIR/aggregate/results-summary.json" ]]; then
    log "epzero-ep29: OK -> $RUN_DIR/aggregate/results-summary.json"
else
    log "epzero-ep29: FAILED, no summary"
    exit 1
fi
