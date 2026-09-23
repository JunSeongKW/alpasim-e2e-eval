#!/usr/bin/env bash
# The control for run_epzero_ep29.sh: same checkpoint, the gains everyone else
# on the board was scored with.
#
# The tuned run uses lat 3.0 / lon 0.25, which the controller search preferred on
# a stratified subset -- but that search was run against ep24's weights, so it is
# not established that the preference carries to this checkpoint. With only the
# tuned run, a gap against ep30 has two candidate causes and no way to separate
# them. Holding the checkpoint fixed and moving only the gains back to
# 6.0 / 0.5 / 3 makes the pair a clean two-point comparison in each direction:
# against ep30 it isolates the weights, against the tuned run it isolates the
# gains.
#
# Everything else is identical to the tuned run, including the ego footprint the
# image carries and the 441-scene dev preset, so the only free variable is the
# three controller numbers.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
AXE="$ROOT/e2e_challenge/axe_local_eval"
LOG="$AXE/epzero_ep29_basegains.log"
RUN_DIR="$ROOT/runs/leaderboard-epzero-ep29-basegains"
IMG=alpasim-e2e-drivesuprim-stage3:epzero-ep29
log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== ep29-basegains: starting 16 drivers on GPUs 4-7 ==="
addrs="$(IMAGE="$IMG" \
    EXPECTED_CHECKPOINT_SHA256=07ab5a40027c9b0960765bd2fc9a6297190fd41038c652ebc031c8e52aaa94b4 \
    OFFICIAL_ENV_ONLY=1 \
    CONTAINER_PREFIX="axe-lb-ep29b" \
    BASE_PORT=6900 \
    GPU_INDICES_CSV=4,5,6,7 REPLICAS_PER_GPU=4 \
    "$AXE/start_drivers.sh" 2>>"$LOG" | tail -1)"
[[ "$addrs" == \[* ]] || { log "drivers failed"; exit 1; }

log "=== ep29-basegains: 441 scenes x 1 rollout (dev), gains lat=6.0 lon=0.5 idx=3 ==="
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
MPC_OVERRIDES="controller.gains.long_position_weight=0.5 controller.gains.lat_position_weight=6.0 controller.gains.idx_start_penalty=3" \
    "$AXE/run_curated_val.sh" >> "$RUN_DIR.wizard.log" 2>&1
log "ep29-basegains: wizard exit=$?"

docker ps -aq --filter "name=axe-lb-ep29b" | xargs -r docker rm -f >/dev/null 2>&1 || true

if [[ -f "$RUN_DIR/aggregate/results-summary.json" ]]; then
    log "ep29-basegains: OK -> $RUN_DIR/aggregate/results-summary.json"
else
    log "ep29-basegains: FAILED, no summary"
    exit 1
fi
