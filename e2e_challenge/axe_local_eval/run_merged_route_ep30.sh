#!/usr/bin/env bash
# Evaluate stage3_merged_route_ep30 for the local leaderboard.
#
# The environment is held to what every other subject on that board was scored
# under -- dev preset, the same 441 curated_val scenes, one rollout each, 16
# drivers and 16 renderers on GPUs 4-7, and only the four variables the official
# evaluator supplies. That is what makes the fit comparable; the scene set in
# particular has to match, because ZOIB is fitted across subjects per scene.
#
# The gains are the exception, and deliberately so: lat 1.0 / lon 0.25 / idx 3,
# the best-scoring set from the ep29 controller search. It is worth being
# precise about how strong that evidence is -- one measurement at SE 0.033, an
# improvement of +0.022 over its anchor, so a 95% interval that still contains
# zero. It is the best observed set, not a demonstrated optimum.
#
# The consequence is that a gap between this run and the others mixes two
# causes, the checkpoint and the gains, in the same direction. Only a rerun of
# this checkpoint on 6.0 / 0.5 / 3 separates them.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
AXE="$ROOT/e2e_challenge/axe_local_eval"
LOG="$AXE/merged_route_ep30.log"
RUN_DIR="$ROOT/runs/leaderboard-merged-route-ep30"
IMG=alpasim-e2e-drivesuprim-stage3:merged-route-ep30
log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== merged-route-ep30: starting 16 drivers on GPUs 4-7 ==="
addrs="$(IMAGE="$IMG" \
    EXPECTED_CHECKPOINT_SHA256=364c801e397e9d6e36ea1f9027dc4ca804ad2e429bb588cf4e84185ca851de3d \
    OFFICIAL_ENV_ONLY=1 \
    CONTAINER_PREFIX="axe-lb-mr30" \
    BASE_PORT=6900 \
    GPU_INDICES_CSV=4,5,6,7 REPLICAS_PER_GPU=4 \
    "$AXE/start_drivers.sh" 2>>"$LOG" | tail -1)"
[[ "$addrs" == \[* ]] || { log "drivers failed"; exit 1; }

log "=== merged-route-ep30: 441 scenes x 1 rollout (dev), gains lat=1.0 lon=0.25 idx=3 ==="
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
SERVICE_STARTUP_TIMEOUT_SEC=1800 \
MPC_OVERRIDES="controller.gains.long_position_weight=0.25 controller.gains.lat_position_weight=1.0 controller.gains.idx_start_penalty=3" \
    "$AXE/run_curated_val.sh" >> "$RUN_DIR.wizard.log" 2>&1
log "merged-route-ep30: wizard exit=$?"

docker ps -aq --filter "name=axe-lb-mr30" | xargs -r docker rm -f >/dev/null 2>&1 || true

if [[ -f "$RUN_DIR/aggregate/results-summary.json" ]]; then
    log "merged-route-ep30: OK -> $RUN_DIR/aggregate/results-summary.json"
else
    log "merged-route-ep30: FAILED, no summary"
    exit 1
fi
