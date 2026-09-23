#!/usr/bin/env bash
# Evaluate the stage3 ep30 checkpoint in the configuration that scored best
# locally: ego footprint from the API including the rig->box offset, and the
# tuned MPC gains that run_curated_val.sh applies by default.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
AXE="$ROOT/e2e_challenge/axe_local_eval"
LOG="$AXE/ep30.log"
RUN_DIR="$ROOT/runs/leaderboard-stage3-ep30"
IMG=alpasim-e2e-drivesuprim-stage3:ep30
log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== ep30: starting 16 drivers on GPUs 4-7 ==="
addrs="$(IMAGE="$IMG" \
    EXPECTED_CHECKPOINT_SHA256=28af8090108222c6e05d54f92195c2c270a3aa8725814dff90dd36885cd0cf64 \
    OFFICIAL_ENV_ONLY=1 \
    CONTAINER_PREFIX="axe-lb-ep30" \
    BASE_PORT=6900 \
    GPU_INDICES_CSV=4,5,6,7 REPLICAS_PER_GPU=4 \
    "$AXE/start_drivers.sh" 2>>"$LOG" | tail -1)"
[[ "$addrs" == \[* ]] || { log "drivers failed"; exit 1; }

log "=== ep30: evaluating 441 scenes x 1 rollout (dev, GPUs 4-7) ==="
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
    "$AXE/run_curated_val.sh" >> "$RUN_DIR.wizard.log" 2>&1
log "ep30: wizard exit=$?"
docker rm -f $(docker ps -aq --filter "name=axe-lb-ep30") >/dev/null 2>&1 || true
[[ -f "$RUN_DIR/aggregate/results-summary.json" ]] && log "ep30: OK" || log "ep30: FAILED"
