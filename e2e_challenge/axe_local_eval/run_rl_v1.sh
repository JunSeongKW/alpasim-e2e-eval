#!/usr/bin/env bash
# Evaluate the RL (ARFM/GRPO) submission under exactly the conditions used for
# stage3_new_ep30: dev preset, 441 scenes x 1 rollout, the tuned MPC gains, and
# the ego footprint applied -- the last of which had to be ported into the RL
# image first, since its driver predates that work.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
AXE="$ROOT/e2e_challenge/axe_local_eval"
LOG="$AXE/rl_v1.log"
RUN_DIR="$ROOT/runs/leaderboard-rl-v1"
IMG=alpasim-e2e-junhyeok-nurec-stage3:ep24-rl-v1-egofp
log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== rl-v1: starting 16 drivers on GPUs 4-7 ==="
addrs="$(IMAGE="$IMG" \
    EXPECTED_CHECKPOINT_SHA256= \
    OFFICIAL_ENV_ONLY=1 \
    CONTAINER_PREFIX=axe-lb-rlv1 \
    BASE_PORT=6900 \
    GPU_INDICES_CSV=4,5,6,7 REPLICAS_PER_GPU=4 \
    "$AXE/start_drivers.sh" 2>>"$LOG" | tail -1)"
[[ "$addrs" == \[* ]] || { log "drivers failed"; exit 1; }

log "=== rl-v1: evaluating 441 scenes x 1 rollout (dev, GPUs 4-7) ==="
RUN_DIR="$RUN_DIR" PRESET=dev CONTESTANT_IMAGE="$IMG" DRIVER_ADDRESSES="$addrs" \
N_ROLLOUTS=1 ROLLOUT_WORKERS=16 RENDER_GPUS_CSV=4,5,6,7 RENDERER_REPLICAS_PER_GPU=4 \
NRE_CACHE_SIZE=1 ENABLE_AUTORESUME=true \
    "$AXE/run_curated_val.sh" >> "$RUN_DIR.wizard.log" 2>&1
log "rl-v1: wizard exit=$?"
docker rm -f $(docker ps -aq --filter 'name=axe-lb-rlv1') >/dev/null 2>&1 || true
[[ -f "$RUN_DIR/aggregate/results-summary.json" ]] && log "rl-v1: OK" || log "rl-v1: FAILED"
