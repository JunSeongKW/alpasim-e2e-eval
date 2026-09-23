#!/usr/bin/env bash
# Evaluate axe-v5 on GPUs 0-3 while axe-v4 occupies GPUs 4-7.
#
# Same contract as run_v4_v5_leaderboard.sh: dev preset, 441 scenes x 1 rollout,
# readiness on "driver listening" because v5 loads its policy lazily at the
# first session.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
AXE="$ROOT/e2e_challenge/axe_local_eval"
ECR=696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe
LOG="$AXE/v5.log"
RUN_DIR="$ROOT/runs/leaderboard-axe-v5"
log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== axe-v5: starting 16 drivers on GPUs 0-3 ==="
addrs="$(IMAGE="$ECR:axe-v5" \
    EXPECTED_CHECKPOINT_SHA256= \
    OFFICIAL_ENV_ONLY=1 \
    READY_MARKER="driver listening" \
    CONTAINER_PREFIX="axe-lb-axe-v5" \
    BASE_PORT=6840 \
    GPU_INDICES_CSV=0,1,2,3 REPLICAS_PER_GPU=4 \
    "$AXE/start_drivers.sh" 2>>"$LOG" | tail -1)"
[[ "$addrs" == \[* ]] || { log "drivers failed"; exit 1; }

log "=== axe-v5: evaluating 441 scenes x 1 rollout (dev, GPUs 0-3) ==="
RUN_DIR="$RUN_DIR" \
PRESET=dev \
CONTESTANT_IMAGE="$ECR:axe-v5" \
DRIVER_ADDRESSES="$addrs" \
N_ROLLOUTS=1 \
ROLLOUT_WORKERS=16 \
RENDER_GPUS_CSV=0,1,2,3 \
RENDERER_REPLICAS_PER_GPU=4 \
NRE_CACHE_SIZE=1 \
ENABLE_AUTORESUME=true \
    "$AXE/run_curated_val.sh" >> "$RUN_DIR.wizard.log" 2>&1
log "axe-v5: wizard exit=$?"
docker rm -f $(docker ps -aq --filter "name=axe-lb-axe-v5") >/dev/null 2>&1 || true
[[ -f "$RUN_DIR/aggregate/results-summary.json" ]] && log "axe-v5: OK" || log "axe-v5: FAILED"
