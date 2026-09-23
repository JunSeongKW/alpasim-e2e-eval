#!/usr/bin/env bash
# Evaluate flatvits_planonly_nurec under the same harness as stage3_new_ep30:
# dev preset, 441 scenes x 1 rollout, the tuned MPC gains, GPUs 4-7.
#
# No ego footprint is ported in, and that is not an omission: this model has
# feasibility_enabled=false and no BEV segmentation head, so no trajectory gate
# exists for the ego box to feed. All 4,096 candidates reach the ranking.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
AXE="$ROOT/e2e_challenge/axe_local_eval"
LOG="$AXE/flatvits.log"
RUN_DIR="$ROOT/runs/leaderboard-flatvits-ep25"
IMG=alpasim-e2e-flatvits-planonly:ep25
log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== flatvits: starting 16 drivers on GPUs 4-7 ==="
addrs="$(IMAGE="$IMG" \
    EXPECTED_CHECKPOINT_SHA256=4ccd30c047689b3e4a7224f1814f1a7b13dd9218f842da643f6e1ac9fa7fbe78 \
    OFFICIAL_ENV_ONLY=1 CONTAINER_PREFIX=axe-lb-fv BASE_PORT=6900 \
    GPU_INDICES_CSV=4,5,6,7 REPLICAS_PER_GPU=4 \
    "$AXE/start_drivers.sh" 2>>"$LOG" | tail -1)"
[[ "$addrs" == \[* ]] || { log "drivers failed"; exit 1; }

log "=== flatvits: evaluating 441 scenes x 1 rollout (dev, GPUs 4-7) ==="
RUN_DIR="$RUN_DIR" PRESET=dev CONTESTANT_IMAGE="$IMG" DRIVER_ADDRESSES="$addrs" \
N_ROLLOUTS=1 ROLLOUT_WORKERS=16 RENDER_GPUS_CSV=4,5,6,7 RENDERER_REPLICAS_PER_GPU=4 \
NRE_CACHE_SIZE=1 ENABLE_AUTORESUME=true \
    "$AXE/run_curated_val.sh" >> "$RUN_DIR.wizard.log" 2>&1
log "flatvits: wizard exit=$?"
docker rm -f $(docker ps -aq --filter 'name=axe-lb-fv') >/dev/null 2>&1 || true
[[ -f "$RUN_DIR/aggregate/results-summary.json" ]] && log "flatvits: OK" || log "flatvits: FAILED"
