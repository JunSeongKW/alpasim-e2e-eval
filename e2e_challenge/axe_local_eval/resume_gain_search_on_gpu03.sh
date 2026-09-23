#!/usr/bin/env bash
# Resume the controller-gain search on GPUs 0-3 as soon as the axe-v5
# leaderboard run frees them.
#
# The search keeps the ec2 preset and the submit-v1 image it used before, so the
# 11 results already in results.jsonl stay comparable; only the GPUs change.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
AXE="$ROOT/e2e_challenge/axe_local_eval"
GS="$AXE/gain_search"
LOG="$GS/loop.log"
V5="$ROOT/runs/leaderboard-axe-v5/aggregate/results-summary.json"
log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

log "waiting for axe-v5 to release GPUs 0-3"
while [[ ! -f "$V5" ]]; do
    pgrep -f 'leaderboard-axe-v5' >/dev/null || pgrep -f alpasim_wizard >/dev/null || {
        sleep 90
        [[ -f "$V5" ]] || { log "axe-v5 ended without a summary; not resuming"; exit 1; }
    }
    sleep 30
done
log "axe-v5 done; freeing its drivers"
docker rm -f $(docker ps -aq --filter 'name=axe-lb-axe-v5') >/dev/null 2>&1 || true
sleep 10

log "starting 16 search drivers on GPUs 0-3"
addrs="$(IMAGE=alpasim-e2e-drivesuprim-stage3:submit-v1 \
    OFFICIAL_ENV_ONLY=1 \
    CONTAINER_PREFIX=axe-gs-drv \
    BASE_PORT=6960 \
    GPU_INDICES_CSV=0,1,2,3 REPLICAS_PER_GPU=4 \
    "$AXE/start_drivers.sh" 2>>"$LOG" | tail -1)"
[[ "$addrs" == \[* ]] || { log "search drivers failed to start"; exit 1; }
printf '%s\n' "$addrs" > "$GS/driver_addresses.txt"

log "resuming the gain search on GPUs 0-3"
rm -f "$GS/STOP"
exec env DRIVER_ADDRESSES="$addrs" RENDER_GPUS_CSV=0,1,2,3 "$GS/loop.sh"
