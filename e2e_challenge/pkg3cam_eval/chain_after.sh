#!/usr/bin/env bash
# Start the next bundle's 441 run as soon as the previous one finishes.
#
#   chain_after.sh <previous run name> <next model> <extra MiB per driver> [run_pkg3cam args...]
#   e.g. ... vitl512_3layer_lr1 1800 --prune-asl
#
# Waits for runs/<previous>/status.json to reach "complete" (stops on "failed"),
# then picks drivers-per-GPU so the next run stays under the 81 GB the user set:
# the previous run's highest per-GPU peak plus the next model's extra memory per
# driver times 6 must stay under 79,000 MiB (2 GB margin for the other
# researcher's ~3 GB processes already counted in the peak); otherwise 5 per GPU.
# The launch itself still waits for all GPUs to drop under 4 GB (run_pkg3cam --wait).
#
# SMOKE_GPU=<n>: once the previous run is complete and every GPU is back under
# 4 GB, run the next model's GPU smoke (strict load + real forward) on GPU n and
# start the 441 run only if it passes. Used for vitl512 on 10-08, whose smoke
# could not run earlier: with the disk full, docker could not create containers
# while the previous run was writing.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREV="$1"; NEXT="$2"; EXTRA="${3:-0}"; shift 3 || shift $#
STATUS="$ROOT/runs/$PREV/status.json"
log() { echo "[$(date '+%F %T')] $*"; }
log "waiting for $PREV to complete before $NEXT"
while :; do
    stage="$(python3 -c "import json;print(json.load(open('$STATUS'))['stage'])" 2>/dev/null)"
    [[ "$stage" == complete ]] && break
    [[ "$stage" == failed ]] && { log "$PREV failed; not starting $NEXT"; exit 1; }
    sleep 60
done
peak="$(python3 -c "import json;print(max(json.load(open('$ROOT/runs/$PREV/vram-peaks.json')).values()))")"
gpus_free() {
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '$1 >= 4000 {busy=1} END {exit busy}'
}
if [[ -n "${SMOKE_GPU:-}" ]]; then
    stable=0
    while (( stable < 2 )); do
        if gpus_free; then stable=$((stable + 1)); else stable=0; fi
        (( stable < 2 )) && sleep 30
    done
    log "GPUs free; running $NEXT smoke on GPU $SMOKE_GPU"
    "$ROOT/.venv/bin/python" "$ROOT/e2e_challenge/pkg3cam_eval/run_pkg3cam.py" --model "$NEXT" --smoke "$SMOKE_GPU" \
        || { log "$NEXT smoke FAILED; not starting the 441 run"; exit 1; }
fi
replicas=6
(( peak + 6 * EXTRA < 79000 )) || replicas=5
log "$PREV complete; its peak ${peak} MiB, $NEXT extra ${EXTRA} MiB/driver -> $replicas drivers per GPU"
exec "$ROOT/.venv/bin/python" "$ROOT/e2e_challenge/pkg3cam_eval/run_pkg3cam.py" \
    --model "$NEXT" --wait --replicas "$replicas" "$@"
