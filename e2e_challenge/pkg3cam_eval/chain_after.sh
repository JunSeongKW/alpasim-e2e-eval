#!/usr/bin/env bash
# Start the next bundle's 441 run as soon as the previous one finishes.
#
#   chain_after.sh <previous run name> <next model> [extra MiB per driver]
#
# Waits for runs/<previous>/status.json to reach "complete" (stops on "failed"),
# then picks drivers-per-GPU so the next run stays under the 81 GB the user set:
# the previous run's highest per-GPU peak plus the next model's extra memory per
# driver times 6 must stay under 79,000 MiB (2 GB margin for the other
# researcher's ~3 GB processes already counted in the peak); otherwise 5 per GPU.
# The launch itself still waits for all GPUs to drop under 4 GB (run_pkg3cam --wait).
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PREV="$1"; NEXT="$2"; EXTRA="${3:-0}"
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
replicas=6
(( peak + 6 * EXTRA < 79000 )) || replicas=5
log "$PREV complete; its peak ${peak} MiB, $NEXT extra ${EXTRA} MiB/driver -> $replicas drivers per GPU"
exec "$ROOT/.venv/bin/python" "$ROOT/e2e_challenge/pkg3cam_eval/run_pkg3cam.py" \
    --model "$NEXT" --wait --replicas "$replicas"
