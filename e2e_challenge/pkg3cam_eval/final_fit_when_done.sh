#!/usr/bin/env bash
# After the vitl512 441 run completes, fit the final 39-subject leaderboard.
# Refuses on infrastructure failures (pkg3cam_leaderboard.py checks), and never
# runs on a failed or stopped vitl run.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
S="$ROOT/runs/leaderboard-stage3-vitl512-3layer-lr1-20261008/status.json"
log() { echo "[$(date '+%F %T')] $*"; }
log "waiting for vitl512 to complete"
while :; do
    stage="$(python3 -c "import json;print(json.load(open('$S'))['stage'])" 2>/dev/null)"
    [[ "$stage" == complete ]] && break
    [[ "$stage" == failed || "$stage" == invalid_stopped || "$stage" == paused ]] && { log "vitl stage=$stage; no fit"; exit 1; }
    sleep 60
done
log "vitl complete; fitting 39 subjects"
cd "$ROOT" && UV_OFFLINE=1 "$ROOT/.venv/bin/python" "$ROOT/e2e_challenge/pkg3cam_eval/pkg3cam_leaderboard.py" \
    vits512_6layer vits768_3layer vitl512_3layer_lr1 && log "FIT DONE" || log "FIT FAILED"
