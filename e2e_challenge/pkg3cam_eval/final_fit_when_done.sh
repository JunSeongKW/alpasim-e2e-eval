#!/usr/bin/env bash
# After the last bundle's 441 run completes, fit the leaderboard with the given bundles.
#
#   final_fit_when_done.sh <run to wait for> <model> [<model> ...]
#   e.g. ... leaderboard-stage3-vits512-vidarrender-20261010 \
#        vits512_6layer vits768_3layer vitl512_3layer_lr1 vits512_g112_kv28 vits512_vidarrender
#
# Refuses on infrastructure failures (pkg3cam_leaderboard.py checks), and never
# runs on a failed or stopped run.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN="${1:?run name to wait for}"; shift
(( $# )) || { echo "no models given" >&2; exit 2; }
S="$ROOT/runs/$RUN/status.json"
log() { echo "[$(date '+%F %T')] $*"; }
log "waiting for $RUN to complete"
while :; do
    stage="$(python3 -c "import json;print(json.load(open('$S'))['stage'])" 2>/dev/null)"
    [[ "$stage" == complete ]] && break
    [[ "$stage" == failed || "$stage" == invalid_stopped || "$stage" == paused ]] && { log "$RUN stage=$stage; no fit"; exit 1; }
    sleep 60
done
log "$RUN complete; fitting 36 + $# subjects"
cd "$ROOT" && UV_OFFLINE=1 "$ROOT/.venv/bin/python" "$ROOT/e2e_challenge/pkg3cam_eval/pkg3cam_leaderboard.py" "$@" \
    && log "FIT DONE" || log "FIT FAILED"
