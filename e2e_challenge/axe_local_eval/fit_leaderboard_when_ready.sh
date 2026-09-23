#!/usr/bin/env bash
# Fit the 11-subject local leaderboard once both parallel runs have a summary.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
LOG="$ROOT/e2e_challenge/axe_local_eval/v4v5.log"
V4="$ROOT/runs/leaderboard-axe-v4/aggregate/results-summary.json"
V5="$ROOT/runs/leaderboard-axe-v5/aggregate/results-summary.json"
E30="$ROOT/runs/leaderboard-stage3-ep30/aggregate/results-summary.json"
log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

while [[ ! -f "$V4" || ! -f "$V5" || ! -f "$E30" ]]; do
    # Give up if neither run is still producing rollouts -- a stalled wizard
    # would otherwise leave this waiting forever.
    if ! pgrep -f 'alpasim_wizard' >/dev/null; then
        sleep 120
        [[ -f "$V4" && -f "$V5" && -f "$E30" ]] || { log "both wizards gone but summaries missing; aborting"; exit 1; }
    fi
    sleep 60
done

log "=== all summaries present; fitting 12-subject leaderboard ==="
OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 \
uv run --extra local-evaluation python "$ROOT/e2e_challenge/local_evaluation/evaluate.py" \
    --track pai \
    --run axe-v4="$ROOT/runs/leaderboard-axe-v4" \
    --run axe-v5="$ROOT/runs/leaderboard-axe-v5" \
    --run axe-v8-ep24="$ROOT/runs/axe-ep24-curatedval-egobox" \
    --run stage3-ep30="$ROOT/runs/leaderboard-stage3-ep30" \
    --output-dir "$ROOT/runs/leaderboard-axe-v4v5v8" \
    --device cpu >> "$LOG" 2>&1
log "=== fit done ==="
