#!/usr/bin/env bash
# Run the base-gain control as soon as the tuned ep29 evaluation finishes.
#
# Both runs want all sixteen driver replicas and all four GPUs, so they cannot
# overlap; chaining them here keeps the machine busy without anyone having to
# watch for the first one to end. The wait keys on the wizard process rather
# than on the summary file, because a run that fails never writes a summary and
# would otherwise block the chain forever.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AXE="$ROOT/e2e_challenge/axe_local_eval"
LOG="$AXE/chain_ep29_pair.log"
log() { echo "[$(date '+%F %H:%M:%S')] $*" >> "$LOG"; }

log "waiting for the tuned ep29 run to finish"
while pgrep -f 'run_epzero_ep29\.sh' >/dev/null 2>&1; do sleep 120; done

tuned="$ROOT/runs/leaderboard-epzero-ep29/aggregate/results-summary.json"
if [[ -f "$tuned" ]]; then
    log "tuned run produced a summary"
else
    log "WARNING: tuned run left no summary; running the control anyway"
fi

# The tuned run removes its own containers, but a crashed run would not.
docker ps -aq --filter 'name=axe-lb-ep29' | xargs -r docker rm -f >/dev/null 2>&1 || true
sleep 30

log "starting the base-gain control"
"$AXE/run_epzero_ep29_basegains.sh" >> "$LOG" 2>&1
log "control finished with exit=$?"
