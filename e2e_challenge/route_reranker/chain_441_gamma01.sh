#!/usr/bin/env bash
# Run the 441 set at gamma=0.1 once the gamma=0.02 run has finished.
#
# Same arm, same contract, same clips: cache-centre-max, 16 drivers / 16
# renderers / 16 workers on GPUs 0-3, dev preset, no video, gains lat 1.0 /
# lon 0.25 / idx 3. Only gamma differs, so the two 441 runs are paired with
# each other as well as with axe-v9.
#
# Why gamma=0.1 is worth six hours. On the 38-clip sweep it rescued the most
# failures (11/26 against 5/26 at 0.02) and scored highest there, but it also
# broke two controls and doubled the collision rate, and reweighting that sweep
# to the 441 set's 9% corridor-failure share put it LAST of the three. The
# 40-clip validation then chose 0.02 without ever testing 0.1. So the
# extrapolation's central claim -- that the sweep's winner loses on the full
# set -- has not actually been measured. This measures it.
#
# It waits rather than running now because both runs need all four cards, and
# it tears down the previous run's containers first because run_441_reranker.sh
# refuses to start unless the cards are nearly empty.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/e2e_challenge/route_reranker"
cd "$ROOT"
LOG="$HERE/chain_441_g0p1.log"
log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

PREV_LOG="$HERE/441_g0p02.log"
log "waiting for gamma=0.02 to finish ($PREV_LOG)"
until grep -qE '^\[.*\] (OK|FAILED):' "$PREV_LOG" 2>/dev/null; do sleep 300; done
log "gamma=0.02 finished: $(grep -E '^\[.*\] (OK|FAILED):' "$PREV_LOG" | tail -1 | cut -c1-120)"

# Its own launcher removes the driver containers, but the simulator stack is
# brought down by compose and can leave exited containers holding names and
# ports. Remove ours by name, print what is going, and never touch anything
# that is not this run's.
log "clearing gamma=0.02 containers"
for c in $(docker ps -a --format '{{.Names}}' | grep -E '^(axe-rr-rr441-g0p02|rr441-g0p02)' | sort); do
    docker rm -f "$c" >/dev/null 2>&1 && log "  removed $c"
done
log "left: $(docker ps -a --format '{{.Names}}' | grep -cE '^(axe-rr-rr441-g0p02|rr441-g0p02)')"

# The launcher checks the cards are under 4 GB before it starts; give the
# driver teardown time to return the memory.
for _ in $(seq 1 30); do
    busy=0
    for mib in $(nvidia-smi -i 0,1,2,3 --query-gpu=memory.used --format=csv,noheader,nounits); do
        (( mib < 4000 )) || busy=1
    done
    (( busy == 0 )) && break
    sleep 60
done
log "GPU state: $(nvidia-smi -i 0,1,2,3 --query-gpu=memory.used --format=csv,noheader | paste -sd' ')"

log "=== launching 441 at gamma=0.1 ==="
GAMMA=0.1 "$HERE/run_441_reranker.sh" > "$HERE/441_g0p1.nohup" 2>&1
log "launcher exit=$? -- see $HERE/441_g0p1.log"

# Progress, in the run directory so it survives any session.
log "=== done ==="
