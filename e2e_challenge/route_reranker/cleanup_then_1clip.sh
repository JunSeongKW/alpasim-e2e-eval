#!/usr/bin/env bash
# Finish tearing down the stopped gamma run, then run the single clip.
#
# Why this needs its own script. The gamma=0.005 run was stopped mid-flight, and
# `docker rm -f` on its containers hangs: their processes sit in uninterruptible
# sleep because the ext4 journal (jbd2/vda1-8) is congested, so SIGKILL cannot be
# delivered until the pending writes drain. Piling up more rm calls makes the
# congestion worse, so this retries slowly, by name, and waits for the card
# rather than forcing anything.
#
# Everything here is scoped to this one run's container names. Another
# researcher's evaluation is on the same daemon and the same filesystem.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/e2e_challenge/route_reranker"
cd "$ROOT"
PREFIX='rr441-g0p005-augep29'
CARD="${CARD:-4}"
LOG="$HERE/cleanup_then_1clip.log"
log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

log "=== draining $PREFIX containers (GPU $CARD target) ==="
for round in $(seq 1 60); do
    left=$(docker ps -a --format '{{.Names}}' | grep -cE "^(axe-rr-)?$PREFIX" || true)
    (( left == 0 )) && { log "all containers gone"; break; }
    # One at a time, with a timeout, so a hung reap does not block the rest.
    for c in $(docker ps -a --format '{{.Names}}' | grep -E "^(axe-rr-)?$PREFIX" | head -4); do
        timeout 45 docker rm -f "$c" >/dev/null 2>&1 || true
    done
    now=$(docker ps -a --format '{{.Names}}' | grep -cE "^(axe-rr-)?$PREFIX" || true)
    log "round $round: $left -> $now containers, GPU $CARD $(nvidia-smi -i "$CARD" --query-gpu=memory.used --format=csv,noheader)"
    (( now == left )) && sleep 60   # no progress: give the journal room
done

log "waiting for GPU $CARD to come back"
for _ in $(seq 1 60); do
    mib=$(nvidia-smi -i "$CARD" --query-gpu=memory.used --format=csv,noheader,nounits)
    (( mib < 4000 )) && break
    sleep 30
done
mib=$(nvidia-smi -i "$CARD" --query-gpu=memory.used --format=csv,noheader,nounits)
if (( mib >= 4000 )); then
    log "GPU $CARD still holds ${mib} MiB after 30 min -- not starting the clip run"
    log "remaining: $(docker ps -a --format '{{.Names}}' | grep -cE "^(axe-rr-)?$PREFIX" || true) containers"
    exit 2
fi
log "GPU $CARD free (${mib} MiB)"

CARD="$CARD" "$HERE/run_1clip_ep29.sh"
log "=== 1-clip run exit=$? ==="
