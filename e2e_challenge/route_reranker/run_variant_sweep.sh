#!/usr/bin/env bash
# Four reranker arms on the ten lateral-corridor clips, two at a time.
#
# All four accumulate the route, so the reranker finally has near-field geometry
# to measure against; they differ only in HOW the distance is measured:
#
#            measured from        aggregated over 4 s
#   cache            vocab pose   mean   <- the bundle, plus the cache
#   cache-centre     body centre  mean
#   cache-max        vocab pose   max
#   cache-centre-max body centre  max    <- both, matching the scorer
#
# Two at a time because the 441-clip evaluation is on the same four cards: it
# holds ~47 GB of 81.5 GB, one arm adds ~12 GB, and three would not fit. Each
# arm gets its own driver ports and its own wizard port block so the two running
# together cannot collide on the host network.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/e2e_challenge/route_reranker"
cd "$ROOT"
LOG="$HERE/variant_sweep.log"
log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

ARMS=(cache cache-centre cache-max cache-centre-max)
PORTS=(6940 6950 6960 6970)
WIZ=(18000 18100 18200 18300)

# `set -u` plus a backgrounded function is the trap here: the subshell inherits
# nounset, so an index read before the assignment lands aborts the arm silently
# and the sweep races to the end. Bind the values in the caller instead.
run_one() {   # arm port wizardport
    local arm="$1" port="$2" wiz="$3"
    ARM="$arm" RUN_TAG=rr4 BASE_PORT="$port" WIZARD_BASEPORT="$wiz" \
        GPUS=0,1,2,3 "$HERE/run_10clips_reranker.sh" \
        > "$HERE/rr4_${arm}.nohup" 2>&1
}

for batch in 0 2; do
    a=$batch; b=$((batch + 1))
    log "=== batch: ${ARMS[$a]} + ${ARMS[$b]} ==="
    run_one "${ARMS[$a]}" "${PORTS[$a]}" "${WIZ[$a]}" & p1=$!
    sleep 30                      # stagger the driver starts, they are disk-heavy
    run_one "${ARMS[$b]}" "${PORTS[$b]}" "${WIZ[$b]}" & p2=$!
    wait "$p1" "$p2"
    for arm in "${ARMS[$a]}" "${ARMS[$b]}"; do
        d="$ROOT/runs/rr4-$arm"
        log "$arm: summary=$([[ -f $d/aggregate/results-summary.json ]] && echo yes || echo NO)" \
            "videos=$(find "$d" -name '*.mp4' 2>/dev/null | wc -l)"
    done
done

log "=== 4 arms done; comparing ==="
args=()
for arm in "${ARMS[@]}"; do
    f="$ROOT/runs/rr4-$arm/aggregate/results-summary.json"
    [[ -f "$f" ]] && args+=("$arm=$f")
done
# The two earlier arms are the same launcher, image and clips, so they belong in
# the same table: `before` is the reranker off, `rerank` is the bundle without
# the cache.
for prev in before rerank; do
    f="$ROOT/runs/rerank10-$prev/aggregate/results-summary.json"
    [[ -f "$f" ]] && args+=("$prev=$f")
done
"$ROOT/.venv/bin/python" "$ROOT/e2e_challenge/axe_local_eval/compare_on_clips.py" "${args[@]}" \
    2>&1 | tee -a "$LOG"
