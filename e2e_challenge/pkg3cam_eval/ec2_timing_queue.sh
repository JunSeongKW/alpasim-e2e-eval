#!/usr/bin/env bash
# Time every model under the official ec2 contract on the same first N scenes,
# one after another, then write the table (441 wall time estimated per model).
#
#   nohup e2e_challenge/pkg3cam_eval/ec2_timing_queue.sh 20261012 128 runs/latency-bench-20261012_0839 \
#       > runs/ec2-timing-queue-20261012.progress.log 2>&1 &
#
# Latency percentiles and the steady throughput need a few waves of the 32
# slots, not all 441 scenes; run_ec2_timing.py --table extrapolates the full
# run and checks the extrapolation against axe-v9's one full 441 run. Runs are
# sequential: the official topology uses all eight GPUs. A failed run is logged
# and the queue moves on; rerun it with a single run_ec2_timing.py call.
set -uo pipefail
DATE="${1:?date tag}"
SCENES="${2:-128}"
BENCH="${3:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
log() { echo "[$(date '+%F %T')] $*"; }
# axe-v9 last: its full 441 reference run holds the same driver names until it ends
for model in vits512_6layer vits768_3layer vitl512_3layer_lr1 vits512_g112_kv28 vits512_vidarrender axe-v9; do
    run="ec2-timing-${model//_/}-${DATE}-first${SCENES}"
    if [[ -f "runs/$run/aggregate/results-summary.json" ]]; then
        log "$model already done"; continue
    fi
    log "$model: start ($SCENES scenes)"
    .venv/bin/python e2e_challenge/pkg3cam_eval/run_ec2_timing.py --model "$model" --date "$DATE" --scenes "$SCENES" \
        --wait >> "runs/$run.progress.log" 2>&1 \
        && log "$model: done" || log "$model: FAILED (see runs/$run.progress.log)"
done
.venv/bin/python e2e_challenge/pkg3cam_eval/run_ec2_timing.py --table --date "$DATE" --scenes "$SCENES" ${BENCH:+--bench "$BENCH"}
log "table: runs/ec2-timing-summary-${DATE}.md"
