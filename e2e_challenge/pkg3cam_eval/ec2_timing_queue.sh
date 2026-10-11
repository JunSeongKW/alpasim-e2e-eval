#!/usr/bin/env bash
# Time every model's 441 run under the official ec2 contract, one after another,
# then write the table.
#
#   nohup e2e_challenge/pkg3cam_eval/ec2_timing_queue.sh 20261012 runs/latency-bench-20261012_0839 \
#       > runs/ec2-timing-queue-20261012.progress.log 2>&1 &
#
# Runs are sequential: the official topology uses all eight GPUs, and two runs
# side by side would time each other's contention instead of the model. A run
# that fails is logged and the queue moves on; its rerun is a single
# run_ec2_timing.py call.
set -uo pipefail
DATE="${1:?date tag}"
BENCH="${2:-}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
log() { echo "[$(date '+%F %T')] $*"; }
for model in axe-v9 vits512_6layer vits768_3layer vitl512_3layer_lr1 vits512_g112_kv28 vits512_vidarrender; do
    if [[ -f "runs/ec2-timing-${model//_/}-${DATE}/aggregate/results-summary.json" ]]; then
        log "$model already done"; continue
    fi
    log "$model: start"
    .venv/bin/python e2e_challenge/pkg3cam_eval/run_ec2_timing.py --model "$model" --date "$DATE" --wait \
        >> "runs/ec2-timing-${model//_/}-${DATE}.progress.log" 2>&1 \
        && log "$model: done" || log "$model: FAILED (see runs/ec2-timing-${model//_/}-${DATE}.progress.log)"
done
.venv/bin/python e2e_challenge/pkg3cam_eval/run_ec2_timing.py --table --date "$DATE" ${BENCH:+--bench "$BENCH"}
log "table: runs/ec2-timing-summary-${DATE}.md"
