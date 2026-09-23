#!/usr/bin/env bash
# Run a queue of sweep trials back to back, so a finished trial always starts the
# next one instead of leaving the box idle.
#
#   ./run_sweep_queue.sh "R2e:fine DAC 0.15" "R2f:fine DAC 0.0"
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOGDIR="${LOGDIR:-/tmp/claude-1000/-home-kaist5/f6b573e8-9153-4e4c-ba0a-564444b8bd06/scratchpad}"
for spec in "$@"; do
    name="${spec%%:*}"; note="${spec#*:}"
    echo "=== $(date +%H:%M:%S)  $name  ($note) ==="
    "$SCRIPT_DIR/run_rank_trial.sh" "$name" "$note" > "$LOGDIR/trial_${name}.log" 2>&1 \
        || echo "  $name 실패 (계속 진행)"
    tail -8 "$LOGDIR/trial_${name}.log" | grep -E '평균|회전각별' || true
done
echo "=== queue done $(date +%H:%M:%S) ==="
