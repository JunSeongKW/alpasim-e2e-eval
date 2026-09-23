#!/usr/bin/env bash
# Keeps the sweep running. Reads sweep_queue.txt top to bottom, runs each pending
# entry, marks it done, and re-reads the file every pass -- so entries appended
# while it runs are picked up without restarting it.
#
# Idles by polling rather than exiting, so an empty queue never leaves the box
# stopped for want of a new invocation.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QUEUE="$SCRIPT_DIR/sweep_queue.txt"
LOGDIR="${LOGDIR:-/tmp/claude-1000/-home-kaist5/f6b573e8-9153-4e4c-ba0a-564444b8bd06/scratchpad}"
IDLE_POLL="${IDLE_POLL:-120}"
MAX_IDLE="${MAX_IDLE:-100000}"          # 사실상 종료하지 않는다 — 중단 지시가 있을 때까지 대기

# Wait out any queue/trial already running, so two runners never overlap.
while pgrep -f 'run_rank_trial.sh|run_sweep_queue.sh' > /dev/null 2>&1; do sleep 60; done

idle=0
while :; do
    line=$(grep -vE '^\s*(#|$)' "$QUEUE" 2>/dev/null | grep -v '^done:' | head -1)
    if [[ -z "$line" ]]; then
        # Queue dry: propose the next batch from what has been measured so far,
        # rather than idling until a human types the next values.
        echo "=== $(date +%H:%M:%S) queue empty -> 자동 제안 생성 ==="
        if python3 "$SCRIPT_DIR/gen_next_combos.py" "${BATCH:-8}"; then
            idle=0; continue
        fi
        idle=$((idle+1))
        [[ "$idle" -ge "$MAX_IDLE" ]] && { echo "=== daemon exit $(date +%H:%M:%S) ==="; break; }
        sleep "$IDLE_POLL"; continue
    fi
    idle=0
    name="${line%%|*}"; note="${line#*|}"
    echo "=== $(date +%H:%M:%S) START $name ($note) ==="
    "$SCRIPT_DIR/run_rank_trial.sh" "$name" "$note" > "$LOGDIR/trial_${name}.log" 2>&1 \
        || echo "  $name 실패 — 다음으로 진행"
    grep -E '평균|회전각별' "$LOGDIR/trial_${name}.log" | head -2
    # mark done in place
    python3 - "$QUEUE" "$name" <<'PY'
import sys, pathlib
q, name = pathlib.Path(sys.argv[1]), sys.argv[2]
out = []
for l in q.read_text().splitlines():
    if l.startswith(name + "|"):
        l = "done:" + l
    out.append(l)
q.write_text("\n".join(out) + "\n")
PY
    echo "=== $(date +%H:%M:%S) DONE $name ==="
done
