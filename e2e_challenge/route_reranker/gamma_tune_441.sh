#!/usr/bin/env bash
# Tune the route-rerank gamma for one checkpoint on the full 441-clip set,
# one 441 run at a time, until told to stop.
#
#   nohup e2e_challenge/route_reranker/gamma_tune_441.sh > .../tune.nohup 2>&1 &
#
# Each iteration asks pick_next_gamma.py for the next value, runs the 441 set at
# it under the axe-v9 contract, appends the result to the ledger, and repeats. A
# run costs about four hours, so the ledger is the point: it is what makes the
# search resumable and what the next agent reads instead of the logs.
#
# Stopping. Three ways, in order of precedence:
#   * touch STOP_GAMMA_TUNE next to this script -- the loop finishes the run in
#     flight and then exits. This is the one to use; it never truncates a run.
#   * pick_next_gamma.py prints DONE, meaning the bracket is narrower than the
#     simulator's own noise and another four hours would not change the answer.
#   * MAX_RUNS as a backstop against an unattended loop running for days.
#
# The gamma=0 point is seeded from the checkpoint's existing no-reranker 441 run
# rather than measured again: it is the same checkpoint, the same contract and
# the same 16 workers, so it is already the paired reference. Worker count is
# what makes that fair -- at four workers the same code disagreed with itself on
# 8 of 38 clips, so a 16-worker run may only be compared with another.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/e2e_challenge/route_reranker"
cd "$ROOT"

CKPT="${CKPT:-$ROOT/../models/stage3_aug_ep29_final.ckpt}"
IMG="${IMG:-alpasim-e2e-axe-v9:stage3-aug-ep29-final}"
CKPT_SHA="${CKPT_SHA:-29c6fca63c4d4bd561fb90101ebbf39f7cb23b87a09251b1db3a766f05ffdef8}"
SUFFIX="${SUFFIX:--augep29}"
BASELINE_RUN="${BASELINE_RUN:-$ROOT/runs/leaderboard-aug-ep29-final}"
GPUS="${GPUS:-4,5,6,7}"
BASE_PORT="${BASE_PORT:-8100}"
WIZARD_BASEPORT="${WIZARD_BASEPORT:-21300}"
MAX_RUNS="${MAX_RUNS:-12}"

LEDGER="$ROOT/runs/gamma_tune${SUFFIX}.jsonl"
LOG="$HERE/gamma_tune${SUFFIX}.log"
STOP="$HERE/STOP_GAMMA_TUNE"
PY="$ROOT/.venv/bin/python"
log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

record() {  # gamma run_dir  -> one ledger line, read back from the summary
    "$PY" - "$1" "$2" "$LEDGER" <<'EOF'
import json, sys, statistics, pathlib
gamma, run_dir, ledger = sys.argv[1], sys.argv[2], sys.argv[3]
s = pathlib.Path(run_dir) / "aggregate" / "results-summary.json"
row = {"gamma": float(gamma), "run_dir": run_dir}
if s.exists():
    rs = [r for r in json.load(open(s))["rollouts"] if r.get("score") is not None]
    row["clips"] = len(rs)
    row["score"] = statistics.fmean(r["score"] for r in rs)
    row["zero_rate"] = sum(1 for r in rs if r["score"] == 0) / len(rs)
    for k in ("collision_at_fault", "offroad", "left_corridor_laterally"):
        row[k] = statistics.fmean((r.get("metrics") or {}).get(k, 0.0) for r in rs)
else:
    row["score"] = None
    row["error"] = "no summary"
with open(ledger, "a") as f:
    f.write(json.dumps(row) + "\n")
print(f"  ledger: gamma={row['gamma']} score={row['score']} clips={row.get('clips')}")
EOF
}

# Seed gamma=0 from the no-reranker run, once.
if [[ ! -f "$LEDGER" ]]; then
    if [[ -f "$BASELINE_RUN/aggregate/results-summary.json" ]]; then
        log "seeding gamma=0 from $BASELINE_RUN (no reranker, same contract)"
        record 0 "$BASELINE_RUN"
    else
        log "WARNING: no baseline at $BASELINE_RUN; gamma=0 will be unmeasured"
        : > "$LEDGER"
    fi
fi

log "=== gamma tuning: $(basename "$CKPT") on GPUs $GPUS, ledger $LEDGER ==="
log "stop with: touch $STOP"

for ((run = 1; run <= MAX_RUNS; run++)); do
    if [[ -e "$STOP" ]]; then
        log "stop file present -- finishing"
        break
    fi
    g="$("$PY" "$HERE/pick_next_gamma.py" --ledger "$LEDGER" --explain 2>>"$LOG")"
    if [[ -z "$g" ]]; then
        log "picker says done -- search complete"
        break
    fi
    tag="g${g//./p}${SUFFIX}"
    run_dir="$ROOT/runs/rr441-$tag-cache-centre-max"
    if [[ -e "$run_dir" ]]; then
        log "SKIP gamma=$g: $run_dir already exists"
        record "$g" "$run_dir"
        continue
    fi

    log "--- run $run/$MAX_RUNS: gamma=$g"
    start=$(date +%s)
    GAMMA="$g" IMG="$IMG" CKPT_SHA="$CKPT_SHA" TAG_SUFFIX="$SUFFIX" \
    GPUS="$GPUS" BASE_PORT="$BASE_PORT" WIZARD_BASEPORT="$WIZARD_BASEPORT" \
        "$HERE/run_441_reranker.sh" >> "$LOG" 2>&1
    log "    gamma=$g launcher exit=$? after $((($(date +%s) - start) / 60)) min"
    record "$g" "$run_dir"

    # The simulator stack is brought down by compose but can leave exited
    # containers holding names and ports; the next run refuses to start if the
    # cards are not nearly empty. Remove only this run's own containers.
    for c in $(docker ps -a --format '{{.Names}}' | grep -E "^(axe-rr-rr441-$tag|rr441-$tag)" | sort); do
        docker rm -f "$c" >/dev/null 2>&1 || true
    done
    for _ in $(seq 1 30); do
        busy=0
        for mib in $(nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader,nounits); do
            (( mib < 4000 )) || busy=1
        done
        (( busy == 0 )) && break
        sleep 30
    done
done

log "=== tuning stopped ==="
"$PY" "$HERE/pick_next_gamma.py" --ledger "$LEDGER" --explain >/dev/null 2>>"$LOG" || true
