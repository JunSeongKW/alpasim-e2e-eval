#!/usr/bin/env bash
# Phase 2: tune the collision gate at the gamma the ladder chose, on 441 clips,
# optimising scene score AND at-fault distance together.
#
#   nohup e2e_challenge/route_reranker/gate_sweep_441.sh > .../gate.nohup 2>&1 &
#
# It waits for the gamma tuning loop to exit, reads the winning gamma out of that
# ledger, and then varies the gate one knob at a time from the configuration the
# gamma runs used. That baseline is not re-measured: the winning gamma's own run
# is the reference point, which buys back four hours.
#
# Why one knob at a time and not a grid. A 441 run costs four hours; a 3x3x3 grid
# is 27 runs, over four days. One-at-a-time over three knobs is four runs, and
# because the knobs act on the same quantity -- how much clearance a candidate
# must leave before the gate drops it -- their effects are close to monotone and
# largely additive. The two directions that help most are then combined and
# verified, which is where a genuine interaction would show up.
#
# Which knobs. Only three of the gate's four settings can do anything here:
#   feasibility_collision_margin  extra half-extent on every box side, m
#   feasibility_collision_scale   multiplicative box scale-up
#   feasibility_agent_conf_thresh detection confidence to count an agent at all
# feasibility_predict_horizon is inert: the aux agent head emits the 5-field
# TransFuser box with no velocity, so agents are frozen at their current pose for
# the whole 4 s no matter what the horizon says. Making the horizon matter needs
# the 10-field box, i.e. retraining -- out of scope for a sweep.
#
# The objective is two-dimensional, so the rule is stated rather than implied:
# among the combinations whose at-fault distance is at least the baseline's, take
# the highest scene score. Combinations that raise the score by sacrificing
# at-fault distance are reported but not chosen, because at-fault distance is
# what the official capability score weights most heavily and it is the axis the
# request named. decide_gate.py applies that rule and also prints the Pareto
# front, so a different trade-off can be taken by hand.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/e2e_challenge/route_reranker"
cd "$ROOT"

SUFFIX="${SUFFIX:--augep29}"
IMG="${IMG:-alpasim-e2e-axe-v9:stage3-aug-ep29-final}"
CKPT_SHA="${CKPT_SHA:-29c6fca63c4d4bd561fb90101ebbf39f7cb23b87a09251b1db3a766f05ffdef8}"
GPUS="${GPUS:-4,5,6,7}"
BASE_PORT="${BASE_PORT:-8200}"
WIZARD_BASEPORT="${WIZARD_BASEPORT:-21500}"
MAX_RUNS="${MAX_RUNS:-8}"

GAMMA_LEDGER="$ROOT/runs/gamma_tune${SUFFIX}.jsonl"
LEDGER="$ROOT/runs/gate_sweep${SUFFIX}.jsonl"
LOG="$HERE/gate_sweep${SUFFIX}.log"
STOP="$HERE/STOP_GATE_SWEEP"
PY="$ROOT/.venv/bin/python"
log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

# Wait for phase 1. The gamma loop writes this line when it stops for any reason.
log "waiting for the gamma ladder to finish"
until grep -q '=== tuning stopped ===' "$HERE/gamma_tune${SUFFIX}.log" 2>/dev/null; do
    [[ -e "$STOP" ]] && { log "stop file present before phase 2 began -- exiting"; exit 0; }
    sleep 300
done

GAMMA="$("$PY" - "$GAMMA_LEDGER" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
rows = [r for r in rows if r.get("score") is not None]
print(max(rows, key=lambda r: r["score"])["gamma"] if rows else "")
EOF
)"
[[ -n "$GAMMA" ]] || { log "no gamma in $GAMMA_LEDGER -- cannot start"; exit 2; }
gtag="g${GAMMA//./p}${SUFFIX}"
BASE_RUN="$ROOT/runs/rr441-$gtag-cache-centre-max"
log "=== phase 2: gate sweep at gamma=$GAMMA (baseline $BASE_RUN) ==="
log "stop with: touch $STOP"

record() {  # label margin scale conf run_dir
    "$PY" - "$1" "$2" "$3" "$4" "$5" "$LEDGER" "$GAMMA" <<'EOF'
import json, sys, statistics, pathlib
label, margin, scale, conf, run_dir, ledger, gamma = sys.argv[1:8]
s = pathlib.Path(run_dir) / "aggregate" / "results-summary.json"
row = {"label": label, "gamma": float(gamma), "margin": float(margin),
       "scale": float(scale), "conf": float(conf), "run_dir": run_dir}
if s.exists():
    rs = [r for r in json.load(open(s))["rollouts"] if r.get("score") is not None]
    m = lambda k: statistics.fmean((r.get("metrics") or {}).get(k, 0.0) for r in rs)
    dist = sum((r.get("metrics") or {}).get("dist_traveled_m", 0.0) for r in rs)
    nfault = sum((r.get("metrics") or {}).get("offroad_or_collision_at_fault", 0.0) for r in rs)
    row.update(
        clips=len(rs),
        score=statistics.fmean(r["score"] for r in rs),
        zero_rate=sum(1 for r in rs if r["score"] == 0) / len(rs),
        # The official definition: total km driven per at-fault incident.
        at_fault_km=(dist / nfault / 1000.0) if nfault else None,
        collision_at_fault=m("collision_at_fault"),
        collision_any=m("collision_any"),
        offroad=m("offroad"),
        left_corridor_laterally=m("left_corridor_laterally"),
        dist_traveled_m=dist / len(rs),
    )
else:
    row.update(score=None, error="no summary")
with open(ledger, "a") as f:
    f.write(json.dumps(row) + "\n")
print(f"  ledger: {label} score={row['score']} at_fault_km={row.get('at_fault_km')}")
EOF
}

# Seed the baseline from the winning gamma run: same gate, already measured.
if [[ ! -f "$LEDGER" ]]; then
    log "seeding baseline (margin 0.0, scale 1.1, conf 0.3) from $BASE_RUN"
    record baseline 0.0 1.1 0.3 "$BASE_RUN"
fi

one_run() {  # label margin scale conf
    local label="$1" margin="$2" scale="$3" conf="$4"
    local tag="gate-${label}${SUFFIX}"
    local run_dir="$ROOT/runs/rr441-$tag-cache-centre-max"
    if [[ -e "$run_dir" ]]; then
        log "SKIP $label: $run_dir exists"; record "$label" "$margin" "$scale" "$conf" "$run_dir"; return
    fi
    log "--- $label: margin=$margin scale=$scale conf=$conf (gamma=$GAMMA)"
    local start=$(date +%s)
    GAMMA="$GAMMA" IMG="$IMG" CKPT_SHA="$CKPT_SHA" TAG_SUFFIX="-$tag" \
    GPUS="$GPUS" BASE_PORT="$BASE_PORT" WIZARD_BASEPORT="$WIZARD_BASEPORT" \
    DRIVESUPRIM_FEAS_COLLISION_MARGIN="$margin" \
    DRIVESUPRIM_FEAS_COLLISION_SCALE="$scale" \
    DRIVESUPRIM_FEAS_AGENT_CONF="$conf" \
        "$HERE/run_441_reranker.sh" >> "$LOG" 2>&1
    log "    $label exit=$? after $((($(date +%s) - start) / 60)) min"
    record "$label" "$margin" "$scale" "$conf" "$run_dir"
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
}

# Stage A: one knob at a time. Two margins because margin is the knob that acts
# directly on "how close is too close", which the clip analysis pointed at: the
# median passing clip cleared its nearest obstacle by only 1.06 m.
STAGE_A=(
    "margin0.3|0.3|1.1|0.3"
    "margin0.6|0.6|1.1|0.3"
    "scale1.25|0.0|1.25|0.3"
    "conf0.20|0.0|1.1|0.20"
)
n=0
for spec in "${STAGE_A[@]}"; do
    [[ -e "$STOP" ]] && { log "stop file present -- finishing"; break; }
    (( ++n > MAX_RUNS )) && break
    IFS='|' read -r label margin scale conf <<< "$spec"
    one_run "$label" "$margin" "$scale" "$conf"
done

# Stage B: combine the two knobs that helped most, then verify.
if [[ ! -e "$STOP" ]] && (( n < MAX_RUNS )); then
    log "=== stage B: combining the best directions ==="
    read -r cmargin cscale cconf < <("$PY" "$HERE/decide_gate.py" --ledger "$LEDGER" --combine 2>>"$LOG")
    if [[ -n "${cmargin:-}" ]]; then
        one_run "combo" "$cmargin" "$cscale" "$cconf"
    else
        log "no combination worth running (no knob improved both objectives)"
    fi
fi

log "=== gate sweep done ==="
"$PY" "$HERE/decide_gate.py" --ledger "$LEDGER" --explain 2>&1 | tee -a "$LOG"
