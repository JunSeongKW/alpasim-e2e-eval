#!/usr/bin/env bash
# What gamma actually does, on one clip, with video for every value.
#
# ARM: cache-centre-max. The clip fails the lateral-corridor gate, and that gate
# is a MAXIMUM over the four seconds, measured at the BODY CENTRE, against a
# route that has to reach the near field to exist at all. Each of the three
# switches is there because the scored quantity has it:
#   * cache      -- without it the 8 m projection-span gate opens on 17% of
#                   frames and gamma has almost nothing to act on. With it,
#                   256 of 256 candidates become comparable.
#   * max        -- the scorer zeroes a rollout if the corridor is left at ANY
#                   moment. A mean lets a wide swing average away, so a sweep on
#                   the mean mostly measures how little the mean moves. The
#                   worst-moment cost is also several times larger, which is what
#                   gives gamma a readable range instead of a threshold nobody
#                   reaches.
#   * centre_dx  -- vocab poses are the rear axle; the scorer measures the body
#                   centre, 1.467 m ahead. A coordinate correction, not a knob.
#
# CLIP: eaba3f6a. A right turn beginning at 49.6 m that leaves the corridor at
# 73.8 m, so the cache has about 32 m of recovered near field before the
# failure. It is also the one clip of the ten whose outcome the reranker already
# moved with the far-only route, which makes it responsive enough to read.
#
# SCALE: one card, one driver and one renderer per value. A single 20 s clip does
# not need sixteen of anything, and the small stack starts in minutes rather than
# twenty, which is what makes five values affordable while the 441-clip run holds
# the same four cards. Throughput changes; the scored contract does not.
#
# Docker refuses a second publish of the same host port and reports it as
# "drivers failed to start", so every value gets its own port block.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/e2e_challenge/route_reranker"
cd "$ROOT"
LOG="$HERE/gamma_sweep.log"
log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

# 0 is the reference: `weight == 0` returns the model's own pick inside the
# bundle's selector, so it is the same build rather than a separate one.
read -r -a GAMMAS <<< "${GAMMAS:-0 0.0005 0.002 0.01 0.05}"
read -r -a CARDS <<< "${CARDS:-0 1 2 3}"
CLIPS="$HERE/clip1_turn.txt"
pids=()

for idx in "${!GAMMAS[@]}"; do
    g="${GAMMAS[$idx]}"
    tag="g${g//./p}"
    card="${CARDS[$((idx % ${#CARDS[@]}))]}"
    # 6900-6915 belongs to the 441-clip evaluation's sixteen drivers; start well
    # clear of it so a publish never collides with a run that must not be touched.
    port=$((7100 + idx * 10))
    wiz=$((18000 + idx * 200))
    log "=== gamma=$g  (tag $tag, GPU $card, driver port $port, wizard from $wiz) ==="
    ARM=cache-centre-max RUN_TAG="gam-$tag" \
    ROUTE_RERANK_WEIGHT="$g" \
    CLIPS="$CLIPS" REPLICAS=1 ROLLOUT_WORKERS=1 WATCH_LIMIT=5 \
    BASE_PORT="$port" WIZARD_BASEPORT="$wiz" GPUS="$card" RENDER_GPUS="$card" \
        "$HERE/run_10clips_reranker.sh" > "$HERE/gam_${tag}.nohup" 2>&1 &
    pids+=($!)
    sleep 60                       # stagger: container creation is disk-bound
    if (( ${#pids[@]} >= ${#CARDS[@]} )); then
        wait "${pids[0]}"
        pids=("${pids[@]:1}")
    fi
done
wait

log "=== sweep done ==="
args=()
for g in "${GAMMAS[@]}"; do
    tag="g${g//./p}"
    f="$ROOT/runs/gam-$tag-cache-centre-max/aggregate/results-summary.json"
    if [[ -f "$f" ]]; then args+=("gamma$g=$f"); else log "gamma=$g: NO summary"; fi
done
if (( ${#args[@]} > 1 )); then
    "$ROOT/.venv/bin/python" "$ROOT/e2e_challenge/axe_local_eval/compare_on_clips.py" \
        "${args[@]}" 2>&1 | tee -a "$LOG"
fi
"$HERE/gamma_report.sh" 2>&1 | tee -a "$LOG"
