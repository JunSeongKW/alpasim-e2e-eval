#!/usr/bin/env bash
# Measure the shortlisted gammas on a random slice of the 441 set: three arms
# side by side, each at 8 drivers / 8 renderers / 8 workers.
#
# Why a second set. The 38-clip sweep is two-thirds corridor failures and its
# controls are all clips scoring above 0.95. The 441 set is 9% corridor
# failures, and among its other 401 clips about 97 score zero for collisions or
# leaving the road -- a population the sweep never touched, and the one most
# likely to react to a route penalty in either direction. clips_random40.txt is
# a stratified random sample of the 441 by baseline score (9 zeros, 3 low, 4
# mid, 24 high), so its mean tracks the full set's without any reweighting.
#
# Why three arms at once, and why 8 rather than 16. The pipeline is not
# GPU-bound: one worker per card left the cards 85% idle, because each step is
# render -> inference -> controller -> physics in series and the card waits on
# the CPU stages. Four workers per card is where utilisation saturates. Three
# arms of 8 put six drivers and six renderers on each card (about 70 GB of
# 81.5) and finish in roughly the time one arm of 16 would take alone. The
# reference is the gamma=0 arm run in this same configuration, so the three
# are paired with each other exactly; the 441 baseline is reported beside them
# for scale only.
#
# gamma=0 is also the cheapest proof that the mounted code reproduces the
# baseline before six hours are spent on the full set: on the 38-clip sweep the
# same code disagreed with the 441 run on 8 clips, and whether that is
# concurrency or plain run-to-run nondeterminism, its size on these 40 clips is
# the noise floor every other gamma has to clear.
#
# Gammas are read from pick_gamma_candidates.py at launch, so this can be
# chained behind the sweep without knowing the answer in advance.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/e2e_challenge/route_reranker"
cd "$ROOT"
LOG="$HERE/gamma_validate.log"
log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

TOP="${TOP:-2}"
if [[ -n "${GAMMAS:-}" ]]; then
    read -r -a GAMMAS <<< "$GAMMAS"
else
    mapfile -t cand < <("$ROOT/.venv/bin/python" "$HERE/pick_gamma_candidates.py" | head -n "$TOP")
    GAMMAS=(0 "${cand[@]}")
fi
CLIPS="$HERE/clips_random40.txt"
n_clips="$(grep -vc '^[[:space:]]*$' "$CLIPS")"
log "=== validation set: $n_clips clips; gammas: ${GAMMAS[*]}; 3 x (8/8/8) in parallel ==="

pids=()
for idx in "${!GAMMAS[@]}"; do
    g="${GAMMAS[$idx]}"
    tag="g${g//./p}"
    port=$((7300 + idx * 20))        # 8 driver ports per arm, well apart
    wiz=$((19500 + idx * 100))       # ~21 service ports per arm
    log "=== gamma=$g  (drivers on $port+, wizard from $wiz) ==="
    ARM=cache-centre-max RUN_TAG="val-$tag" \
    ROUTE_RERANK_WEIGHT="$g" \
    CLIPS="$CLIPS" REPLICAS=2 ROLLOUT_WORKERS=8 WATCH_LIMIT=10 \
    RENDER_VIDEO=false \
    BASE_PORT="$port" WIZARD_BASEPORT="$wiz" GPUS=0,1,2,3 RENDER_GPUS=0,1,2,3 \
        "$HERE/run_10clips_reranker.sh" > "$HERE/val_${tag}.nohup" 2>&1 &
    pids+=($!)
    sleep 120                        # stagger: 16 containers per arm, disk-bound to create
done
start=$(date +%s)
wait "${pids[@]}"
log "=== all arms done in $((($(date +%s) - start) / 60)) min after the last launch ==="
for g in "${GAMMAS[@]}"; do
    d="$ROOT/runs/val-g${g//./p}-cache-centre-max"
    log "gamma=$g summary=$([[ -f $d/aggregate/results-summary.json ]] && echo yes || echo NO)"
done
"$HERE/gamma_validate_report.sh" 2>&1 | tee -a "$LOG"
log "=== validation done ==="
