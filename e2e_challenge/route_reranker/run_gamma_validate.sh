#!/usr/bin/env bash
# Measure the shortlisted gammas on a random slice of the 441 set, at the
# 441 run's own concurrency, so the number that comes out IS the 441 estimate.
#
# Why a second set. The 38-clip sweep is two-thirds corridor failures and its
# controls are all clips scoring above 0.95. The 441 set is 9% corridor
# failures, and among its other 401 clips about 97 score zero for collisions or
# leaving the road -- a population the sweep never touched, and the one most
# likely to react to a route penalty in either direction. clips_random40.txt is
# a stratified random sample of the 441 by baseline score (9 zeros, 3 low, 4
# mid, 24 high), so its mean tracks the full set's without any reweighting.
#
# Why sixteen of everything. gamma=0 run at four workers disagreed with the
# 441 run on 8 of 38 clips -- same code, same checkpoint, only the concurrency
# differed, which moves gRPC timing and flips clips near the 4 m boundary. The
# 441 baseline was produced at 16 drivers / 16 renderers / 16 workers; running
# the validation at the same contract makes leaderboard-merged-route-ep30 a
# valid paired reference on these 40 clips and saves a separate baseline arm.
# gamma=0 is still run once, as the cheapest possible proof that the mounted
# code reproduces the baseline at this concurrency before six hours are spent
# on the full set.
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
log "=== validation set: $n_clips clips; gammas: ${GAMMAS[*]} ==="

for g in "${GAMMAS[@]}"; do
    tag="g${g//./p}"
    log "=== gamma=$g  ($n_clips clips, 16 drivers / 16 renderers / 16 workers) ==="
    start=$(date +%s)
    ARM=cache-centre-max RUN_TAG="val-$tag" \
    ROUTE_RERANK_WEIGHT="$g" \
    CLIPS="$CLIPS" REPLICAS=4 ROLLOUT_WORKERS=16 WATCH_LIMIT=10 \
    RENDER_VIDEO=false \
    BASE_PORT=7300 WIZARD_BASEPORT=19500 GPUS=0,1,2,3 RENDER_GPUS=0,1,2,3 \
        "$HERE/run_10clips_reranker.sh" > "$HERE/val_${tag}.nohup" 2>&1
    d="$ROOT/runs/val-$tag-cache-centre-max"
    log "gamma=$g done in $((($(date +%s) - start) / 60)) min;" \
        "summary=$([[ -f $d/aggregate/results-summary.json ]] && echo yes || echo NO)"
    "$HERE/gamma_validate_report.sh" 2>&1 | tee -a "$LOG"
done
log "=== validation done ==="
