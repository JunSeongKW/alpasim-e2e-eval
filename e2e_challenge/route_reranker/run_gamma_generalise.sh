#!/usr/bin/env bash
# Does the gamma that rescued one turn clip hold on the rest of them?
#
# One clip told us gamma has to reach about 0.05 before the reranker overturns a
# corridor failure, and that 0.01 traded that failure for a collision. Neither
# number means anything yet: a value tuned on a single rollout is a value tuned
# on noise, and a gate that fixes failures by driving more conservatively will
# also spoil the clips that already work. This runs the same arm over every
# comparable failure AND over clips the baseline already passes, so both halves
# of that question are answered by the same sweep.
#
# CLIPS (e2e_challenge/route_reranker/clips_gamma_set.txt, 38 of them)
#   26 failures -- every clip of the 441 where axe-v9 left the lateral corridor,
#      minus the ones that failed before 45 m (the cache cannot have closed the
#      near field yet, so the reranker has nothing to act on) and minus the ones
#      that failed past the end of the recorded drive, which is a scoring
#      artefact rather than a driving mistake. 40 corridor failures, 26 usable.
#   12 controls -- clips the baseline scores above 0.95, sampled across the same
#      range of travelled distance so the control set is not accidentally made of
#      short easy ones. A gamma that fixes failures and breaks these is not a
#      better gamma.
#
# Turn-ness is decided AFTER the run, not before it: nothing in the run
# directory or the scene artefacts records whether a clip turns, so the driver
# logs the heading change of the route it is measuring against, and
# gamma_generalise_report.sh splits the table on it. That is the same geometry
# the cost is computed from, which is the honest thing to classify on.
#
# NO VIDEO. Rendering every frame of 38 clips at five gammas is many hours for
# footage that answers none of this; the numbers decide the value and the video
# comes afterwards for whichever gamma wins. Rollouts are kept so that footage
# can be redrawn without simulating again.
#
# Gammas run one at a time across all four cards rather than four at once on one
# card each: a whole card per renderer roughly halves the wall clock per clip,
# and a sweep that reports its first arm in ninety minutes is more useful than
# one that reports nothing for eight hours.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/e2e_challenge/route_reranker"
cd "$ROOT"
LOG="$HERE/gamma_generalise.log"
log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

# 0 first: it is the paired baseline on this exact configuration, and it is the
# arm whose driver logs classify every clip as turn or straight. Then the value
# that worked on the single clip, then its neighbours.
read -r -a GAMMAS <<< "${GAMMAS:-0 0.05 0.02 0.1 0.01}"
CLIPS="${CLIPS:-$HERE/clips_gamma_set.txt}"
n_clips="$(grep -vc '^[[:space:]]*$' "$CLIPS")"

for idx in "${!GAMMAS[@]}"; do
    g="${GAMMAS[$idx]}"
    tag="g${g//./p}"
    log "=== gamma=$g  ($((idx + 1))/${#GAMMAS[@]}, $n_clips clips, 4 drivers on GPUs 0-3) ==="
    start=$(date +%s)
    ARM=cache-centre-max RUN_TAG="gen-$tag" \
    ROUTE_RERANK_WEIGHT="$g" \
    CLIPS="$CLIPS" REPLICAS=1 ROLLOUT_WORKERS=4 WATCH_LIMIT=10 \
    RENDER_VIDEO=false \
    BASE_PORT=7200 WIZARD_BASEPORT=19000 GPUS=0,1,2,3 RENDER_GPUS=0,1,2,3 \
        "$HERE/run_10clips_reranker.sh" > "$HERE/gen_${tag}.nohup" 2>&1
    d="$ROOT/runs/gen-$tag-cache-centre-max"
    log "gamma=$g done in $((($(date +%s) - start) / 60)) min;" \
        "summary=$([[ -f $d/aggregate/results-summary.json ]] && echo yes || echo NO)"
    "$HERE/gamma_generalise_report.sh" 2>&1 | tee -a "$LOG"
done

log "=== sweep done ==="
"$HERE/gamma_generalise_report.sh" 2>&1 | tee -a "$LOG"
