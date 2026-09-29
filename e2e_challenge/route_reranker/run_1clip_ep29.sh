#!/usr/bin/env bash
# One clip, aug-ep29-final with the route reranker at gamma=0.01, with video.
#
#   e2e_challenge/route_reranker/run_1clip_ep29.sh
#
# The clip is eaba3f6a, the representative failing turn the gamma sweep was
# built around, so this run is directly comparable with that sweep's videos.
# Gamma 0.01 is what the 441-clip ladder chose for this checkpoint (0.7180,
# the ladder's best).
#
# Video is ON here, unlike every 441 run: with one clip the render costs a few
# minutes instead of hours, and the point of a single clip is to watch it.
# RENDER_VIDEO=true also makes generate_combined_video meaningful, so the arm
# produces both the per-clip camera view and the combined overlay.
#
# One clip needs one driver and one worker. Sixteen would idle fifteen of them,
# and worker count changes gRPC timing -- but that only matters for comparing
# aggregate scores across runs, which a single clip is not for. The gamma
# sweep's own single-clip runs used exactly this shape (REPLICAS=1,
# ROLLOUT_WORKERS=1, one card), so the comparison against them is fair.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/e2e_challenge/route_reranker"
cd "$ROOT"

GAMMA="${GAMMA:-0.01}"
# ARM=before turns the reranker off entirely -- the control for "what does the
# checkpoint do on its own". Any other arm keeps the reranker on; cache-centre-max
# is the one the gamma ladder used, so gamma is only meaningful with that.
ARM="${ARM:-cache-centre-max}"
IMG="${IMG:-alpasim-e2e-axe-v9:stage3-aug-ep29-final}"
CKPT_SHA="${CKPT_SHA:-29c6fca63c4d4bd561fb90101ebbf39f7cb23b87a09251b1db3a766f05ffdef8}"
CARD="${CARD:-4}"
BASE_PORT="${BASE_PORT:-8300}"
WIZARD_BASEPORT="${WIZARD_BASEPORT:-21700}"
RUN_TAG="${RUN_TAG:-clip1-ep29-g${GAMMA//./p}}"
LOG="$HERE/${RUN_TAG}.log"
log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

RUN_DIR="$ROOT/runs/$RUN_TAG-$ARM"
[[ ! -e "$RUN_DIR" ]] || { log "refusing to overwrite $RUN_DIR"; exit 2; }

used="$(nvidia-smi -i "$CARD" --query-gpu=memory.used --format=csv,noheader,nounits)"
(( used < 4000 )) || { log "GPU $CARD not free (${used} MiB); refusing"; exit 2; }

log "=== 1 clip, ep29, arm=$ARM gamma=$GAMMA, video ON, GPU $CARD ==="
log "clip: $(cat "$HERE/clip1_turn.txt")"
start=$(date +%s)
ARM="$ARM" RUN_TAG="$RUN_TAG" \
IMG="$IMG" EXPECTED_CHECKPOINT_SHA256="$CKPT_SHA" \
ROUTE_RERANK_WEIGHT="$GAMMA" \
CLIPS="$HERE/clip1_turn.txt" REPLICAS=1 ROLLOUT_WORKERS=1 WATCH_LIMIT=5 \
RENDER_VIDEO=true \
BASE_PORT="$BASE_PORT" WIZARD_BASEPORT="$WIZARD_BASEPORT" \
GPUS="$CARD" RENDER_GPUS="$CARD" \
    "$HERE/run_10clips_reranker.sh" > "$HERE/${RUN_TAG}.nohup" 2>&1
log "launcher exit=$? after $((($(date +%s) - start) / 60)) min"

if [[ -f "$RUN_DIR/aggregate/results-summary.json" ]]; then
    "$ROOT/.venv/bin/python" - "$RUN_DIR/aggregate/results-summary.json" <<'EOF'
import json, sys
rs = json.load(open(sys.argv[1]))["rollouts"]
for r in rs:
    m = r.get("metrics") or {}
    sm = r.get("score_metrics") or {}
    print(f"clip {r['clipgt_id']}")
    print(f"  score            {r.get('score')}")
    print(f"  failure_reason   {r.get('failure_reason')}")
    for k in ("collision_at_fault", "offroad", "left_corridor_laterally",
              "progress_clipped_rel", "gt_dist_traveled_m",
              "lateral_dist_to_gt_trajectory", "dist_traveled_m",
              "min_distance_to_obstacle_m", "duration_frac_20s"):
        v = sm.get(k, m.get(k))
        if v is not None:
            print(f"  {k:<28}{v}")
EOF
    log "videos:"
    find "$RUN_DIR" -name '*.mp4' -printf '  %10s  %p\n' 2>/dev/null | sort -k2 | tee -a "$LOG"
else
    log "FAILED: no summary"
    exit 1
fi
log "=== done: $RUN_DIR ==="
