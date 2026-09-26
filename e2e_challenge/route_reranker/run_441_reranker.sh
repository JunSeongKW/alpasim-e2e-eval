#!/usr/bin/env bash
# The full 441-clip evaluation with the chosen gamma, under the exact contract
# the axe-v9 baseline was scored with.
#
#   GAMMA=0.02 e2e_challenge/route_reranker/run_441_reranker.sh
#
# Everything except the reranker is held to leaderboard-merged-route-ep30: the
# same image and checkpoint, dev preset, all 441 curated_val clips, one rollout
# each, 16 drivers / 16 renderers / 16 workers on four cards, gains lat 1.0 /
# lon 0.25 / idx 3, no video. Sixteen workers is not a throughput choice here:
# concurrency moves gRPC timing, and at four workers the same code disagreed
# with the baseline on 8 of 38 clips. Matching the contract is what makes the
# baseline a paired reference for every one of the 441 clips.
#
# Arm is cache-centre-max: accumulated route, body-centre measurement, worst-
# moment aggregation. Gamma is the one thing this run varies from the sweeps.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/e2e_challenge/route_reranker"
cd "$ROOT"
GAMMA="${GAMMA:?set GAMMA, e.g. GAMMA=0.02}"
GPUS="${GPUS:-0,1,2,3}"
# Which weights. The default is the axe-v9 submission image; a different
# checkpoint is evaluated by building the same overlay on the same base
# (Dockerfile.checkpoint_overlay) and naming it here, so the only thing
# that moves between two runs is the weights.
IMG="${IMG:-alpasim-e2e-drivesuprim-stage3:merged-route-ep30}"
CKPT_SHA="${CKPT_SHA:-364c801e397e9d6e36ea1f9027dc4ca804ad2e429bb588cf4e84185ca851de3d}"
TAG_SUFFIX="${TAG_SUFFIX:-}"
tag="g${GAMMA//./p}${TAG_SUFFIX}"
RUN_TAG="rr441-$tag"
RUN_DIR="$ROOT/runs/$RUN_TAG-cache-centre-max"
LOG="$HERE/441_${tag}.log"
log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

[[ ! -e "$RUN_DIR" ]] || { log "refusing to overwrite $RUN_DIR"; exit 2; }

mapfile -t used < <(nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader,nounits)
for mib in "${used[@]}"; do
    (( mib < 4000 )) || { log "GPUs $GPUS not free (used MiB: ${used[*]}); refusing"; exit 2; }
done

cat > "$RUN_DIR.provenance.json" <<EOF
{
  "image": "$IMG",
  "checkpoint_sha256": "$CKPT_SHA",
  "baseline_run": "runs/leaderboard-merged-route-ep30",
  "arm": "cache-centre-max",
  "gamma": $GAMMA,
  "preset": "dev", "scenes": 441, "rollouts_per_scene": 1,
  "gpus": "$GPUS", "drivers": 16, "workers": 16, "renderer_replicas_per_gpu": 4,
  "mpc": {"long_position_weight": 0.25, "lat_position_weight": 1.0, "idx_start_penalty": 3},
  "git_commit": "$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
}
EOF

log "=== 441 clips, gamma=$GAMMA, arm=cache-centre-max, 16/16/16 on GPUs $GPUS ==="
start=$(date +%s)
# CLIPS unset -> run_10clips_reranker.sh falls back to clips10.txt, so the 441
# set is expressed by passing the curated_val suite through: an empty
# SCENE_IDS_FILE makes run_curated_val.sh keep +nurec_scenes=curated_val.
ARM=cache-centre-max RUN_TAG="$RUN_TAG" \
IMG="$IMG" EXPECTED_CHECKPOINT_SHA256="$CKPT_SHA" \
ROUTE_RERANK_WEIGHT="$GAMMA" \
CLIPS="" FULL_SET=1 REPLICAS=4 ROLLOUT_WORKERS=16 WATCH_LIMIT=20 \
RENDER_VIDEO=false \
BASE_PORT=7400 WIZARD_BASEPORT=19700 GPUS="$GPUS" RENDER_GPUS="$GPUS" \
    "$HERE/run_10clips_reranker.sh" > "$HERE/441_${tag}.nohup" 2>&1
status=$?
log "launcher exit=$status after $((($(date +%s) - start) / 60)) min"

if [[ -f "$RUN_DIR/aggregate/results-summary.json" ]]; then
    freed="$(du -sm "$RUN_DIR/rollouts" 2>/dev/null | cut -f1)"
    rm -rf "$RUN_DIR/rollouts" "$RUN_DIR/txt-logs" "$RUN_DIR/controller"
    mv "$RUN_DIR.provenance.json" "$RUN_DIR/evaluation_provenance.json"
    log "OK: $RUN_DIR/aggregate/results-summary.json; removed raw rollouts (~${freed:-0} MB)"
    "$ROOT/.venv/bin/python" "$ROOT/e2e_challenge/axe_local_eval/compare_on_clips.py" \
        "gamma$GAMMA=$RUN_DIR/aggregate/results-summary.json" \
        "axe-v9=$ROOT/runs/leaderboard-merged-route-ep30/aggregate/results-summary.json" \
        2>&1 | tee -a "$LOG"
else
    log "FAILED: no summary; preserving run evidence"
    exit "${status:-1}"
fi
