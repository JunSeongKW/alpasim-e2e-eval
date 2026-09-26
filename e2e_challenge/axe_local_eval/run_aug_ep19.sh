#!/usr/bin/env bash
# stage3_aug_ep19 on the local leaderboard, under the axe-v9 contract exactly.
#
# This is run_merged_route_ep30.sh with one substitution: the image carries
# stage3_aug_ep19 instead of stage3_merged_route_ep30, built by the same overlay
# on the same axe-v9 base so the runtime, entrypoint, workdir and user are
# byte-identical and only the weights differ. NO route reranker: the driver is
# the image's own, started by the same start_drivers.sh with OFFICIAL_ENV_ONLY,
# so the container receives only the four variables the official evaluator
# supplies and none of the reranker's files are mounted over it.
#
# Everything the fit depends on is held: dev preset, the same 441 curated_val
# scenes, one rollout each, 16 drivers and 16 renderers, gains lat 1.0 /
# lon 0.25 / idx 3, no video. The scene set in particular has to match, because
# ZOIB is fitted across subjects per scene.
#
# Two deviations from the baseline run, neither a scored quantity:
#   * GPUs 0-3 rather than 4-7. Another researcher's ep30 stack holds 4-7; the
#     cards are placement only.
#   * run_curated_val.sh now defaults FAST_STARTUP=1, which adds
#     PYTHONDONTWRITEBYTECODE and a seeded renderer cache. Those decide how long
#     startup takes, not what the renderer produces -- without them sixteen
#     renderers took down four attempts at a different evaluation.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
AXE="$ROOT/e2e_challenge/axe_local_eval"
LOG="$AXE/aug_ep19.log"
RUN_DIR="$ROOT/runs/leaderboard-aug-ep19"
IMG=alpasim-e2e-axe-v9:stage3-aug-ep19
SHA=9843a91da10e2d8df3173dc917fa9877562952b97132182d532c0bb2a4088394
GPUS="${GPUS:-0,1,2,3}"
PREFIX=axe-lb-aug19
log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

[[ ! -e "$RUN_DIR" ]] || { log "refusing to overwrite $RUN_DIR"; exit 2; }
actual="$(docker image inspect "$IMG" \
    --format '{{index .Config.Labels "org.alpasim.checkpoint.sha256"}}' 2>/dev/null || true)"
[[ "$actual" == "$SHA" ]] || { log "image checkpoint mismatch: ${actual:-missing}"; exit 2; }

mapfile -t used < <(nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader,nounits)
for mib in "${used[@]}"; do
    (( mib < 4000 )) || { log "GPUs $GPUS not free (used MiB: ${used[*]})"; exit 2; }
done

cat > "$RUN_DIR.provenance.json" <<EOF
{
  "checkpoint": "$ROOT/../models/stage3_aug_ep19.ckpt",
  "checkpoint_sha256": "$SHA",
  "image": "$IMG",
  "base_runtime": "696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v9",
  "route_reranker": "off",
  "baseline_run": "runs/leaderboard-merged-route-ep30",
  "preset": "dev", "scenes": 441, "rollouts_per_scene": 1,
  "gpus": "$GPUS", "drivers": 16, "workers": 16, "renderer_replicas_per_gpu": 4,
  "official_env_only": true,
  "mpc": {"long_position_weight": 0.25, "lat_position_weight": 1.0, "idx_start_penalty": 3},
  "git_commit": "$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
}
EOF

log "=== aug-ep19: starting 16 drivers on GPUs $GPUS (no reranker) ==="
addrs="$(IMAGE="$IMG" \
    EXPECTED_CHECKPOINT_SHA256="$SHA" \
    OFFICIAL_ENV_ONLY=1 \
    CONTAINER_PREFIX="$PREFIX" \
    BASE_PORT=7500 \
    GPU_INDICES_CSV="$GPUS" REPLICAS_PER_GPU=4 \
    "$AXE/start_drivers.sh" 2>>"$LOG" | tail -1)"
[[ "$addrs" == \[* ]] || { log "drivers failed"; exit 1; }
log "drivers: $addrs"

log "=== aug-ep19: 441 scenes x 1 rollout (dev), gains lat=1.0 lon=0.25 idx=3 ==="
# wizard.baseport moves our service ports off 6000, which another
# researcher's ep30 stack is holding. A comment must never sit inside the
# assignment chain below: the backslash would swallow the rest of the line
# and the variables would become shell locals the child never sees.
start=$(date +%s)
RUN_DIR="$RUN_DIR" \
PRESET=dev \
CONTESTANT_IMAGE="$IMG" \
DRIVER_ADDRESSES="$addrs" \
N_ROLLOUTS=1 \
ROLLOUT_WORKERS=16 \
RENDER_GPUS_CSV="$GPUS" \
RENDERER_REPLICAS_PER_GPU=4 \
NRE_CACHE_SIZE=1 \
ENABLE_AUTORESUME=true \
SERVICE_STARTUP_TIMEOUT_SEC=1800 \
MPC_OVERRIDES="controller.gains.long_position_weight=0.25 controller.gains.lat_position_weight=1.0 controller.gains.idx_start_penalty=3" \
EXTRA_OVERRIDES="wizard.baseport=19900" \
    "$AXE/run_curated_val.sh" >> "$RUN_DIR.wizard.log" 2>&1
log "wizard exit=$? after $((($(date +%s) - start) / 60)) min"

docker ps -aq --filter "name=$PREFIX" | xargs -r docker rm -f >/dev/null 2>&1 || true

# run_curated_val.sh ends in `exec`, so its own EXIT trap never fires and a
# 441-scene run leaves ~175 GB of rollout.asl behind. Only the summary is read
# back, so sweep them here, and only once that summary exists.
if [[ -f "$RUN_DIR/aggregate/results-summary.json" ]]; then
    freed="$(du -sm "$RUN_DIR/rollouts" 2>/dev/null | cut -f1)"
    rm -rf "$RUN_DIR/rollouts" "$RUN_DIR/txt-logs" "$RUN_DIR/controller"
    mv "$RUN_DIR.provenance.json" "$RUN_DIR/evaluation_provenance.json" 2>/dev/null || true
    log "OK -> $RUN_DIR/aggregate/results-summary.json (removed ~${freed:-0} MB of rollouts)"
    "$ROOT/.venv/bin/python" "$AXE/compare_on_clips.py" \
        "aug-ep19=$RUN_DIR/aggregate/results-summary.json" \
        "axe-v9=$ROOT/runs/leaderboard-merged-route-ep30/aggregate/results-summary.json" \
        2>&1 | tee -a "$LOG"
else
    log "FAILED, no summary"
    exit 1
fi
