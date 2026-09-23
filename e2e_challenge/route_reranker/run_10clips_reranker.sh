#!/usr/bin/env bash
# One arm of the route-reranker comparison on the ten lateral-corridor clips.
#
#   ARM=before   reranker off  (the submitted axe-v9 behaviour)
#   ARM=rerank   reranker on   (mean-L2, gamma 0.0005, 8 m span gate)
#
# Same image, checkpoint, gains, preset, clips and mounted files in both arms;
# DRIVESUPRIM_ROUTE_RERANK is the only difference. The two arms are meant to
# run side by side on the same cards, so each gets its own driver ports and its
# own wizard port block -- the simulator services bind on the host network and
# another evaluation may already own 6000+.
#
# Video settings follow route_cache_filter/run_ab_10clips.sh: every frame kept,
# the BEV panel fed from the driver's debug payload (that is where the route
# message and the pre-rerank winner are drawn), and rollouts retained so the
# footage can be redrawn without re-simulating.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/e2e_challenge/route_reranker"
AXE="$ROOT/e2e_challenge/axe_local_eval"
cd "$ROOT"

ARM="${ARM:?set ARM=before or ARM=rerank}"
case "$ARM" in
    before) ROUTE_RERANK=0 ;;
    rerank) ROUTE_RERANK=1 ;;
    *) echo "unknown ARM=$ARM" >&2; exit 2 ;;
esac
IMG="${IMG:-alpasim-e2e-drivesuprim-stage3:merged-route-ep30}"
RUN_TAG="${RUN_TAG:-rerank10}"
RUN_DIR="$ROOT/runs/$RUN_TAG-$ARM"
LOG="$HERE/10clips_${RUN_TAG}_${ARM}.log"
CLIPS="${CLIPS:-$ROOT/e2e_challenge/route_cache_filter/clips10.txt}"
GPUS="${GPUS:-0,1,2,3}"
RENDER_GPUS="${RENDER_GPUS:-$GPUS}"
REPLICAS="${REPLICAS:-1}"
BASE_PORT="${BASE_PORT:-6940}"
WIZARD_BASEPORT="${WIZARD_BASEPORT:-18000}"
ROUTE_RERANK_WEIGHT="${ROUTE_RERANK_WEIGHT:-0.0005}"
IFS=',' read -r -a gpu_list <<< "$GPUS"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-$(( REPLICAS * ${#gpu_list[@]} ))}"

log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

VIDEO_OVERRIDES=(
    "wizard.baseport=${WIZARD_BASEPORT}"
    "eval.parse_unstructured_debug_info=true"
    "eval.video.video_layouts=[DEFAULT]"
    "eval.video.overlay_plans_on_camera=true"
    "eval.video.render_every_nth_frame=1"
    "eval.video.generate_combined_video=true"
    "eval.video.combined_video_speed_factor=0.33"
    "eval.video.camera_id_to_render=camera_front_wide_120fov"
    "eval.video.map_video.map_radius_m=20"
    "eval.video.map_video.rotate_map_to_ego=true"
    "eval.video.map_video.ego_loc=BOTTOM_CENTER"
)

log "=== arm=$ARM  DRIVESUPRIM_ROUTE_RERANK=$ROUTE_RERANK weight=$ROUTE_RERANK_WEIGHT  drivers on $GPUS, renderers on $RENDER_GPUS, ${ROLLOUT_WORKERS} workers, wizard ports from $WIZARD_BASEPORT ==="
rm -rf "$RUN_DIR"

PREFIX="axe-rr-$RUN_TAG-$ARM"
addrs="$(IMAGE="$IMG" \
    CONTAINER_PREFIX="$PREFIX" \
    BASE_PORT="$BASE_PORT" \
    GPU_INDICES_CSV="$GPUS" REPLICAS_PER_GPU="$REPLICAS" \
    ROUTE_RERANK="$ROUTE_RERANK" ROUTE_RERANK_WEIGHT="$ROUTE_RERANK_WEIGHT" \
    "$HERE/start_drivers_reranker.sh" 2>>"$LOG" | tail -1)"
[[ "$addrs" == \[* ]] || { log "arm=$ARM: drivers failed to start"; exit 1; }
log "arm=$ARM: drivers at $addrs"

RUN_DIR="$RUN_DIR" \
PRESET=dev \
CONTESTANT_IMAGE="$IMG" \
DRIVER_ADDRESSES="$addrs" \
SCENE_IDS_FILE="$CLIPS" \
N_ROLLOUTS=1 \
ROLLOUT_WORKERS="$ROLLOUT_WORKERS" \
RENDER_GPUS_CSV="$RENDER_GPUS" \
RENDERER_REPLICAS_PER_GPU="$REPLICAS" \
NRE_CACHE_SIZE=1 \
ENABLE_AUTORESUME=false \
SERVICE_STARTUP_TIMEOUT_SEC=1800 \
RENDER_VIDEO=true \
KEEP_ROLLOUTS=1 \
MPC_OVERRIDES="controller.gains.long_position_weight=0.25 controller.gains.lat_position_weight=1.0 controller.gains.idx_start_penalty=3" \
EXTRA_OVERRIDES="${VIDEO_OVERRIDES[*]}" \
    "$AXE/run_curated_val.sh" >> "$RUN_DIR.wizard.log" 2>&1
log "arm=$ARM: wizard exit=$?"

# The RERANK lines are the only per-frame record of what the selector did.
for c in $(docker ps -aq --filter "name=$PREFIX"); do
    docker logs "$c" > "$RUN_DIR.driver-$c.log" 2>&1 || true
done
docker ps -aq --filter "name=$PREFIX" | xargs -r docker rm -f >/dev/null 2>&1 || true

n="$(find "$RUN_DIR" -name '*.mp4' 2>/dev/null | wc -l)"
log "arm=$ARM: $n video file(s); summary: $([[ -f "$RUN_DIR/aggregate/results-summary.json" ]] && echo yes || echo NO)"
