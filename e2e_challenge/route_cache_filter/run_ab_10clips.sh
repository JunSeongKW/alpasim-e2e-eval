#!/usr/bin/env bash
# Before/after footage for the route-cache corridor filter, on ten clips that
# axe-v9 lost to the lateral-corridor gate.
#
# Both arms run the same image, the same checkpoint, the same MPC gains, the
# same preset and the same clips; the mounted driver package is the same files
# too. The only difference is DRIVESUPRIM_ROUTE_CACHE. Two of the ten clips are
# controls whose corridor exit happens before the cache can have closed the
# near field, so the feature has nothing to act on there -- if those two change,
# the change is a bug rather than an effect.
#
# Video settings follow run_viz10_axe_v9.sh: every panel on, every frame kept,
# and parse_unstructured_debug_info forced true because the dev preset disables
# it and the BEV panel -- which is where the cached route and the rejected
# candidates are drawn -- is fed from that payload.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/e2e_challenge/route_cache_filter"
AXE="$ROOT/e2e_challenge/axe_local_eval"
cd "$ROOT"

IMG="${IMG:-alpasim-e2e-drivesuprim-stage3:merged-route-ep30}"
RUN_TAG="${RUN_TAG:-routecache}"
LOG="$HERE/ab_10clips_${RUN_TAG}.log"
CLIPS="$HERE/clips10.txt"
GPUS="${GPUS:-4,5}"
RENDER_GPUS="${RENDER_GPUS:-4,5,6}"   # 7 is another researcher's
ARMS="${ARMS:-before after}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-2}"
RENDERER_REPLICAS_PER_GPU="${RENDERER_REPLICAS_PER_GPU:-1}"
ROUTE_GATE_HORIZON_S="${ROUTE_GATE_HORIZON_S:-4.0}"

log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

VIDEO_OVERRIDES=(
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

run_arm() {
    local arm="$1" route_cache="$2"
    local run_dir="$ROOT/runs/$RUN_TAG-$arm"
    log "=== arm=$arm  DRIVESUPRIM_ROUTE_CACHE=$route_cache  ROUTE_GATE_HORIZON=${ROUTE_GATE_HORIZON_S}s ==="
    rm -rf "$run_dir"

    local addrs
    addrs="$(IMAGE="$IMG" \
        CONTAINER_PREFIX="axe-rc-$arm" \
        BASE_PORT=6940 \
        GPU_INDICES_CSV="$GPUS" REPLICAS_PER_GPU=1 \
        DRIVER_PKG="$HERE/drivesuprim_challenge" \
        ROUTE_CACHE="$route_cache" \
        ROUTE_GATE_HORIZON_S="$ROUTE_GATE_HORIZON_S" \
        "$HERE/start_drivers_routecache.sh" 2>>"$LOG" | tail -1)"
    if [[ "$addrs" != \[* ]]; then
        log "arm=$arm: drivers failed to start"
        return 1
    fi
    log "arm=$arm: drivers at $addrs"

    # `uv run` revalidates the direct-URL dependencies at every launch, and
    # github.com answering 504 is then enough to stop a run that needs nothing
    # from the network -- drive-irt is already in the uv cache and installed in
    # the venv. Offline mode makes uv use what it has instead of asking.
    RUN_DIR="$run_dir" \
    UV_OFFLINE=1 \
    PRESET=dev \
    CONTESTANT_IMAGE="$IMG" \
    DRIVER_ADDRESSES="$addrs" \
    SCENE_IDS_FILE="$CLIPS" \
    N_ROLLOUTS=1 \
    ROLLOUT_WORKERS="$ROLLOUT_WORKERS" \
    RENDER_GPUS_CSV="$RENDER_GPUS" \
    RENDERER_REPLICAS_PER_GPU="$RENDERER_REPLICAS_PER_GPU" \
    NRE_CACHE_SIZE=1 \
    ENABLE_AUTORESUME=false \
    SERVICE_STARTUP_TIMEOUT_SEC=1800 \
    RENDER_VIDEO=true \
    KEEP_ROLLOUTS=1 \
    MPC_OVERRIDES="controller.gains.long_position_weight=0.25 controller.gains.lat_position_weight=1.0 controller.gains.idx_start_penalty=3" \
    EXTRA_OVERRIDES="${VIDEO_OVERRIDES[*]}" \
        "$AXE/run_curated_val.sh" >> "$run_dir.wizard.log" 2>&1
    log "arm=$arm: wizard exit=$?"

    # Keep the driver logs: the ROUTEGATE lines are the only record of how many
    # candidates the filter dropped per frame, and the containers go next.
    for c in $(docker ps -aq --filter "name=axe-rc-$arm"); do
        docker logs "$c" > "$run_dir.driver-$c.log" 2>&1 || true
    done
    docker ps -aq --filter "name=axe-rc-$arm" | xargs -r docker rm -f >/dev/null 2>&1 || true

    local n
    n="$(find "$run_dir" -name '*.mp4' 2>/dev/null | wc -l)"
    log "arm=$arm: $n video file(s)"
}

for arm in $ARMS; do
    case "$arm" in
        before) run_arm before 0 ;;
        after)  run_arm after  1 ;;
        *) log "unknown arm: $arm"; exit 2 ;;
    esac
done

log "=== done ==="
for arm in $ARMS; do
    d="$ROOT/runs/$RUN_TAG-$arm/aggregate/results-summary.json"
    [[ -f "$d" ]] && log "$arm: $d"
done
