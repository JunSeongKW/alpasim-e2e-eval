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

# Arms. Everything except the named switch is held constant: same image, same
# checkpoint, same MPC gains, same preset, same ten clips, same mounted files.
#
#   before        reranker off -- the submitted axe-v9 behaviour
#   rerank        bundle as archived: raw 42-80 m route, pose, mean over 4 s
#   cache         + accumulated route, so the near field exists to measure against
#   cache-centre  + measure from the body centre (1.467 m ahead of the pose),
#                 which is the point the scorer's corridor metric uses
#   cache-max     + worst moment instead of the average one, matching the
#                 scorer's "leave the corridor once and the rollout is zero"
#   cache-centre-max  both variants together
ARM="${ARM:?set ARM=before|rerank|cache|cache-centre|cache-max|cache-centre-max}"
ROUTE_RERANK=1; ROUTE_RERANK_CACHE=0; ROUTE_RERANK_AGG=mean; ROUTE_RERANK_CENTRE_DX=0.0
# Score penalty per metre. The bundle's provisional default; the gamma sweep
# overrides it per run, which is the one thing arm names do not encode.
ROUTE_RERANK_WEIGHT="${ROUTE_RERANK_WEIGHT:-0.0005}"
case "$ARM" in
    before)           ROUTE_RERANK=0 ;;
    rerank)           ;;
    cache)            ROUTE_RERANK_CACHE=1 ;;
    cache-centre)     ROUTE_RERANK_CACHE=1; ROUTE_RERANK_CENTRE_DX=1.467 ;;
    cache-max)        ROUTE_RERANK_CACHE=1; ROUTE_RERANK_AGG=max ;;
    cache-centre-max) ROUTE_RERANK_CACHE=1; ROUTE_RERANK_AGG=max; ROUTE_RERANK_CENTRE_DX=1.467 ;;
    *) echo "unknown ARM=$ARM" >&2; exit 2 ;;
esac
IMG="${IMG:-alpasim-e2e-drivesuprim-stage3:merged-route-ep30}"
RUN_TAG="${RUN_TAG:-rerank10}"
RUN_DIR="$ROOT/runs/$RUN_TAG-$ARM"
LOG="$HERE/10clips_${RUN_TAG}_${ARM}.log"
CLIPS="${CLIPS:-$ROOT/e2e_challenge/route_cache_filter/clips10.txt}"
# FULL_SET=1 evaluates the whole curated_val suite instead of a clip list --
# the 441-clip run with the chosen gamma. SCENE_IDS_FILE is then left unset
# and run_curated_val.sh keeps +nurec_scenes=curated_val as the scene source.
FULL_SET="${FULL_SET:-0}"
# RESUME=1 keeps an existing run directory and lets the runtime skip the
# rollouts that carry a _complete marker. The validation sweep died at
# 24/40 when the session that had launched it ended; three arms of 40
# clips are an hour and a half, and two-thirds of it was already done.
RESUME="${RESUME:-0}"
GPUS="${GPUS:-0,1,2,3}"
RENDER_GPUS="${RENDER_GPUS:-$GPUS}"
REPLICAS="${REPLICAS:-1}"
BASE_PORT="${BASE_PORT:-6940}"
WIZARD_BASEPORT="${WIZARD_BASEPORT:-18000}"
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

log "=== arm=$ARM  rerank=$ROUTE_RERANK cache=$ROUTE_RERANK_CACHE agg=$ROUTE_RERANK_AGG centre_dx=$ROUTE_RERANK_CENTRE_DX gamma=$ROUTE_RERANK_WEIGHT  drivers on $GPUS, renderers on $RENDER_GPUS, ${ROLLOUT_WORKERS} workers, wizard ports from $WIZARD_BASEPORT ==="
if [[ "$RESUME" == 1 && -d "$RUN_DIR" ]]; then
    log "resuming $RUN_DIR: $(find "$RUN_DIR/rollouts" -name _complete 2>/dev/null | wc -l) rollouts already complete"
else
    rm -rf "$RUN_DIR"
fi

PREFIX="axe-rr-$RUN_TAG-$ARM"
addrs="$(IMAGE="$IMG" \
    CONTAINER_PREFIX="$PREFIX" \
    BASE_PORT="$BASE_PORT" \
    GPU_INDICES_CSV="$GPUS" REPLICAS_PER_GPU="$REPLICAS" \
    ROUTE_RERANK="$ROUTE_RERANK" \
    ROUTE_RERANK_CACHE="$ROUTE_RERANK_CACHE" \
    ROUTE_RERANK_AGG="$ROUTE_RERANK_AGG" \
    ROUTE_RERANK_CENTRE_DX="$ROUTE_RERANK_CENTRE_DX" \
    ROUTE_RERANK_WEIGHT="$ROUTE_RERANK_WEIGHT" \
    "$HERE/start_drivers_reranker.sh" 2>>"$LOG" | tail -1)"
[[ "$addrs" == \[* ]] || { log "arm=$ARM: drivers failed to start"; exit 1; }
log "arm=$ARM: drivers at $addrs"

# Startup gate. "policy load complete" proves the checkpoint loaded, not that
# the reranker runs: the first sweep drove eight clips with the reranker raising
# ModuleNotFoundError on every frame, because the driver catches inference
# errors and keeps the previous plan. So before any scene is simulated, run one
# forward through the mounted code inside a driver container and refuse to
# continue unless it logs a RERANK decision. Costs ~1 min; the alternative was
# two hours of invalid rollouts.
if [[ "$ROUTE_RERANK" == 1 ]]; then
    probe="$(docker ps --filter "name=^${PREFIX}-g" --format '{{.Names}}' | head -1)"
    if ! "$HERE/probe_reranker.sh" "$probe" >> "$LOG" 2>&1; then
        log "arm=$ARM: reranker probe FAILED in $probe -- not simulating on broken code"
        docker ps -aq --filter "name=$PREFIX" | xargs -r docker rm -f >/dev/null 2>&1 || true
        exit 3
    fi
    log "arm=$ARM: reranker probe OK"
fi

# Watch the driver logs while the arm drives: an exception the driver
# swallows is invisible in the run directory but fatal to the result.
"$HERE/watch_arm.sh" "$PREFIX" "$LOG" "${WATCH_LIMIT:-20}" &
watchdog=$!
trap 'kill "$watchdog" 2>/dev/null' EXIT

RUN_DIR="$RUN_DIR" \
PRESET=dev \
CONTESTANT_IMAGE="$IMG" \
DRIVER_ADDRESSES="$addrs" \
SCENE_IDS_FILE="$([[ "$FULL_SET" == 1 ]] && echo "" || echo "$CLIPS")" \
N_ROLLOUTS=1 \
ROLLOUT_WORKERS="$ROLLOUT_WORKERS" \
RENDER_GPUS_CSV="$RENDER_GPUS" \
RENDERER_REPLICAS_PER_GPU="$REPLICAS" \
NRE_CACHE_SIZE=1 \
ENABLE_AUTORESUME="$([[ "$RESUME" == 1 ]] && echo true || echo false)" \
SERVICE_STARTUP_TIMEOUT_SEC=1800 \
RENDER_VIDEO="${RENDER_VIDEO:-true}" \
KEEP_ROLLOUTS=1 \
MPC_OVERRIDES="controller.gains.long_position_weight=0.25 controller.gains.lat_position_weight=1.0 controller.gains.idx_start_penalty=3" \
EXTRA_OVERRIDES="${VIDEO_OVERRIDES[*]}" \
    "$AXE/run_curated_val.sh" >> "$RUN_DIR.wizard.log" 2>&1
log "arm=$ARM: wizard exit=$?"

# The RERANK lines are the only per-frame record of what the selector did.
for c in $(docker ps -aq --filter "name=$PREFIX"); do
    docker logs -t "$c" > "$RUN_DIR.driver-$c.log" 2>&1 || true
done
docker ps -aq --filter "name=$PREFIX" | xargs -r docker rm -f >/dev/null 2>&1 || true

n="$(find "$RUN_DIR" -name '*.mp4' 2>/dev/null | wc -l)"
log "arm=$ARM: $n video file(s); summary: $([[ -f "$RUN_DIR/aggregate/results-summary.json" ]] && echo yes || echo NO)"
