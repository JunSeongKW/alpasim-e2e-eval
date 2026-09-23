#!/usr/bin/env bash
# Render visualization video for the axe-v9 submission candidate on 10 clips.
#
# The point is footage, not a score, so this differs from a scoring run in three
# ways: rollouts are kept (the video is built from them), every video panel is
# switched on, and the clip list is a fixed random sample rather than the full
# suite. Everything that affects driving -- image, gains, preset -- is held to
# what the 441-clip run used, so the footage shows the submitted policy rather
# than a variant of it.
#
# The five panels come from eval/video.py's layout: map (top left), the model's
# own BEV (top centre), the metrics table (top right), the camera with plans
# overlaid (bottom, spanning two columns), and the ranking decomposition
# (bottom right). The BEV and ranking panels are fed by the driver's debug
# payload, which is built unconditionally, so the submission image produces them
# with no extra environment.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
AXE="$ROOT/e2e_challenge/axe_local_eval"
LOG="$AXE/viz10_axe_v9.log"
RUN_DIR="$ROOT/runs/viz10-axe-v9"
IMG=696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v9
log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

# Two replicas of everything, not the sixteen a scoring run uses. Ten clips do
# not need the parallelism, and the startup of each controller replica is a
# metadata-heavy `uv run` over a venv: eight at once put every one of them into
# uninterruptible disk wait for seventeen minutes while another researcher's
# training job saturated the ext4 journal. Fewer replicas start slower but they
# start.
rm -rf "$RUN_DIR"

log "=== viz10: starting 2 drivers on GPUs 4-5 ==="
addrs="$(IMAGE="$IMG" \
    EXPECTED_CHECKPOINT_SHA256=364c801e397e9d6e36ea1f9027dc4ca804ad2e429bb588cf4e84185ca851de3d \
    OFFICIAL_ENV_ONLY=1 \
    CONTAINER_PREFIX="axe-viz10" \
    BASE_PORT=6900 \
    GPU_INDICES_CSV=4,5 REPLICAS_PER_GPU=1 \
    "$AXE/start_drivers.sh" 2>>"$LOG" | tail -1)"
[[ "$addrs" == \[* ]] || { log "drivers failed"; exit 1; }

# Panels the scoring runs leave at their defaults. generate_combined_video
# stitches the ten clips into one file; overlay_plans_on_camera draws the
# candidate trajectories onto the camera view; render_every_nth_frame=1 keeps
# every frame so slow review is possible.
VIDEO_OVERRIDES=(
    # The two right-hand panels -- the metrics table and the ranking-influence
    # bars -- are fed by the driver's own debug payload, and the dev preset
    # turns the parsing of it off (dev.yaml sets parse_unstructured_debug_info
    # false, over a base default of true, because the field is pickle-encoded
    # and an untrusted driver image should not be unpickled). That is the right
    # default for scoring a stranger's image and the wrong one here, where the
    # image is ours and the panels are the whole point.
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

log "=== viz10: 10 clips x 1 rollout (dev), gains lat=1.0 lon=0.25 idx=3, video ON ==="
RUN_DIR="$RUN_DIR" \
PRESET=dev \
CONTESTANT_IMAGE="$IMG" \
DRIVER_ADDRESSES="$addrs" \
SCENE_IDS_FILE="$AXE/viz10_clips.txt" \
N_ROLLOUTS=1 \
ROLLOUT_WORKERS=2 \
RENDER_GPUS_CSV=4,5,6,7 \
RENDERER_REPLICAS_PER_GPU=1 \
NRE_CACHE_SIZE=1 \
ENABLE_AUTORESUME=false \
SERVICE_STARTUP_TIMEOUT_SEC=1800 \
RENDER_VIDEO=true \
KEEP_ROLLOUTS=1 \
MPC_OVERRIDES="controller.gains.long_position_weight=0.25 controller.gains.lat_position_weight=1.0 controller.gains.idx_start_penalty=3" \
EXTRA_OVERRIDES="${VIDEO_OVERRIDES[*]}" \
    "$AXE/run_curated_val.sh" >> "$RUN_DIR.wizard.log" 2>&1
log "viz10: wizard exit=$?"

docker ps -aq --filter "name=axe-viz10" | xargs -r docker rm -f >/dev/null 2>&1 || true

n="$(find "$RUN_DIR" -name '*.mp4' 2>/dev/null | wc -l)"
if (( n > 0 )); then
    log "viz10: OK, $n video file(s)"
    find "$RUN_DIR" -name '*.mp4' -printf '  %s  %p\n' | sort -k2 | tee -a "$LOG"
else
    log "viz10: no video produced"
    exit 1
fi
