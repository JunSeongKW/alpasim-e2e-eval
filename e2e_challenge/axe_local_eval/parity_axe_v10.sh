#!/usr/bin/env bash
# Prove the baked axe-v10 image behaves like the mounted configuration that was
# measured, on one clip.
#
# This is the check that matters most before submitting. Every reranker number in
# this project came from a run where the feature was bind-mounted over the image
# and switched on with injected environment variables. The submission can do
# neither, so the code and the switches were baked into axe-v10 -- and "baked
# correctly" is an assumption until a run confirms it.
#
# The comparison is against runs/clip1-ep29-g0p01-cache-centre-max, the same clip
# under the mounted configuration:
#
#   score 0.6485338950771624   progress_clipped_rel 0.5188271160617299
#   dist_traveled_m 69.13925880638344   collision_rear 1   min_obstacle 0.0
#
# Everything the wizard sees is copied from that run: dev preset, one rollout,
# one worker, one renderer, video on, the same MPC gains, the same video/debug
# overrides. The only difference is how the reranker gets in -- here the driver is
# started by start_drivers.sh with OFFICIAL_ENV_ONLY=1, which is the shape the
# official evaluator uses: no mounts, and only the four ALPASIM_* variables. If
# the numbers match, the baking is faithful and the local measurement transfers.
#
# A mismatch would mean the submitted image is not the thing that was scored, and
# the submission should not go out.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AXE="$ROOT/e2e_challenge/axe_local_eval"
HERE="$ROOT/e2e_challenge/route_reranker"
cd "$ROOT"

IMG="${IMG:-696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v10}"
SHA="${SHA:-29c6fca63c4d4bd561fb90101ebbf39f7cb23b87a09251b1db3a766f05ffdef8}"
CARD="${CARD:-7}"
BASE_PORT="${BASE_PORT:-8400}"
WIZARD_BASEPORT="${WIZARD_BASEPORT:-22100}"
RUN_TAG="${RUN_TAG:-parity-v10}"
REF="${REF:-$ROOT/runs/clip1-ep29-g0p01-cache-centre-max}"
RUN_DIR="$ROOT/runs/$RUN_TAG"
LOG="$AXE/$RUN_TAG.log"
PREFIX="axe-parity-v10"
log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

[[ ! -e "$RUN_DIR" ]] || { log "refusing to overwrite $RUN_DIR"; exit 2; }
# One clip needs about 12 GB: 3.6 for the driver, the rest for one renderer plus
# physics/controller/runtime. What matters is free memory, not whether the card is
# empty -- the 441 launchers demand an empty card because they fill it, but a
# single clip can share one. NEED_FREE keeps a margin above the 12 GB so a
# neighbour that grows a little does not OOM because of this run.
NEED_FREE="${NEED_FREE:-15000}"
free="$(nvidia-smi -i "$CARD" --query-gpu=memory.free --format=csv,noheader,nounits)"
(( free >= NEED_FREE )) || { log "GPU $CARD has only ${free} MiB free, need ${NEED_FREE}"; exit 2; }
log "GPU $CARD: ${free} MiB free"

# The same overrides the mounted run used, so the eval config is identical.
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

log "=== parity: axe-v10 baked, NO mounts, OFFICIAL_ENV_ONLY, GPU $CARD ==="
log "clip: $(cat "$HERE/clip1_turn.txt")"
addrs="$(IMAGE="$IMG" EXPECTED_CHECKPOINT_SHA256="$SHA" OFFICIAL_ENV_ONLY=1 \
    CONTAINER_PREFIX="$PREFIX" BASE_PORT="$BASE_PORT" \
    GPU_INDICES_CSV="$CARD" REPLICAS_PER_GPU=1 READY_TIMEOUT_SEC=900 \
    "$AXE/start_drivers.sh" 2>>"$LOG" | tail -1)"
[[ "$addrs" == \[* ]] || { log "drivers failed"; exit 1; }
log "drivers: $addrs"

# The reranker must have switched itself on from the image alone.
rr="$(docker logs "$PREFIX-g$CARD-0" 2>&1 | grep -m1 'ROUTE RERANK:' || true)"
log "driver reports: ${rr:-<no ROUTE RERANK line>}"
case "$rr" in
    *"enabled=True"*"weight=0.01"*) log "reranker active from the image alone" ;;
    *) log "ABORT: reranker not enabled by the image"; docker rm -f "$PREFIX-g$CARD-0" >/dev/null 2>&1; exit 3 ;;
esac

start=$(date +%s)
RUN_DIR="$RUN_DIR" \
PRESET=dev \
CONTESTANT_IMAGE="$IMG" \
DRIVER_ADDRESSES="$addrs" \
SCENE_IDS_FILE="$HERE/clip1_turn.txt" \
N_ROLLOUTS=1 \
ROLLOUT_WORKERS=1 \
RENDER_GPUS_CSV="$CARD" \
RENDERER_REPLICAS_PER_GPU=1 \
NRE_CACHE_SIZE=1 \
ENABLE_AUTORESUME=false \
SERVICE_STARTUP_TIMEOUT_SEC=1800 \
RENDER_VIDEO=true \
KEEP_ROLLOUTS=1 \
MPC_OVERRIDES="controller.gains.long_position_weight=0.25 controller.gains.lat_position_weight=1.0 controller.gains.idx_start_penalty=3" \
EXTRA_OVERRIDES="${VIDEO_OVERRIDES[*]}" \
    "$AXE/run_curated_val.sh" >> "$RUN_DIR.wizard.log" 2>&1
log "wizard exit=$? after $((($(date +%s) - start) / 60)) min"
docker ps -aq --filter "name=$PREFIX" | xargs -r docker rm -f >/dev/null 2>&1 || true

"$ROOT/.venv/bin/python" - "$RUN_DIR" "$REF" <<'EOF' 2>&1 | tee -a "$LOG"
import json, pathlib, sys
def one(d):
    p = pathlib.Path(d) / "aggregate" / "results-summary.json"
    if not p.exists():
        return None
    r = json.load(open(p))["rollouts"][0]
    sm, m = r.get("score_metrics") or {}, r.get("metrics") or {}
    out = {"score": r.get("score"), "status": r.get("status")}
    for k in ("progress_clipped_rel", "dist_traveled_m", "collision_at_fault",
              "offroad", "left_corridor_laterally", "collision_rear",
              "lateral_dist_to_gt_trajectory", "min_distance_to_obstacle_m",
              "duration_frac_20s"):
        out[k] = sm.get(k, m.get(k))
    return out
new, ref = one(sys.argv[1]), one(sys.argv[2])
if new is None:
    print("PARITY FAIL: the baked run produced no summary"); sys.exit(1)
if ref is None:
    print("no reference summary to compare against"); print(json.dumps(new, indent=1)); sys.exit(0)
print(f"\n{'항목':<32}{'axe-v10 (구움)':>18}{'마운트 방식':>18}  판정")
bad = 0
for k in new:
    a, b = new[k], ref[k]
    if isinstance(a, float) and isinstance(b, float):
        same = abs(a - b) < 1e-9
        av, bv = f"{a:.6f}", f"{b:.6f}"
    else:
        same = a == b
        av, bv = str(a), str(b)
    if not same:
        bad += 1
    print(f"{k:<32}{av:>18}{bv:>18}  {'일치' if same else '다름'}")
print("\nPARITY PASSED — 구운 이미지가 측정된 구성과 동일하게 동작한다"
      if bad == 0 else f"\nPARITY FAIL — {bad}개 항목이 다르다. 제출하면 안 된다.")
sys.exit(0 if bad == 0 else 1)
EOF
log "=== parity done: $RUN_DIR ==="
