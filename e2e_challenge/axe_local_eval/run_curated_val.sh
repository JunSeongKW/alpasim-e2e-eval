#!/usr/bin/env bash
# Evaluate the AXE driver on the official nurec_curated_val split.
#
# The scoring contract comes entirely from `+e2e_challenge=dev` (six 10 Hz
# cameras, 200 sim steps, 1.7 s force-GT, nonlinear MPC) and
# `+nurec_scenes=curated_val`, exactly as the organizer-published reference runs
# were produced. Everything overridden below is placement only -- which GPUs the
# renderer and physics land on and how many rollouts run at once -- because GPUs
# 0-3 belong to other jobs on this box.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

DRIVER_ADDRESSES="${DRIVER_ADDRESSES:-}"
[[ -n "$DRIVER_ADDRESSES" ]] || {
    echo "ERROR: set DRIVER_ADDRESSES to the hydra list printed by start_drivers.sh" >&2
    exit 2
}

# Which competition preset pins the scoring contract.
#   dev - what the organizer-published reference runs used: harmonizer on, the
#         driver is called during the force-GT warmup. Comparable to the
#         reference bundle, so the local Drive-IRT leaderboard is meaningful.
#   ec2 - what the leaderboard itself runs: harmonizer off,
#         skip_driver_during_force_gt, batched RGB render. Closer to the real
#         evaluation, but NOT comparable to the reference bundle.
# Both are used with deploy=local/topology=1gpu; only placement differs from the
# preset, never a scored quantity.
PRESET="${PRESET:-dev}"
[[ "$PRESET" == dev || "$PRESET" == ec2 ]] || {
    echo "ERROR: PRESET must be dev or ec2" >&2
    exit 2
}

RUN_NAME="${RUN_NAME:-axe-ep24-curatedval-${PRESET}}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/${RUN_NAME}}"
# Rollouts per scene. The published references use 3.
N_ROLLOUTS="${N_ROLLOUTS:-3}"
# Concurrent rollouts. Capacity per service is
# len(gpus) x replicas_per_container x n_concurrent_rollouts, so the renderer
# replica count below has to match this.
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-8}"
RENDER_GPUS_CSV="${RENDER_GPUS_CSV:-4,5,6,7}"
RENDERER_REPLICAS_PER_GPU="${RENDERER_REPLICAS_PER_GPU:-2}"
PHYSICS_REPLICAS_PER_GPU="${PHYSICS_REPLICAS_PER_GPU:-2}"
# Each cached NuRec scene stays resident in VRAM.
NRE_CACHE_SIZE="${NRE_CACHE_SIZE:-2}"
SERVICE_STARTUP_TIMEOUT_SEC="${SERVICE_STARTUP_TIMEOUT_SEC:-900}"
SCENE_LIMIT="${SCENE_LIMIT:-0}"
# Restrict the run to an explicit clip list (one clipgt id per line). Used by the
# controller-gain search, which scores a stratified subset instead of all 441.
SCENE_IDS_FILE="${SCENE_IDS_FILE:-}"
RENDER_VIDEO="${RENDER_VIDEO:-false}"
# A full 441 x 3 run takes many hours; let it pick up where it left off.
ENABLE_AUTORESUME="${ENABLE_AUTORESUME:-true}"
# Controller gains are a submitted artifact in this competition, not a fixed
# part of the contract. These are the tuned values the AXE submission uses; all
# three sit inside the published allowed ranges. Set MPC_OVERRIDES="" to fall
# back to the stock 2.0/1.0/10 defaults.
MPC_OVERRIDES="${MPC_OVERRIDES-controller.gains.long_position_weight=0.5 controller.gains.lat_position_weight=6.0 controller.gains.idx_start_penalty=3}"

IFS=',' read -r -a render_gpus <<< "$RENDER_GPUS_CSV"
gpus_hydra="[$(IFS=,; echo "${render_gpus[*]}")]"

# Renderer replicas are expressed by repeating the GPU rather than by raising
# replicas_per_container. Replicas sharing one container also share
# /home/.cache/nre, and they each fetch the Harmonizer checkpoint at startup:
# concurrent downloads to the same path race, and a replica that reads the file
# mid-rename dies with "The provided filename ... does not exist". One container
# per replica gives each its own cache directory.
renderer_gpu_list=()
for gpu in "${render_gpus[@]}"; do
    for ((r = 0; r < RENDERER_REPLICAS_PER_GPU; r++)); do
        renderer_gpu_list+=("${gpu//[[:space:]]/}")
    done
done
renderer_gpus_hydra="[$(IFS=,; echo "${renderer_gpu_list[*]}")]"

renderer_capacity=${#renderer_gpu_list[@]}
[[ "$renderer_capacity" -ge "$ROLLOUT_WORKERS" ]] || {
    echo "ERROR: renderer capacity ${renderer_capacity} < ROLLOUT_WORKERS ${ROLLOUT_WORKERS}" >&2
    exit 2
}

# Physics carries concurrency inside one replica per GPU instead of fanning out.
physics_concurrent=$(( (ROLLOUT_WORKERS + ${#render_gpus[@]} - 1) / ${#render_gpus[@]} ))

mkdir -p "$RUN_DIR"
# Which code produced this run. HANDOFF.md and the experiment ledger refer to
# runs by directory; this ties the directory back to a commit.
{ git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo "no-git"; git -C "$ROOT" status --short 2>/dev/null | head -50; } > "$RUN_DIR/git-commit.txt"
export ALPASIM_DATA_DIR="$ROOT/data"
export ALPASIM_IMAGE="${ALPASIM_IMAGE:-nvcr.io/nvidia/nre/nre-ga:26.04}"

overrides=(
    "+e2e_challenge=${PRESET}"
    +nurec_scenes=curated_val
    topology=1gpu
    "wizard.log_dir=${RUN_DIR}"
    "wizard.external_services.driver=${DRIVER_ADDRESSES}"
    "runtime.simulation_config.n_rollouts=${N_ROLLOUTS}"
    "runtime.nr_workers=${ROLLOUT_WORKERS}"
    "services.renderer.gpus=${renderer_gpus_hydra}"
    services.renderer.replicas_per_container=1
    "services.physics.gpus=${gpus_hydra}"
    services.physics.replicas_per_container=1
    "services.trafficsim.gpus=${gpus_hydra}"
    "services.controller.replicas_per_container=${ROLLOUT_WORKERS}"
    runtime.endpoints.renderer.n_concurrent_rollouts=1
    # Official EC2 runs two rollouts per contestant replica (topology
    # 8gpu_32rollouts). Keep 1 for scoring runs, raise it to reproduce the
    # concurrency the throughput budget is actually measured under.
    "runtime.endpoints.driver.n_concurrent_rollouts=${DRIVER_CONCURRENT_ROLLOUTS:-1}"
    "runtime.endpoints.physics.n_concurrent_rollouts=${physics_concurrent}"
    runtime.endpoints.controller.n_concurrent_rollouts=1
    "+runtime.endpoints.startup_timeout_s=${SERVICE_STARTUP_TIMEOUT_SEC}"
    "runtime.enable_autoresume=${ENABLE_AUTORESUME}"
    "defines.nre_cache_size=${NRE_CACHE_SIZE}"
    eval.enabled=true
    "eval.video.render_video=${RENDER_VIDEO}"
)

if [[ "$PRESET" == ec2 ]]; then
    # ec2.yaml is written for the organizers' EC2 hosts: it selects the
    # competition_ec2 deploy (which only emits a compose file for someone else
    # to run) and the 8-GPU topology. Put the deployment back on this box while
    # keeping every scored setting the preset defines.
    overrides+=(
        deploy=local
        wizard.validate_mount_points=false
    )
    # The preset leaves the contestant image as ??? for the EC2 runner to fill.
    export CONTESTANT_IMAGE="${CONTESTANT_IMAGE:-alpasim-e2e-drivesuprim-stage3:ep24-sweep}"
fi

if [[ "$SCENE_LIMIT" -gt 0 ]]; then
    overrides+=("scenes.limit_to_first_n=${SCENE_LIMIT}")
fi

if [[ -n "$SCENE_IDS_FILE" ]]; then
    [[ -f "$SCENE_IDS_FILE" ]] || {
        echo "ERROR: SCENE_IDS_FILE not found: $SCENE_IDS_FILE" >&2
        exit 2
    }
    scene_ids_hydra="[$(paste -sd, "$SCENE_IDS_FILE")]"
    # The wizard refuses both at once, and the preset sets the suite; an explicit
    # clip list is a strict subset of it, so the suite is what has to give way.
    overrides+=("scenes.test_suite_id=null" "scenes.scene_ids=${scene_ids_hydra}")
fi

if [[ -n "$MPC_OVERRIDES" ]]; then
    # shellcheck disable=SC2206  # deliberate word splitting into hydra overrides
    overrides+=($MPC_OVERRIDES)
fi

# Anything else the caller needs to reach in the hydra config, space separated.
# Video panel settings are the reason this exists: a visualization run wants
# eval.video.* keys that a scoring run never touches, and routing them through
# MPC_OVERRIDES would put them somewhere nobody would look for them.
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
    # shellcheck disable=SC2206  # deliberate word splitting into hydra overrides
    overrides+=($EXTRA_OVERRIDES)
fi

# ---------------------------------------------------------------------------
# Startup speed. Nothing below is a scored quantity: it decides how fast the
# simulator stack comes up, not what it computes. FAST_STARTUP=0 restores the
# stock behaviour.
#
# Measured on this box on 2026-09-23 with 16 renderers + 16 controllers: every
# fresh container compiles Python bytecode on import and writes the .pyc files
# into its own overlay layer. Eight minutes after `compose up`, `docker diff`
# listed 745 new .pyc in one renderer and 236 in the controller, all 37 python
# processes sat in uninterruptible disk wait, vda was pinned at 100%, and not
# one service port was bound. Each renderer additionally downloads ~2.8 GB of
# harmonizer / huggingface weights into $HOME/.cache. Another researcher's
# stack on the same machine went from "not ready after 50 min" to ready in
# about 7 min with the fixes below (their .changes/2026-09-22 REPORT.md).
#
#   1. PYTHONDONTWRITEBYTECODE=1 in renderer, physics, controller and runtime.
#   2. One host directory as every renderer's $HOME/.cache (HOME=/home in the
#      nre-ga image; a mount anywhere else is never read), seeded once by
#      warm_renderer_cache.sh. It has to be seeded BEFORE sixteen replicas
#      start, or they race on the download rename (see the comment on
#      renderer replicas above). The inductor / triton compile caches go in
#      the same directory so kernels compile once, not per container.
#   3. UV_OFFLINE=1: `uv run` otherwise revalidates a direct-URL dependency
#      against github.com at every launch; an outage there stopped two runs
#      that needed nothing from the network.
FAST_STARTUP="${FAST_STARTUP:-1}"
RENDER_CACHE="${RENDER_CACHE:-$ROOT/.cache/renderer-shared}"
if [[ "$FAST_STARTUP" == 1 ]]; then
    export UV_OFFLINE="${UV_OFFLINE:-1}"
    renderer_env=("OMP_NUM_THREADS=1" "PYTHONDONTWRITEBYTECODE=1" "TORCHINDUCTOR_COMPILE_THREADS=2")
    if [[ "${EXTRA_OVERRIDES:-}" == *services.renderer.volumes=* ]]; then
        echo "WARN: EXTRA_OVERRIDES sets services.renderer.volumes; renderer cache not mounted" >&2
    else
        # RENDER_CACHE_WARM=0 skips the seeding check (only when you know the
        # cache is full and want to save the du).
        if [[ "${RENDER_CACHE_WARM:-1}" == 1 ]]; then
            RENDER_CACHE="$RENDER_CACHE" WARM_GPU="${render_gpus[0]}" \
                "$ROOT/e2e_challenge/axe_local_eval/warm_renderer_cache.sh" || {
                echo "ERROR: renderer cache is not seeded and warm-up failed" >&2
                exit 2
            }
        fi
        overrides+=("services.renderer.volumes=[\"\${scenes.scene_cache}:/mnt/nre-data\",\"\${defines.sensordata}/ego-hoods:/mnt/ego-hoods\",\"${RENDER_CACHE}:/home/.cache\"]")
        renderer_env+=("TORCHINDUCTOR_CACHE_DIR=/home/.cache/torchinductor" "TRITON_CACHE_DIR=/home/.cache/triton")
    fi
    renderer_env_hydra="[$(printf '"%s",' "${renderer_env[@]}" | sed 's/,$//')]"
    overrides+=(
        "services.renderer.environments=${renderer_env_hydra}"
        "services.physics.environments=[\"WARP_CACHE_PATH=/mnt/warp\",\"PYTHONDONTWRITEBYTECODE=1\"]"
        "services.controller.environments=[\"PYTHONDONTWRITEBYTECODE=1\"]"
        "services.runtime.environments=[\"PYTHONDONTWRITEBYTECODE=1\"]"
    )
fi

# A 441x3 run writes ~516 GB of rollout.asl and only aggregate/results-summary.json
# is ever read back -- that is all the local Drive-IRT leaderboard needs. Leaving
# them behind filled a 9.7 TB disk and took down a running evaluation, so drop
# them once the wizard has aggregated. Set KEEP_ROLLOUTS=1 to retain them when a
# run is meant for video rendering or per-step debugging.
KEEP_ROLLOUTS="${KEEP_ROLLOUTS:-0}"
if [[ "$KEEP_ROLLOUTS" != 1 ]]; then
    trap '
        if [[ -f "$RUN_DIR/aggregate/results-summary.json" ]]; then
            echo "cleanup: removing rollouts (KEEP_ROLLOUTS=1 to retain)"
            rm -rf "$RUN_DIR/rollouts" "$RUN_DIR/txt-logs" "$RUN_DIR/controller"
        else
            echo "cleanup: no results-summary.json, keeping rollouts for diagnosis"
        fi
    ' EXIT
fi

echo "preset        : ${PRESET}"
echo "run dir       : $RUN_DIR"
echo "drivers       : $DRIVER_ADDRESSES"
echo "renderer       : ${renderer_capacity} replicas on ${renderer_gpus_hydra}"
echo "rollouts      : ${N_ROLLOUTS} per scene, ${ROLLOUT_WORKERS} concurrent"
echo "scene limit   : ${SCENE_LIMIT:-all}"
echo "fast startup  : ${FAST_STARTUP} (renderer cache: ${RENDER_CACHE}, UV_OFFLINE=${UV_OFFLINE:-0})"

exec uv run alpasim_wizard "${overrides[@]}"
