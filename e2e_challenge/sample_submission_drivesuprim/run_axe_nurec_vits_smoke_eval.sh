#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

IMAGE="${IMAGE:-alpasim-e2e-drivesuprim-axe-nurec-vits-driver:epoch02-step4503}"
DRIVER_PORT="${DRIVER_PORT:-6860}"
RENDER_GPU="${RENDER_GPU:-1}"
DRIVER_PORTS_CSV="${DRIVER_PORTS_CSV:-$DRIVER_PORT}"
RENDER_GPUS_CSV="${RENDER_GPUS_CSV:-$RENDER_GPU}"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-1}"
DATASET_DIR="${DATASET_DIR:-/home/kaist5/Dataset/alpasim/data/nre-artifacts/all-usdzs}"
HF_DATASET_ROOT="${HF_DATASET_ROOT:-/home/kaist5/.cache/huggingface/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles-NuRec}"
RUN_DIR="${RUN_DIR:-$ROOT/runs/e2e_challenge_drivesuprim_smoke/axe_nurec_vits_$(date +%Y%m%d_%H%M%S)}"
SCENE_LIMIT="${SCENE_LIMIT:-1}"
SIM_STEPS="${SIM_STEPS:-199}"
SCENE_IDS_CSV="${SCENE_IDS_CSV:-}"
RENDER_VIDEO="${RENDER_VIDEO:-false}"
GENERATE_COMBINED_VIDEO="${GENERATE_COMBINED_VIDEO:-false}"
NAVSIM_2HZ_HISTORY="${NAVSIM_2HZ_HISTORY:-1}"
DRIVER_WAIT_SEC="${DRIVER_WAIT_SEC:-1200}"
SERVICE_STARTUP_TIMEOUT_SEC="${SERVICE_STARTUP_TIMEOUT_SEC:-900}"
CHALLENGE_COMPAT_MODE="${CHALLENGE_COMPAT_MODE:-0}"
CHALLENGE_DEBUG_VISUALIZATION="${CHALLENGE_DEBUG_VISUALIZATION:-0}"
CONTROLLER_PRESET="${CONTROLLER_PRESET:-default}"
SKIP_DRIVER_DURING_FORCE_GT="${SKIP_DRIVER_DURING_FORCE_GT:-false}"
PARSE_UNSTRUCTURED_DEBUG_INFO="${PARSE_UNSTRUCTURED_DEBUG_INFO:-true}"
USE_HOST_SOURCE_MOUNTS="${USE_HOST_SOURCE_MOUNTS:-true}"
# Renderer scene cache. Each cached NuRec scene is resident in VRAM, so drop to 1
# when sharing a GPU with other jobs.
NRE_CACHE_SIZE="${NRE_CACHE_SIZE:-2}"
# Service ports are host-bound (network_mode: host), so two stacks on one
# machine collide unless they are given different bases. 6000 is the wizard
# default; a second concurrent run needs its own window.
WIZARD_BASEPORT="${WIZARD_BASEPORT:-6000}"
OMIT_FAIL_ON_DRIVER_TERMINATION="${OMIT_FAIL_ON_DRIVER_TERMINATION:-false}"

[[ "$SIM_STEPS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: SIM_STEPS must be a positive integer" >&2; exit 2; }
[[ "$SCENE_LIMIT" =~ ^[0-9]+$ ]] || { echo "ERROR: SCENE_LIMIT must be a non-negative integer" >&2; exit 2; }
[[ "$ROLLOUT_WORKERS" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: ROLLOUT_WORKERS must be a positive integer" >&2; exit 2; }
[[ "$SERVICE_STARTUP_TIMEOUT_SEC" =~ ^[1-9][0-9]*$ ]] || { echo "ERROR: SERVICE_STARTUP_TIMEOUT_SEC must be a positive integer" >&2; exit 2; }
[[ "$NAVSIM_2HZ_HISTORY" == 0 || "$NAVSIM_2HZ_HISTORY" == 1 ]] || { echo "ERROR: NAVSIM_2HZ_HISTORY must be 0 or 1" >&2; exit 2; }
[[ "$CHALLENGE_COMPAT_MODE" == 0 || "$CHALLENGE_COMPAT_MODE" == 1 ]] || { echo "ERROR: CHALLENGE_COMPAT_MODE must be 0 or 1" >&2; exit 2; }
[[ "$CHALLENGE_DEBUG_VISUALIZATION" == 0 || "$CHALLENGE_DEBUG_VISUALIZATION" == 1 ]] || { echo "ERROR: CHALLENGE_DEBUG_VISUALIZATION must be 0 or 1" >&2; exit 2; }
for flag in "$RENDER_VIDEO" "$GENERATE_COMBINED_VIDEO" "$SKIP_DRIVER_DURING_FORCE_GT" "$PARSE_UNSTRUCTURED_DEBUG_INFO" "$USE_HOST_SOURCE_MOUNTS" "$OMIT_FAIL_ON_DRIVER_TERMINATION"; do
    [[ "$flag" == true || "$flag" == false ]] || {
        echo "ERROR: video flags must be true or false" >&2
        exit 2
    }
done

if [[ "$CHALLENGE_COMPAT_MODE" == 1 ]]; then
    # Mirror the current public PAI EC2 preset's score-affecting semantics.
    # The private scene set and 8-GPU/32-rollout production topology cannot be
    # reproduced by this local 4-GPU runner.
    CONTROLLER_PRESET="nonlinear"
    SKIP_DRIVER_DURING_FORCE_GT="true"
    PARSE_UNSTRUCTURED_DEBUG_INFO="false"
    USE_HOST_SOURCE_MOUNTS="false"

    # Diagnostic-only exception: retain the public challenge's score-affecting
    # controller/force-GT semantics, but trust this local driver and mount the
    # local evaluator so its DriveSuprim BEV panel can decode and render the
    # pickle-backed debug payload. Never enable this for an untrusted image.
    if [[ "$CHALLENGE_DEBUG_VISUALIZATION" == 1 ]]; then
        PARSE_UNSTRUCTURED_DEBUG_INFO="true"
        USE_HOST_SOURCE_MOUNTS="true"
    fi
fi

[[ "$CONTROLLER_PRESET" == default || "$CONTROLLER_PRESET" == nonlinear ]] || {
    echo "ERROR: CONTROLLER_PRESET must be default or nonlinear" >&2
    exit 2
}

if [[ "$NAVSIM_2HZ_HISTORY" == 1 ]]; then
    camera_history_profile="2hz:t-1.0s,t-0.5s,t"
else
    camera_history_profile="10hz:latest-3-consecutive"
fi

driver_addresses_hydra="["
driver_separator=""
IFS=',' read -r -a driver_ports <<< "$DRIVER_PORTS_CSV"
for port in "${driver_ports[@]}"; do
    port="${port//[[:space:]]/}"
    [[ "$port" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid driver port: $port" >&2; exit 2; }
    driver_addresses_hydra+="${driver_separator}\"localhost:${port}\""
    driver_separator=","
done
driver_addresses_hydra+="]"

renderer_gpus_hydra="["
gpu_separator=""
IFS=',' read -r -a renderer_gpus <<< "$RENDER_GPUS_CSV"
for gpu in "${renderer_gpus[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid renderer GPU: $gpu" >&2; exit 2; }
    renderer_gpus_hydra+="${gpu_separator}${gpu}"
    gpu_separator=","
done
renderer_gpus_hydra+="]"

[[ "${#driver_ports[@]}" -ge "$ROLLOUT_WORKERS" ]] || {
    echo "ERROR: need at least $ROLLOUT_WORKERS driver ports" >&2
    exit 2
}
[[ "${#renderer_gpus[@]}" -ge "$ROLLOUT_WORKERS" ]] || {
    echo "ERROR: need at least $ROLLOUT_WORKERS renderer GPUs" >&2
    exit 2
}

scene_ids_hydra="null"
scene_description="$SCENE_LIMIT scene(s) from the cache"
scene_count=0
if [[ -n "$SCENE_IDS_CSV" ]]; then
    scene_ids_hydra="["
    separator=""
    IFS=',' read -r -a requested_scene_ids <<< "$SCENE_IDS_CSV"
    for scene_id in "${requested_scene_ids[@]}"; do
        scene_id="${scene_id//[[:space:]]/}"
        [[ -n "$scene_id" ]] || continue
        [[ "$scene_id" == clipgt-* ]] || scene_id="clipgt-$scene_id"
        [[ "$scene_id" =~ ^clipgt-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ ]] || {
            echo "ERROR: invalid NuRec scene ID: $scene_id" >&2
            exit 2
        }
        scene_uuid="${scene_id#clipgt-}"
        scene_usdz="$DATASET_DIR/${scene_uuid}.usdz"
        [[ -f "$scene_usdz" && -r "$scene_usdz" ]] || {
            echo "ERROR: scene USDZ not found or unreadable: $scene_usdz" >&2
            exit 2
        }
        scene_ids_hydra+="${separator}\"${scene_id}\""
        separator=","
        scene_count=$((scene_count + 1))
    done
    [[ "$scene_count" -gt 0 ]] || { echo "ERROR: SCENE_IDS_CSV did not contain a scene ID" >&2; exit 2; }
    scene_ids_hydra+="]"
    scene_description="$scene_count explicitly selected scene(s)"
fi

for port in "${driver_ports[@]}"; do
    echo "Waiting for driver port $port (timeout=${DRIVER_WAIT_SEC}s)..."
    deadline=$((SECONDS + DRIVER_WAIT_SEC))
    until ss -ltn | grep -q ":${port}[[:space:]]"; do
        if (( SECONDS >= deadline )); then
            echo "ERROR: driver port $port is not listening after ${DRIVER_WAIT_SEC}s." >&2
            exit 1
        fi
        sleep 2
    done
    echo "Driver port $port is ready."
done

[[ -d "$DATASET_DIR" ]] || { echo "ERROR: missing dataset: $DATASET_DIR" >&2; exit 2; }
[[ -d "$HF_DATASET_ROOT" ]] || { echo "ERROR: missing NuRec artifact root: $HF_DATASET_ROOT" >&2; exit 2; }
[[ ! -e "$RUN_DIR/wizard-config.yaml" ]] || { echo "ERROR: RUN_DIR is not empty: $RUN_DIR" >&2; exit 2; }

mkdir -p "$RUN_DIR/manifests"
chmod a+rwx "$RUN_DIR"
image_id="$(docker image inspect "$IMAGE" --format '{{.Id}}')"
checkpoint_sha256="$(docker image inspect "$IMAGE" --format '{{index .Config.Labels "org.alpasim.checkpoint.sha256"}}')"
cat > "$RUN_DIR/manifests/axe_nurec_vits_provenance.txt" <<EOF
created=$(date --iso-8601=seconds)
image=$IMAGE
image_id=$image_id
checkpoint_sha256=$checkpoint_sha256
drivers=$driver_addresses_hydra
renderer_gpus=$renderer_gpus_hydra
rollout_workers=$ROLLOUT_WORKERS
service_startup_timeout_sec=$SERVICE_STARTUP_TIMEOUT_SEC
nre_cache_size=$NRE_CACHE_SIZE
virtual_resolution=512x256
virtual_k=377,377,256,128
cameras=CAM_L0,CAM_F0,CAM_R0
camera_history_profile=$camera_history_profile
route_start_offset_m=40.0
route_slots=20
route_norm_m=80.0
vocab=nurec_train_kmeans_4096x40x3.npy
dataset=$DATASET_DIR
scene_ids=$scene_ids_hydra
simulation_steps=$SIM_STEPS
render_video=$RENDER_VIDEO
generate_combined_video=$GENERATE_COMBINED_VIDEO
challenge_compat_mode=$CHALLENGE_COMPAT_MODE
challenge_debug_visualization=$CHALLENGE_DEBUG_VISUALIZATION
controller_preset=$CONTROLLER_PRESET
skip_driver_during_force_gt=$SKIP_DRIVER_DURING_FORCE_GT
parse_unstructured_debug_info=$PARSE_UNSTRUCTURED_DEBUG_INFO
use_host_source_mounts=$USE_HOST_SOURCE_MOUNTS
eval_fail_on_driver_termination=$([[ "$OMIT_FAIL_ON_DRIVER_TERMINATION" == true ]] && echo omitted_for_july_image_schema || echo preset_default_true)
EOF

export DRIVESUPRIM_EVAL_ENABLED=true
export ALPASIM_DATA_DIR="$ROOT/data"
export ALPASIM_IMAGE="${ALPASIM_IMAGE:-nvcr.io/nvidia/nre/nre-ga:26.04}"

physics_volumes="[\"${HF_DATASET_ROOT}:/mnt/nre-data\"]"
runtime_volumes="[\"${HF_DATASET_ROOT}:/mnt/nre-data\",\"${RUN_DIR}:/mnt/log_dir\",\"${RUN_DIR}:/mnt/array_job_dir\"]"
controller_volumes="[\"${RUN_DIR}/controller:/mnt/output\",\"${RUN_DIR}:/mnt/config\"]"
if [[ "$USE_HOST_SOURCE_MOUNTS" == true ]]; then
    physics_volumes="[\"${HF_DATASET_ROOT}:/mnt/nre-data\",\"${ROOT}/src:/repo/src\",\"${ROOT}/plugins:/repo/plugins\"]"
    runtime_volumes="[\"${HF_DATASET_ROOT}:/mnt/nre-data\",\"${RUN_DIR}:/mnt/log_dir\",\"${RUN_DIR}:/mnt/array_job_dir\",\"${ROOT}/src:/repo/src\",\"${ROOT}/plugins:/repo/plugins\"]"
    controller_volumes="[\"${RUN_DIR}/controller:/mnt/output\",\"${RUN_DIR}:/mnt/config\",\"${ROOT}/src:/repo/src\",\"${ROOT}/plugins:/repo/plugins\"]"
fi

overrides=(
    +e2e_challenge=dev
    topology=1gpu
    "controller=${CONTROLLER_PRESET}"
    "wizard.external_services.driver=${driver_addresses_hydra}"
    "services.renderer.gpus=${renderer_gpus_hydra}"
    services.renderer.replicas_per_container=1
    "services.renderer.volumes=[\"${HF_DATASET_ROOT}:/mnt/nre-data\",\"${ROOT}/data/nre-artifacts/ego-hoods:/mnt/ego-hoods\"]"
    "services.physics.gpus=${renderer_gpus_hydra}"
    services.physics.replicas_per_container=1
    "services.physics.volumes=${physics_volumes}"
    "services.trafficsim.gpus=${renderer_gpus_hydra}"
    "services.controller.replicas_per_container=${ROLLOUT_WORKERS}"
    "services.controller.volumes=${controller_volumes}"
    "runtime.nr_workers=${ROLLOUT_WORKERS}"
    "+runtime.endpoints.startup_timeout_s=${SERVICE_STARTUP_TIMEOUT_SEC}"
    "runtime.simulation_config.n_sim_steps=${SIM_STEPS}"
    runtime.enable_autoresume=false
    runtime.endpoints.driver.n_concurrent_rollouts=1
    runtime.endpoints.physics.n_concurrent_rollouts=1
    runtime.endpoints.controller.n_concurrent_rollouts=1
    runtime.endpoints.renderer.n_concurrent_rollouts=1
    "services.runtime.volumes=${runtime_volumes}"
    runtime.simulation_config.render_bundling=BATCH_RENDER_RGB
    runtime.simulation_config.route_start_offset_m=40.0
    "runtime.simulation_config.skip_driver_during_force_gt=${SKIP_DRIVER_DURING_FORCE_GT}"
    defines.nre_max_workers=1
    "defines.nre_cache_size=${NRE_CACHE_SIZE}"
    eval.enabled=true
    eval.allow_aggregation_with_failed_rollouts=true
    "eval.video.render_video=${RENDER_VIDEO}"
    "eval.video.generate_combined_video=${GENERATE_COMBINED_VIDEO}"
    "eval.parse_unstructured_debug_info=${PARSE_UNSTRUCTURED_DEBUG_INFO}"
    "scenes.scene_ids=${scene_ids_hydra}"
    scenes.test_suite_id=null
    "scenes.scene_cache=${DATASET_DIR}"
    "scenes.local_usdz_dir=${DATASET_DIR}"
    "scenes.limit_to_first_n=${SCENE_LIMIT}"
    "wizard.log_dir=${RUN_DIR}"
    "wizard.baseport=${WIZARD_BASEPORT}"
)

# Extra hydra overrides, space-separated. Used for controller-gain sweeps, e.g.
#   EXTRA_OVERRIDES="controller.gains.lat_position_weight=4.0 controller.gains.idx_start_penalty=4"
# Applied last so they win over anything set above.
if [[ -n "${EXTRA_OVERRIDES:-}" ]]; then
    read -r -a extra_overrides_arr <<< "$EXTRA_OVERRIDES"
    overrides+=("${extra_overrides_arr[@]}")
fi

if [[ "$CHALLENGE_COMPAT_MODE" == 1 && "$OMIT_FAIL_ON_DRIVER_TERMINATION" == true ]]; then
    # Only needed against a simulator image older than the host wizard: the
    # July alpasim-base:0.89.0 build neither defines nor consumes this field,
    # so the generated eval config would not parse. A simulator image built
    # from the current checkout defines it, and both official presets set it
    # true -- deleting it there would silently fall back to the schema default
    # (false) and diverge from the challenge.
    overrides+=("~eval.fail_on_driver_termination")
fi

echo "Starting AXE NuRec ViT-S closed-loop smoke"
echo "  drivers: $driver_addresses_hydra"
echo "  renderer/physics GPUs: $renderer_gpus_hydra"
echo "  concurrent rollout workers: $ROLLOUT_WORKERS"
echo "  service startup timeout: ${SERVICE_STARTUP_TIMEOUT_SEC}s"
echo "  clips: $scene_description"
echo "  simulation steps: $SIM_STEPS"
echo "  camera history: $camera_history_profile"
echo "  challenge compatibility: $CHALLENGE_COMPAT_MODE"
echo "  challenge debug visualization: $CHALLENGE_DEBUG_VISUALIZATION"
echo "  controller: $CONTROLLER_PRESET"
echo "  extra overrides: ${EXTRA_OVERRIDES:-<none>}"
echo "  skip driver during force-GT: $SKIP_DRIVER_DURING_FORCE_GT"
echo "  host source mounts: $USE_HOST_SOURCE_MOUNTS"
echo "  video: render=$RENDER_VIDEO combined=$GENERATE_COMBINED_VIDEO"
echo "  output: $RUN_DIR"

exec uv run --extra wizard alpasim_wizard "${overrides[@]}"
