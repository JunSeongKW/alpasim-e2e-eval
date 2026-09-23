#!/usr/bin/env bash
# Evaluate a PAI-reconstructed NuRec scene with the stage3 ep24 driver.
#
# Differs from the NuRec-scene runners in two ways, both forced by the fact
# that reconstruction produces no HD map:
#   * scenes come from a local usdz dir, not the NuRec scene cache
#   * routes are generated from the recorded ego path, not from map lanes
# `offroad` is therefore not computed and the aggregation defaults it to 0.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$ROOT"

USDZ_DIR="${USDZ_DIR:?USDZ_DIR required}"
RUN_DIR="${RUN_DIR:?RUN_DIR required}"
DRIVER_PORT="${DRIVER_PORT:-6970}"
RENDER_GPU="${RENDER_GPU:-4}"
WIZARD_BASEPORT="${WIZARD_BASEPORT:-6500}"
SIM_STEPS="${SIM_STEPS:-199}"
RENDER_VIDEO="${RENDER_VIDEO:-true}"
MPC_OVERRIDES="${MPC_OVERRIDES:-}"

SRC_MOUNT="${SRC_MOUNT:-1}"   # 0 = use the image's own /repo/src
HF_DATASET_ROOT="${HF_DATASET_ROOT:-/home/kaist5/.cache/huggingface/hub/datasets--nvidia--PhysicalAI-Autonomous-Vehicles-NuRec}"

[[ ! -e "$RUN_DIR/wizard-config.yaml" ]] || { echo "ERROR: RUN_DIR not empty: $RUN_DIR" >&2; exit 2; }
mkdir -p "$RUN_DIR"; chmod a+rwx "$RUN_DIR"

export DRIVESUPRIM_EVAL_ENABLED=true
export ALPASIM_DATA_DIR="$ROOT/data"
export ALPASIM_IMAGE="${ALPASIM_IMAGE:-nvcr.io/nvidia/nre/nre-ga:26.04}"

overrides=(
    +e2e_challenge=dev
    topology=1gpu
    controller=nonlinear
    "wizard.external_services.driver=[\"localhost:${DRIVER_PORT}\"]"
    "services.renderer.gpus=[${RENDER_GPU}]"
    services.renderer.replicas_per_container=1
    "services.renderer.volumes=[\"${USDZ_DIR}:/mnt/nre-data\",\"${ROOT}/data/nre-artifacts/ego-hoods:/mnt/ego-hoods\",\"${ROOT}/src:/repo/src\",\"${ROOT}/plugins:/repo/plugins\"]"
    "services.physics.gpus=[${RENDER_GPU}]"
    services.physics.replicas_per_container=1
    "services.physics.volumes=[\"${USDZ_DIR}:/mnt/nre-data\",\"${ROOT}/src:/repo/src\",\"${ROOT}/plugins:/repo/plugins\"]"
    "services.trafficsim.gpus=[${RENDER_GPU}]"
    services.controller.replicas_per_container=1
    "services.controller.volumes=[\"${RUN_DIR}/controller:/mnt/output\",\"${RUN_DIR}:/mnt/config\",\"${ROOT}/src:/repo/src\",\"${ROOT}/plugins:/repo/plugins\"]"
    "services.runtime.volumes=[\"${USDZ_DIR}:/mnt/nre-data\",\"${RUN_DIR}:/mnt/log_dir\",\"${RUN_DIR}:/mnt/array_job_dir\",\"${ROOT}/src:/repo/src\",\"${ROOT}/plugins:/repo/plugins\"]"
    runtime.nr_workers=1
    "+runtime.endpoints.startup_timeout_s=${SERVICE_STARTUP_TIMEOUT_SEC:-3600}"
    "runtime.simulation_config.n_sim_steps=${SIM_STEPS}"
    # Repeat the same scene to separate run-to-run noise from real differences.
    "runtime.simulation_config.n_rollouts=${N_ROLLOUTS:-1}"
    runtime.enable_autoresume=false
    runtime.endpoints.driver.n_concurrent_rollouts=1
    runtime.endpoints.physics.n_concurrent_rollouts=1
    runtime.endpoints.controller.n_concurrent_rollouts=1
    runtime.endpoints.renderer.n_concurrent_rollouts=1
    runtime.simulation_config.render_bundling=BATCH_RENDER_RGB
    runtime.simulation_config.route_start_offset_m=40.0
    # A reconstructed scene has no HD map, so routes default to the recorded ego
    # path. Set ROUTE_GENERATOR=MAP when a map has been grafted in.
    "runtime.simulation_config.route_generator_type=${ROUTE_GENERATOR:-RECORDED}"
    runtime.simulation_config.skip_driver_during_force_gt=true
    defines.nre_max_workers=1
    defines.nre_cache_size=1
    eval.enabled=true
    eval.allow_aggregation_with_failed_rollouts=true
    "eval.video.render_video=${RENDER_VIDEO}"
    eval.video.generate_combined_video=false
    eval.parse_unstructured_debug_info=true
    scenes.scene_ids=null
    "scenes.test_suite_id=local"
    "scenes.local_usdz_dir=${USDZ_DIR}"
    "scenes.scene_cache=${USDZ_DIR}"
    scenes.limit_to_first_n=0
    "wizard.log_dir=${RUN_DIR}"
    "wizard.baseport=${WIZARD_BASEPORT}"
)
if [[ "$SRC_MOUNT" == 0 ]]; then
    # Drop the host source mounts so the containers run the image's own code.
    for i in "${!overrides[@]}"; do
        overrides[$i]="${overrides[$i]//,\\"${ROOT}\/src:\/repo\/src\\",\\"${ROOT}\/plugins:\/repo\/plugins\\"/}"
    done
    overrides+=("eval.parse_unstructured_debug_info=false")
fi

if [[ -n "$MPC_OVERRIDES" ]]; then
    read -r -a extra <<< "$MPC_OVERRIDES"; overrides+=("${extra[@]}")
fi

echo "PAI scene 평가"
echo "  usdz: $USDZ_DIR"
echo "  driver: localhost:$DRIVER_PORT   renderer GPU: $RENDER_GPU"
echo "  route generator: RECORDED (맵 없음 → offroad 미계산)"
echo "  출력: $RUN_DIR"

exec uv run --extra wizard alpasim_wizard "${overrides[@]}"
