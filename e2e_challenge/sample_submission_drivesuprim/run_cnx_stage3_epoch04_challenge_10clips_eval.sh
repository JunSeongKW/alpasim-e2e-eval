#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export IMAGE="${IMAGE:-alpasim-e2e-drivesuprim-cnx-stage3-driver:epoch04-h100}"
export DRIVER_PORTS_CSV="${DRIVER_PORTS_CSV:-6860,6861,6862,6863}"
export RENDER_GPUS_CSV="${RENDER_GPUS_CSV:-0,1,2,3}"
export ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-4}"
export NAVSIM_2HZ_HISTORY=1
export SERVICE_STARTUP_TIMEOUT_SEC="${SERVICE_STARTUP_TIMEOUT_SEC:-900}"
export SIM_STEPS="${SIM_STEPS:-199}"
export SCENE_LIMIT=0
export CHALLENGE_COMPAT_MODE=1
export CONTROLLER_PRESET=nonlinear
export SKIP_DRIVER_DURING_FORCE_GT=true
export PARSE_UNSTRUCTURED_DEBUG_INFO=false
export USE_HOST_SOURCE_MOUNTS=false
export RENDER_VIDEO="${RENDER_VIDEO:-false}"
export GENERATE_COMBINED_VIDEO="${GENERATE_COMBINED_VIDEO:-false}"
export DATASET_DIR="${DATASET_DIR:-/home/kaist5/Dataset/alpasim/data/nre-artifacts/all-usdzs}"
export SCENE_IDS_CSV="39875322-9804-49ef-afe2-93719b034d37,e3e470e4-1d47-4ae7-a543-ca18292cc1f8,f4e31360-9d31-4064-8cfb-efd7f3e072b7,c099d336-1152-4d57-b121-f325d8acb0e4,00169207-9da7-44c4-a67b-a925e77056ff,bdc3344d-4e43-45b9-807c-f7526199e4ef,ddc3e8df-b368-40f0-bb6a-9d9f4f568cd4,00442956-c080-4b2e-b9c7-647973969c86,161aff42-2c7f-4d9c-8a24-97a606bf3df8,c9047f0b-3765-4793-b6f8-e05eaadcad3b"
export RUN_DIR="${RUN_DIR:-$ROOT/runs/e2e_challenge_drivesuprim_challenge/cnx_stage3_epoch04_10clips_$(date +%Y%m%d_%H%M%S)}"

CONTAINER_PREFIX="${CONTAINER_PREFIX:-drivesuprim-cnx-stage3-epoch04}"
expected_image_id="$(docker image inspect "$IMAGE" --format '{{.Id}}' 2>/dev/null || true)"
[[ -n "$expected_image_id" ]] || { echo "ERROR: image not found: $IMAGE" >&2; exit 2; }
IFS=',' read -r -a expected_gpus <<< "$RENDER_GPUS_CSV"
for gpu in "${expected_gpus[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    container_name="${CONTAINER_PREFIX}-gpu${gpu}"
    running="$(docker inspect "$container_name" --format '{{.State.Running}}' 2>/dev/null || true)"
    container_image_id="$(docker inspect "$container_name" --format '{{.Image}}' 2>/dev/null || true)"
    if [[ "$running" != true || "$container_image_id" != "$expected_image_id" ]]; then
        echo "ERROR: expected ConvNeXt driver is not running: $container_name" >&2
        echo "Stop the previous model drivers, then start the ConvNeXt 4-GPU driver script." >&2
        exit 2
    fi
done

exec "$SCRIPT_DIR/run_cnx_stage3_epoch04_challenge_eval.sh"
