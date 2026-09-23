#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-alpasim-e2e-drivesuprim-axe-nurec-vits-driver:epoch29-step60040-h100}"
EXPECTED_CHECKPOINT_SHA256="${EXPECTED_CHECKPOINT_SHA256:-d2a7e5762de4329cf4e5d052c19b34f042d29ad16ff560e9b8f43725120cf413}"
GPU_INDICES_CSV="${GPU_INDICES_CSV:-0,1,2,3}"
DRIVER_PORTS_CSV="${DRIVER_PORTS_CSV:-6860,6861,6862,6863}"
CONTAINER_PREFIX="${CONTAINER_PREFIX:-drivesuprim-axe-nurec-vits-epoch29}"

IFS=',' read -r -a gpu_indices <<< "$GPU_INDICES_CSV"
IFS=',' read -r -a driver_ports <<< "$DRIVER_PORTS_CSV"
[[ "${#gpu_indices[@]}" -eq "${#driver_ports[@]}" ]] || {
    echo "ERROR: GPU and port counts differ" >&2
    exit 2
}

actual_checkpoint_sha256="$(docker image inspect "$IMAGE" --format '{{index .Config.Labels "org.alpasim.checkpoint.sha256"}}' 2>/dev/null || true)"
[[ "$actual_checkpoint_sha256" == "$EXPECTED_CHECKPOINT_SHA256" ]] || {
    echo "ERROR: $IMAGE is not the expected epoch-29 checkpoint image" >&2
    echo "  expected: $EXPECTED_CHECKPOINT_SHA256" >&2
    echo "  actual:   ${actual_checkpoint_sha256:-missing image/label}" >&2
    exit 2
}

pids=()
container_names=()
cleanup() {
    trap - EXIT INT TERM
    if [[ "${#pids[@]}" -gt 0 ]]; then
        kill "${pids[@]}" 2>/dev/null || true
        wait "${pids[@]}" 2>/dev/null || true
    fi
    if [[ "${#container_names[@]}" -gt 0 ]]; then
        docker rm -f "${container_names[@]}" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

echo "Starting ${#gpu_indices[@]} AXE epoch-29 driver replicas"
for index in "${!gpu_indices[@]}"; do
    gpu="${gpu_indices[$index]//[[:space:]]/}"
    port="${driver_ports[$index]//[[:space:]]/}"
    [[ "$gpu" =~ ^[0-9]+$ && "$port" =~ ^[0-9]+$ ]] || {
        echo "ERROR: invalid GPU/port pair: $gpu/$port" >&2
        exit 2
    }
    container_name="${CONTAINER_PREFIX}-gpu${gpu}"
    container_names+=("$container_name")
    IMAGE="$IMAGE" \
    GPU_INDEX="$gpu" \
    DRIVER_PORT="$port" \
    CONTAINER_NAME="$container_name" \
        "$SCRIPT_DIR/run_axe_nurec_vits_smoke_driver_foreground.sh" &
    pids+=("$!")
done

echo "Driver replicas are supervised in this terminal; Ctrl-C stops only these replicas."
wait -n "${pids[@]}"
status=$?
echo "A driver replica exited (status=$status); stopping the remaining replicas." >&2
exit "$status"
