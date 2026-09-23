#!/usr/bin/env bash
# Supervise one axe-v5 driver replica per (GPU, port) pair.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v5}"
GPU_INDICES_CSV="${GPU_INDICES_CSV:-0,1,2,3}"
DRIVER_PORTS_CSV="${DRIVER_PORTS_CSV:-6860,6861,6862,6863}"
CONTAINER_PREFIX="${CONTAINER_PREFIX:-axev5}"

IFS=',' read -r -a gpu_indices <<< "$GPU_INDICES_CSV"
IFS=',' read -r -a driver_ports <<< "$DRIVER_PORTS_CSV"
[[ "${#gpu_indices[@]}" -eq "${#driver_ports[@]}" ]] || {
    echo "ERROR: GPU and port counts differ" >&2
    exit 2
}

pids=(); container_names=()
cleanup() {
    trap - EXIT INT TERM
    [[ "${#pids[@]}" -gt 0 ]] && { kill "${pids[@]}" 2>/dev/null || true; wait "${pids[@]}" 2>/dev/null || true; }
    [[ "${#container_names[@]}" -gt 0 ]] && { docker rm -f "${container_names[@]}" >/dev/null 2>&1 || true; }
}
trap cleanup EXIT INT TERM

echo "Starting ${#gpu_indices[@]} axe-v5 driver replicas"
for index in "${!gpu_indices[@]}"; do
    gpu="${gpu_indices[$index]//[[:space:]]/}"
    port="${driver_ports[$index]//[[:space:]]/}"
    [[ "$gpu" =~ ^[0-9]+$ && "$port" =~ ^[0-9]+$ ]] || {
        echo "ERROR: invalid GPU/port pair: $gpu/$port" >&2
        exit 2
    }
    container_name="${CONTAINER_PREFIX}-gpu${gpu}"
    container_names+=("$container_name")
    IMAGE="$IMAGE" GPU_INDEX="$gpu" DRIVER_PORT="$port" CONTAINER_NAME="$container_name" \
        "$SCRIPT_DIR/run_axev5_driver.sh" &
    pids+=("$!")
done

wait -n "${pids[@]}"
status=$?
echo "A driver replica exited (status=$status); stopping the rest." >&2
exit "$status"
