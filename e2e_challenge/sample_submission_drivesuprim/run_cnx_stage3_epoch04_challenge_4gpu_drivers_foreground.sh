#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
IMAGE="${IMAGE:-alpasim-e2e-drivesuprim-cnx-stage3-driver:epoch04-h100}"
EXPECTED_CHECKPOINT_SHA256="${EXPECTED_CHECKPOINT_SHA256:-9a8250144083af1b37e906a25584288f78bc14b43a2e5fb5ffd4fd03457ce662}"
GPU_INDICES_CSV="${GPU_INDICES_CSV:-0,1,2,3}"
DRIVER_PORTS_CSV="${DRIVER_PORTS_CSV:-6860,6861,6862,6863}"
CONTAINER_PREFIX="${CONTAINER_PREFIX:-drivesuprim-cnx-stage3-epoch04}"
DRIVER_LOG="${DRIVER_LOG:-$ROOT/runs/driver_logs/cnx_stage3_epoch04_$(date +%Y%m%d_%H%M%S).log}"

mkdir -p "$(dirname "$DRIVER_LOG")"
exec > >(tee -a "$DRIVER_LOG") 2>&1

echo "Driver log: $DRIVER_LOG"
actual_sha="$(docker image inspect "$IMAGE" --format '{{index .Config.Labels "org.alpasim.checkpoint.sha256"}}' 2>/dev/null || true)"
[[ "$actual_sha" == "$EXPECTED_CHECKPOINT_SHA256" ]] || {
    echo "ERROR: unexpected image/checkpoint label" >&2
    echo "  expected: $EXPECTED_CHECKPOINT_SHA256" >&2
    echo "  actual:   ${actual_sha:-missing}" >&2
    exit 2
}

IFS=',' read -r -a gpu_indices <<< "$GPU_INDICES_CSV"
IFS=',' read -r -a driver_ports <<< "$DRIVER_PORTS_CSV"
[[ "${#gpu_indices[@]}" -eq "${#driver_ports[@]}" ]] || { echo "ERROR: GPU and port counts differ" >&2; exit 2; }

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

echo "Starting ${#gpu_indices[@]} ConvNeXt stage3 driver replicas"
for index in "${!gpu_indices[@]}"; do
    gpu="${gpu_indices[$index]//[[:space:]]/}"
    port="${driver_ports[$index]//[[:space:]]/}"
    [[ "$gpu" =~ ^[0-9]+$ && "$port" =~ ^[0-9]+$ ]] || { echo "ERROR: invalid GPU/port: $gpu/$port" >&2; exit 2; }
    container_name="${CONTAINER_PREFIX}-gpu${gpu}"
    container_names+=("$container_name")
    IMAGE="$IMAGE" GPU_INDEX="$gpu" DRIVER_PORT="$port" CONTAINER_NAME="$container_name" \
        "$SCRIPT_DIR/run_cnx_stage3_epoch04_challenge_driver_foreground.sh" &
    pids+=("$!")
done

echo "Driver replicas are supervised in this terminal; Ctrl-C stops only these replicas."
set +e
wait -n "${pids[@]}"
status=$?
set -e
echo "A driver replica exited (status=$status); stopping the remaining replicas." >&2
exit "$status"
