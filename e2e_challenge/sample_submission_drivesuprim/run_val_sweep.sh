#!/usr/bin/env bash
# Large-scale evaluation over an arbitrary clip list with N parallel workers.
#
# Driver containers are named <prefix>-gpu<N>, so one prefix group holds at most
# one replica per GPU; W workers therefore need ceil(W/4) groups.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

IMAGE="${IMAGE:?IMAGE required}"
SHA="${EXPECTED_CHECKPOINT_SHA256:?sha required}"
CLIPS_FILE="${CLIPS_FILE:?CLIPS_FILE required}"
RUN_DIR="${RUN_DIR:?RUN_DIR required}"
WORKERS="${WORKERS:-16}"
GPUS="${GPUS:-0 1 2 3}"
BASE_PORT="${BASE_PORT:-6900}"
BASEPORT="${BASEPORT:-6400}"
MPC_OVERRIDES="${MPC_OVERRIDES:-}"
# Container-name prefix. Two stacks on one box need different ones.
PREFIX="${PREFIX:-val}"
export DRIVESUPRIM_BACKBONE_TYPE="${DRIVESUPRIM_BACKBONE_TYPE:-bevformer_m}"
export DRIVESUPRIM_CONFIG_OVERRIDE="${DRIVESUPRIM_CONFIG_OVERRIDE:-}"
export IMAGE EXPECTED_CHECKPOINT_SHA256="$SHA"

read -r -a gpu_arr <<< "$GPUS"
ngpu=${#gpu_arr[@]}
ngroup=$(( (WORKERS + ngpu - 1) / ngpu ))

mkdir -p "$RUN_DIR/_logs"
pids=(); prefixes=(); all_ports=(); render_gpus=()
w=0
for ((g=0; g<ngroup; g++)); do
    ports=(); gpus=()
    for ((k=0; k<ngpu && w<WORKERS; k++, w++)); do
        ports+=("$((BASE_PORT + w))"); gpus+=("${gpu_arr[k]}")
        all_ports+=("$((BASE_PORT + w))"); render_gpus+=("${gpu_arr[k]}")
    done
    [[ ${#ports[@]} -gt 0 ]] || break
    prefix="${PREFIX}-g${g}"; prefixes+=("$prefix")
    GPU_INDICES_CSV="$(IFS=,; echo "${gpus[*]}")" \
    DRIVER_PORTS_CSV="$(IFS=,; echo "${ports[*]}")" \
    CONTAINER_PREFIX="$prefix" \
    DRIVER_LOG_FILE="$RUN_DIR/_logs/driver_${prefix}.log" \
        "$SCRIPT_DIR/run_axe_nurec_planonly_ep09_4gpu_drivers_foreground.sh" \
        > "$RUN_DIR/_logs/stack_${prefix}.log" 2>&1 &
    pids+=($!)
done

echo "workers=$WORKERS groups=$ngroup ports=$(IFS=,; echo "${all_ports[*]}")"
for _ in $(seq 1 240); do
    ready=$(cat "$RUN_DIR"/_logs/driver_*.log 2>/dev/null | grep -c 'policy load complete') || ready=0
    [[ "$ready" -ge "$WORKERS" ]] && break
    sleep 5
done
echo "drivers ready: $ready/$WORKERS"

RUN_DIR="$RUN_DIR" SCENE_IDS_CSV="$(paste -sd, "$CLIPS_FILE")" \
DRIVER_PORTS_CSV="$(IFS=,; echo "${all_ports[*]}")" \
RENDER_GPUS_CSV="$(IFS=,; echo "${render_gpus[*]}")" \
ROLLOUT_WORKERS="$WORKERS" NRE_CACHE_SIZE=1 \
RENDER_VIDEO="${RENDER_VIDEO:-false}" WIZARD_BASEPORT="$BASEPORT" \
SERVICE_STARTUP_TIMEOUT_SEC="${SERVICE_STARTUP_TIMEOUT_SEC:-3600}" \
EXTRA_OVERRIDES="$MPC_OVERRIDES" \
    "$SCRIPT_DIR/run_axe_nurec_planonly_ep09_debug_video_eval.sh" \
    > "$RUN_DIR/_logs/eval.log" 2>&1 || echo "eval exited $?"

for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done
for pre in "${prefixes[@]}"; do docker rm -f $(docker ps -aq --filter "name=${pre}") 2>/dev/null || true; done
echo "val sweep done: $RUN_DIR"
