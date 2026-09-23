#!/usr/bin/env bash
# Large-scale axe-v5 evaluation over a clip list with N parallel workers.
# Mirrors run_val_sweep.sh, but drives the submitted image with its own ENV.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="${IMAGE:-696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v5}"
CLIPS_FILE="${CLIPS_FILE:?CLIPS_FILE required}"
RUN_DIR="${RUN_DIR:?RUN_DIR required}"
WORKERS="${WORKERS:-16}"
GPUS="${GPUS:-0 1 2 3}"
BASE_PORT="${BASE_PORT:-6900}"
BASEPORT="${BASEPORT:-6400}"
# Empty means the stock challenge gains, i.e. the controller axe-v5 was scored
# with. Set it to compare the same image under the locally tuned gains.
MPC_OVERRIDES="${MPC_OVERRIDES:-}"
PREFIX="${PREFIX:-axev5}"
export IMAGE

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
        "$SCRIPT_DIR/run_axev5_drivers_group.sh" \
        > "$RUN_DIR/_logs/driver_${prefix}.log" 2>&1 &
    pids+=($!)
done

echo "workers=$WORKERS groups=$ngroup ports=$(IFS=,; echo "${all_ports[*]}")"
# axe-v5 loads the policy before it binds, and its "policy load complete" line
# never reaches stdout, so readiness is measured by the listening socket.
for _ in $(seq 1 360); do
    ready=0
    for p in "${all_ports[@]}"; do
        ss -ltn | grep -q ":${p}[[:space:]]" && ready=$((ready + 1))
    done
    [[ "$ready" -ge "$WORKERS" ]] && break
    sleep 5
done
echo "drivers ready: $ready/$WORKERS"

RUN_DIR="$RUN_DIR" SCENE_IDS_CSV="$(paste -sd, "$CLIPS_FILE")" \
DRIVER_PORTS_CSV="$(IFS=,; echo "${all_ports[*]}")" \
RENDER_GPUS_CSV="$(IFS=,; echo "${render_gpus[*]}")" \
ROLLOUT_WORKERS="$WORKERS" NRE_CACHE_SIZE=1 \
RENDER_VIDEO="${RENDER_VIDEO:-false}" WIZARD_BASEPORT="$BASEPORT" \
CHALLENGE_DEBUG_VISUALIZATION="${CHALLENGE_DEBUG_VISUALIZATION:-0}" \
SERVICE_STARTUP_TIMEOUT_SEC="${SERVICE_STARTUP_TIMEOUT_SEC:-3600}" \
EXTRA_OVERRIDES="$MPC_OVERRIDES" \
    "$SCRIPT_DIR/run_axev5_eval.sh" \
    > "$RUN_DIR/_logs/eval.log" 2>&1 || echo "eval exited $?"

for p in "${pids[@]}"; do kill "$p" 2>/dev/null || true; done
for pre in "${prefixes[@]}"; do docker rm -f $(docker ps -aq --filter "name=${pre}") 2>/dev/null || true; done
echo "axev5 sweep done: $RUN_DIR"
