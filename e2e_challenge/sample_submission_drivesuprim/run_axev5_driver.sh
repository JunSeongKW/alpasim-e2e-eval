#!/usr/bin/env bash
# One axe-v5 driver replica, run exactly as submitted.
#
# axe-v5 is the image the team shipped to the organizers. Its own ENV carries
# the score-affecting knobs (vov backbone, fp16=1, inference_interval=0,
# max_batch=2), so this launcher deliberately overrides none of them -- unlike
# the local research launchers, which force fp16=0 and a 10 Hz inference gate.
set -euo pipefail

IMAGE="${IMAGE:-696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v5}"
GPU_INDEX="${GPU_INDEX:-0}"
DRIVER_PORT="${DRIVER_PORT:-6860}"
CONTAINER_PORT=6789
CONTAINER_NAME="${CONTAINER_NAME:-axev5-driver}"
# The image defaults to loading the policy on a gRPC-side thread. On this host
# that hangs inside CUDA init, which is the exact case the image's own
# LOAD_BEFORE_SERVE knob exists for. It moves the load to the main thread and
# changes nothing the policy computes.
DRIVESUPRIM_LOAD_BEFORE_SERVE="${DRIVESUPRIM_LOAD_BEFORE_SERVE:-1}"

docker image inspect "$IMAGE" >/dev/null 2>&1 || {
    echo "ERROR: image not found: $IMAGE" >&2
    exit 2
}

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi

echo "Starting axe-v5 driver (as submitted): GPU=$GPU_INDEX port=$DRIVER_PORT"

exec docker run \
    --rm \
    --init \
    --name "$CONTAINER_NAME" \
    --gpus "device=$GPU_INDEX" \
    --cap-drop ALL \
    --security-opt no-new-privileges:true \
    --read-only \
    --pids-limit 1024 \
    --memory 32g \
    --cpus 8 \
    --tmpfs /tmp:rw,nosuid,nodev,size=4g \
    --tmpfs /run:rw,nosuid,nodev,size=64m \
    -p "127.0.0.1:${DRIVER_PORT}:${CONTAINER_PORT}" \
    -e "ALPASIM_DRIVER_HOST=0.0.0.0" \
    -e "ALPASIM_DRIVER_PORT=$CONTAINER_PORT" \
    -e "DRIVESUPRIM_LOAD_BEFORE_SERVE=$DRIVESUPRIM_LOAD_BEFORE_SERVE" \
    -e "DRIVESUPRIM_SESSION_STARTUP_TIMEOUT_S=${DRIVESUPRIM_SESSION_STARTUP_TIMEOUT_S:-1800}" \
    "$IMAGE"
