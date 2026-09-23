#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-alpasim-e2e-drivesuprim-cnx-stage3-driver:epoch04-h100}"
GPU_INDEX="${GPU_INDEX:-0}"
DRIVER_PORT="${DRIVER_PORT:-6860}"
NAVSIM_2HZ_HISTORY="${NAVSIM_2HZ_HISTORY:-1}"
PINHOLE_PROFILE="${PINHOLE_PROFILE:-NAVSIM_512X256}"
CONTAINER_PORT=6789
CONTAINER_NAME="${CONTAINER_NAME:-drivesuprim-cnx-stage3-epoch04}"

[[ "$NAVSIM_2HZ_HISTORY" == 1 ]] || { echo "ERROR: ConvNeXt training contract requires NAVSIM_2HZ_HISTORY=1" >&2; exit 2; }
[[ "$PINHOLE_PROFILE" == NAVSIM_512X256 ]] || { echo "ERROR: ConvNeXt training contract requires PINHOLE_PROFILE=NAVSIM_512X256" >&2; exit 2; }
docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "ERROR: image not found: $IMAGE" >&2; exit 2; }

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi

echo "Starting DriveSuprim ConvNeXt-v2 stage3 epoch04 driver"
echo "  image: $IMAGE"
echo "  GPU/port: $GPU_INDEX/$DRIVER_PORT"
echo "  input: NAVSIM K=(412,366.2222,256,132.7407), 512x256, L0/F0/R0"
echo "  history: t-1.0s, t-0.5s, t"
echo "  intent: driving command derived from AlpaSim route"

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
    -e "DRIVESUPRIM_REQUIRE_POLICY_FOR_VERSION=1" \
    -e "DRIVESUPRIM_SESSION_STARTUP_TIMEOUT_S=1200" \
    -e "DRIVESUPRIM_MAX_BATCH_SIZE=1" \
    -e "DRIVESUPRIM_INFERENCE_INTERVAL_US=100000" \
    -e "DRIVESUPRIM_USE_FP16=0" \
    -e "DRIVESUPRIM_NATIVE_FP16=0" \
    -e "DRIVESUPRIM_CHANNELS_LAST=0" \
    -e "DRIVESUPRIM_PINHOLE_PROFILE=$PINHOLE_PROFILE" \
    -e "DRIVESUPRIM_NAVSIM_2HZ_HISTORY=$NAVSIM_2HZ_HISTORY" \
    -e "DRIVESUPRIM_REQUIRE_EXACT_CHECKPOINT=1" \
    -e "DRIVESUPRIM_FORCE_PYTORCH_MSDA=1" \
    -e "DRIVESUPRIM_DISABLE_INFERENCE=0" \
    -e "ALPASIM_DRIVER_GRPC_WORKERS=8" \
    "$IMAGE"
