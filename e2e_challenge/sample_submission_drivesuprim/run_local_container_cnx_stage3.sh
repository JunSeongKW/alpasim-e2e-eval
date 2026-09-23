#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-${CONTESTANT_IMAGE:-alpasim-e2e-drivesuprim-cnx-stage3-driver:vtest}}"

HOST_PORT="${ALPASIM_DRIVER_PORT:-6789}"
CONTAINER_PORT="${ALPASIM_DRIVER_CONTAINER_PORT:-6789}"
GPU_INDEX="${ALPASIM_GPU_INDEX:-0}"

REPLICA_INDEX="${ALPASIM_CONTESTANT_REPLICA_INDEX:-0}"
REPLICAS="${ALPASIM_CONTESTANT_REPLICAS:-1}"

MAX_BATCH_SIZE="${DRIVESUPRIM_MAX_BATCH_SIZE:-1}"
BATCH_WAIT_MS="${DRIVESUPRIM_BATCH_WAIT_MS:-5}"
INFERENCE_INTERVAL_US="${DRIVESUPRIM_INFERENCE_INTERVAL_US:-100000}"

USE_FP16="${DRIVESUPRIM_USE_FP16:-0}"

# The BEV input bridge is enabled. Closed-loop inference is the default;
# explicitly set DRIVESUPRIM_DISABLE_INFERENCE=1 only for a calibration probe.
DISABLE_INFERENCE="${DRIVESUPRIM_DISABLE_INFERENCE:-0}"

CONTAINER_NAME="${DRIVESUPRIM_CONTAINER_NAME:-drivesuprim-cnx-stage3-jh}"

if docker ps -a \
    --format '{{.Names}}' \
    | grep -qx "$CONTAINER_NAME"
then
    echo "Removing stale container: $CONTAINER_NAME"
    docker rm -f "$CONTAINER_NAME"
fi

exec docker run \
    --rm \
    --init \
    --name "$CONTAINER_NAME" \
    --gpus "device=${GPU_INDEX}" \
    --cap-drop ALL \
    --security-opt no-new-privileges:true \
    --read-only \
    --pids-limit 1024 \
    --memory 32g \
    --cpus 8 \
    --tmpfs /tmp:rw,nosuid,nodev,size=4g \
    --tmpfs /run:rw,nosuid,nodev,size=64m \
    -p "127.0.0.1:${HOST_PORT}:${CONTAINER_PORT}" \
    -e "ALPASIM_DRIVER_HOST=0.0.0.0" \
    -e "ALPASIM_DRIVER_PORT=${CONTAINER_PORT}" \
    -e "ALPASIM_CONTESTANT_REPLICA_INDEX=${REPLICA_INDEX}" \
    -e "ALPASIM_CONTESTANT_REPLICAS=${REPLICAS}" \
    -e "DRIVESUPRIM_BACKBONE_TYPE=bevformer_m" \
    -e "DRIVESUPRIM_CONFIG_PATH=/app/assets/drivesuprim/cnx_stage3_config.json" \
    -e "DRIVESUPRIM_CHECKPOINT_PATH=/app/assets/drivesuprim/drivesuprim_v2_convnextv2_stage3_epoch04.ckpt" \
    -e "DRIVESUPRIM_VOCAB_PATH=/app/assets/drivesuprim/test_4096_kmeans.npy" \
    -e "DRIVESUPRIM_REQUIRE_POLICY_FOR_VERSION=0" \
    -e "DRIVESUPRIM_SESSION_STARTUP_TIMEOUT_S=600" \
    -e "DRIVESUPRIM_MAX_BATCH_SIZE=${MAX_BATCH_SIZE}" \
    -e "DRIVESUPRIM_BATCH_WAIT_MS=${BATCH_WAIT_MS}" \
    -e "DRIVESUPRIM_INFERENCE_INTERVAL_US=${INFERENCE_INTERVAL_US}" \
    -e "DRIVESUPRIM_USE_FP16=${USE_FP16}" \
    -e "DRIVESUPRIM_NATIVE_FP16=0" \
    -e "DRIVESUPRIM_CHANNELS_LAST=0" \
    -e "DRIVESUPRIM_DISABLE_INFERENCE=${DISABLE_INFERENCE}" \
    -e "ALPASIM_DRIVER_GRPC_WORKERS=8" \
    "$IMAGE"
