#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-alpasim-e2e-drivesuprim-axe-nurec-vits-driver:epoch02-step4503-h100}"
GPU_INDEX="${GPU_INDEX:-0}"
DRIVER_PORT="${DRIVER_PORT:-6860}"
NAVSIM_2HZ_HISTORY="${NAVSIM_2HZ_HISTORY:-1}"
CONTAINER_PORT=6789
CONTAINER_NAME="${CONTAINER_NAME:-drivesuprim-axe-nurec-vits-smoke}"
# The image bakes a backbone type for the checkpoint it shipped with. A package
# built around a different front-end (vits_flat, say) has to say so, or the
# policy refuses to load with a checkpoint/backbone mismatch.
DRIVESUPRIM_BACKBONE_TYPE="${DRIVESUPRIM_BACKBONE_TYPE:-}"

[[ "$NAVSIM_2HZ_HISTORY" == 0 || "$NAVSIM_2HZ_HISTORY" == 1 ]] || {
    echo "ERROR: NAVSIM_2HZ_HISTORY must be 0 or 1" >&2
    exit 2
}

if [[ "$NAVSIM_2HZ_HISTORY" == 1 ]]; then
    history_description="t-1.0s, t-0.5s, t"
else
    history_description="latest 3 consecutive 10Hz frames"
fi

docker image inspect "$IMAGE" >/dev/null 2>&1 || {
    echo "ERROR: image not found: $IMAGE" >&2
    exit 2
}

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    docker rm -f "$CONTAINER_NAME" >/dev/null
fi

echo "Starting AXE NuRec ViT-S driver in foreground"
echo "  image: $IMAGE"
echo "  GPU/port: $GPU_INDEX/$DRIVER_PORT"
echo "  input: K377 512x256, L0/F0/R0"
echo "  history: $history_description"
echo "  route: AlpaSim 20 slots, 40-80m, normalized by 80m"

debug_args=()
if [[ -n "${BEV_DEBUG_DIR:-}" ]]; then
    mkdir -p "$BEV_DEBUG_DIR"
    BEV_DEBUG_DIR="$(realpath "$BEV_DEBUG_DIR")"
    debug_args+=(
        --mount "type=bind,src=$BEV_DEBUG_DIR,dst=/mnt/bev-debug"
        -e "DRIVESUPRIM_BEV_DEBUG_DIR=/mnt/bev-debug"
        -e "DRIVESUPRIM_BEV_DEBUG_EVERY=${BEV_DEBUG_EVERY:-20}"
        -e "DRIVESUPRIM_BEV_DEBUG_MAX=${BEV_DEBUG_MAX:-10}"
        -e "DRIVESUPRIM_BEV_DEBUG_ABLATION=${BEV_DEBUG_ABLATION:-1}"
        -e "DRIVESUPRIM_BEV_DEBUG_REPLICA=${BEV_DEBUG_REPLICA:-port${DRIVER_PORT}}"
    )
    echo "  BEV debug: $BEV_DEBUG_DIR (every=${BEV_DEBUG_EVERY:-20}, max=${BEV_DEBUG_MAX:-10})"
fi

# Swap the DriveSuprim config JSON without rebuilding the image. Used for
# ranking-weight sweeps, e.g. a variant with a softened ego_progress exponent.
config_args=()
if [[ -n "${DRIVESUPRIM_CONFIG_OVERRIDE:-}" ]]; then
    override_path="$(realpath "$DRIVESUPRIM_CONFIG_OVERRIDE")"
    [[ -f "$override_path" ]] || { echo "ERROR: config override not found: $override_path" >&2; exit 1; }
    config_args+=(
        --mount "type=bind,src=$override_path,dst=/mnt/drivesuprim-config.json,readonly"
        -e "DRIVESUPRIM_CONFIG_PATH=/mnt/drivesuprim-config.json"
    )
    echo "  config override: $override_path"
fi

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
    -e "DRIVESUPRIM_NAVSIM_2HZ_HISTORY=$NAVSIM_2HZ_HISTORY" \
    ${DRIVESUPRIM_BACKBONE_TYPE:+-e "DRIVESUPRIM_BACKBONE_TYPE=$DRIVESUPRIM_BACKBONE_TYPE"} \
    -e "DRIVESUPRIM_REQUIRE_EXACT_CHECKPOINT=1" \
    -e "DRIVESUPRIM_DISABLE_INFERENCE=0" \
    ${DRIVESUPRIM_REFINE_DEBUG:+-e "DRIVESUPRIM_REFINE_DEBUG=$DRIVESUPRIM_REFINE_DEBUG"} \
    ${DRIVESUPRIM_RANK_LOG_EVERY:+-e "DRIVESUPRIM_RANK_LOG_EVERY=$DRIVESUPRIM_RANK_LOG_EVERY"} \
    -e "ALPASIM_DRIVER_GRPC_WORKERS=8" \
    "${debug_args[@]}" \
    "${config_args[@]}" \
    "$IMAGE"
