#!/usr/bin/env bash
# start_drivers.sh with the route reranker mounted over the submission image.
#
# The image bakes navsim and the challenge driver in at /app and runs
# read-only, so the reranker is bind-mounted file by file: the two navsim
# modules from the 2026-09-23 bundle, the image's own model/config with the
# reranker patch, and the challenge package with the DRIVESUPRIM_ROUTE_RERANK
# switch. The mounted files are the image's copies plus the feature, and the
# feature is off unless the switch is set, so the "before" arm mounts exactly
# the same files as the "rerank" arm -- one variable between the two.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="${IMAGE:-alpasim-e2e-drivesuprim-stage3:merged-route-ep30}"
EXPECTED_CHECKPOINT_SHA256="${EXPECTED_CHECKPOINT_SHA256-364c801e397e9d6e36ea1f9027dc4ca804ad2e429bb588cf4e84185ca851de3d}"
GPU_INDICES_CSV="${GPU_INDICES_CSV:-0,1,2,3}"
REPLICAS_PER_GPU="${REPLICAS_PER_GPU:-1}"
BASE_PORT="${BASE_PORT:-6940}"
CONTAINER_PREFIX="${CONTAINER_PREFIX:-axe-rerank}"
CONTAINER_PORT="${CONTAINER_PORT:-6789}"
TMPFS_SIZE="${TMPFS_SIZE:-2g}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-1200}"
DRIVER_PKG="${DRIVER_PKG:-$HERE/drivesuprim_challenge}"
MODEL_PY="${MODEL_PY:-$HERE/drivesuprim_model.py}"
CONFIG_PY="${CONFIG_PY:-$HERE/drivesuprim_config.py}"
RERANKER_PY="${RERANKER_PY:-$HERE/route_reranker.py}"
ROUTE_INPUTS_PY="${ROUTE_INPUTS_PY:-$HERE/route_inputs.py}"

ROUTE_RERANK="${ROUTE_RERANK:-0}"
ROUTE_RERANK_WEIGHT="${ROUTE_RERANK_WEIGHT:-0.0005}"

for f in "$DRIVER_PKG/driver.py" "$MODEL_PY" "$CONFIG_PY" "$RERANKER_PY" "$ROUTE_INPUTS_PY"; do
    [[ -f "$f" ]] || { echo "ERROR: missing $f" >&2; exit 2; }
done

if [[ -n "$EXPECTED_CHECKPOINT_SHA256" ]]; then
actual_sha="$(docker image inspect "$IMAGE" \
    --format '{{index .Config.Labels "org.alpasim.checkpoint.sha256"}}' 2>/dev/null || true)"
[[ "$actual_sha" == "$EXPECTED_CHECKPOINT_SHA256" ]] || {
    echo "ERROR: $IMAGE does not carry the expected checkpoint" >&2
    echo "  expected: $EXPECTED_CHECKPOINT_SHA256" >&2
    echo "  actual:   ${actual_sha:-<missing image or label>}" >&2
    exit 2
}
fi

IFS=',' read -r -a gpu_indices <<< "$GPU_INDICES_CSV"
ports=(); names=()

READY_MARKER="${READY_MARKER:-policy load complete}"
if [[ "${REUSE_DRIVERS:-0}" == 1 ]]; then
    want=$(( ${#gpu_indices[@]} * REPLICAS_PER_GPU ))
    have=0; p=$BASE_PORT; reuse_addrs=()
    for gpu in "${gpu_indices[@]}"; do
        gpu="${gpu//[[:space:]]/}"
        for ((r = 0; r < REPLICAS_PER_GPU; r++)); do
            n="${CONTAINER_PREFIX}-g${gpu}-${r}"
            if docker ps --filter "name=^${n}$" --format '{{.Names}}' | grep -q . \
                    && docker logs "$n" 2>&1 | grep -q "$READY_MARKER"; then
                have=$((have + 1))
            fi
            reuse_addrs+=("\"localhost:${p}\"")
            p=$((p + 1))
        done
    done
    if (( have == want )); then
        echo "reusing ${have} running driver replicas (${CONTAINER_PREFIX}-g*)" >&2
        printf '[%s]\n' "$(IFS=,; echo "${reuse_addrs[*]}")"
        exit 0
    fi
    echo "REUSE_DRIVERS=1 but only ${have}/${want} replicas are up and ready; starting fresh" >&2
fi

port=$BASE_PORT
for gpu in "${gpu_indices[@]}"; do
    gpu="${gpu//[[:space:]]/}"
    [[ "$gpu" =~ ^[0-9]+$ ]] || { echo "ERROR: bad GPU index '$gpu'" >&2; exit 2; }
    for ((r = 0; r < REPLICAS_PER_GPU; r++)); do
        name="${CONTAINER_PREFIX}-g${gpu}-${r}"
        replica_index=${#names[@]}
        docker rm -f "$name" >/dev/null 2>&1 || true
        docker run -d \
            --name "$name" \
            --init \
            --gpus "device=${gpu}" \
            --cap-drop ALL \
            --security-opt no-new-privileges:true \
            --read-only \
            --pids-limit 1024 \
            --memory 32g \
            --cpus 8 \
            --tmpfs "/tmp:rw,nosuid,nodev,size=${TMPFS_SIZE}" \
            --tmpfs /run:rw,nosuid,nodev,size=64m \
            -v "${DRIVER_PKG}:/app/drivesuprim_challenge:ro" \
            -v "${MODEL_PY}:/app/navsim/agents/drivesuprim/drivesuprim_model.py:ro" \
            -v "${CONFIG_PY}:/app/navsim/agents/drivesuprim/drivesuprim_config.py:ro" \
            -v "${RERANKER_PY}:/app/navsim/agents/drivesuprim/route_reranker.py:ro" \
            -v "${ROUTE_INPUTS_PY}:/app/navsim/agents/drivesuprim/route_inputs.py:ro" \
            -p "127.0.0.1:${port}:${CONTAINER_PORT}" \
            -e "ALPASIM_DRIVER_HOST=0.0.0.0" \
            -e "ALPASIM_DRIVER_PORT=${CONTAINER_PORT}" \
            -e "ALPASIM_CONTESTANT_REPLICA_INDEX=${replica_index}" \
            -e "ALPASIM_CONTESTANT_REPLICAS=$((${#gpu_indices[@]} * REPLICAS_PER_GPU))" \
            -e "PYTHONDONTWRITEBYTECODE=1" \
            -e "DRIVESUPRIM_ROUTE_RERANK=${ROUTE_RERANK}" \
            -e "DRIVESUPRIM_ROUTE_RERANK_WEIGHT=${ROUTE_RERANK_WEIGHT}" \
            "$IMAGE" >/dev/null
        ports+=("$port"); names+=("$name")
        port=$((port + 1))
    done
done

echo "started ${#names[@]} driver replicas on GPUs ${GPU_INDICES_CSV}" \
     "(route_rerank=${ROUTE_RERANK}, weight=${ROUTE_RERANK_WEIGHT})" >&2

deadline=$(( $(date +%s) + READY_TIMEOUT_SEC ))
for i in "${!names[@]}"; do
    name="${names[$i]}"
    until docker logs "$name" 2>&1 | grep -q "$READY_MARKER"; do
        if ! docker ps --filter "name=^${name}$" --format '{{.Names}}' | grep -q .; then
            echo "ERROR: ${name} exited before the policy loaded" >&2
            docker logs --tail 40 "$name" >&2 2>/dev/null || true
            exit 1
        fi
        if [[ $(date +%s) -ge $deadline ]]; then
            echo "ERROR: ${name} did not report '${READY_MARKER}' in time" >&2
            exit 1
        fi
        sleep 10
    done
    echo "  ready: ${name} on 127.0.0.1:${ports[$i]}" >&2
done

printf '['
for i in "${!ports[@]}"; do
    [[ $i -gt 0 ]] && printf ','
    printf '"localhost:%s"' "${ports[$i]}"
done
printf ']\n'
