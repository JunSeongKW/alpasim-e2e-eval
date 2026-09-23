#!/usr/bin/env bash
# start_drivers.sh with the driver package overridden from the working tree.
#
# The submission image bakes `drivesuprim_challenge` in at /app and the
# container runs read-only, so the two differences from the original launcher
# are a bind mount over that package and a way to pass the feature's own
# environment variables through OFFICIAL_ENV_ONLY. Everything else -- the
# checkpoint label guard, the capability drops, the tmpfs sizes, the readiness
# probe -- is left exactly as the scoring runs have it, because the A/B is only
# meaningful if the baseline arm is the submitted configuration.
#
# The mounted source is byte-identical to the image's copy apart from the
# feature, and the feature is off unless DRIVESUPRIM_ROUTE_CACHE says otherwise,
# so the "before" arm mounts the same files as the "after" arm. That keeps the
# comparison to one variable instead of two.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="${IMAGE:-alpasim-e2e-drivesuprim-stage3:merged-route-ep30}"
EXPECTED_CHECKPOINT_SHA256="${EXPECTED_CHECKPOINT_SHA256-364c801e397e9d6e36ea1f9027dc4ca804ad2e429bb588cf4e84185ca851de3d}"
GPU_INDICES_CSV="${GPU_INDICES_CSV:-4,5,6,7}"
REPLICAS_PER_GPU="${REPLICAS_PER_GPU:-1}"
BASE_PORT="${BASE_PORT:-6940}"
CONTAINER_PREFIX="${CONTAINER_PREFIX:-axe-routecache}"
CONTAINER_PORT="${CONTAINER_PORT:-6789}"
TMPFS_SIZE="${TMPFS_SIZE:-2g}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-900}"
DRIVER_PKG="${DRIVER_PKG:-$HERE/drivesuprim_challenge}"
# The corridor gate belongs beside the drivable/collision gates, which live in
# the model rather than the adapter, so one navsim file is overridden too.
# Mounting the single file keeps the rest of the package the image's own.
MODEL_PY="${MODEL_PY:-$HERE/drivesuprim_model.py}"
# The agent copies features into per-forward dicts by an explicit whitelist, so
# a new key has to be named there too or it never reaches the model.
AGENT_PY="${AGENT_PY:-$HERE/drivesuprim_agent.py}"

# The feature's own switches. Both arms of the A/B set these explicitly rather
# than relying on the defaults, so the log of either run states what it ran.
ROUTE_CACHE="${ROUTE_CACHE:-0}"
ROUTE_CORRIDOR_M="${ROUTE_CORRIDOR_M:-4.0}"
ROUTE_GATE_HORIZON_S="${ROUTE_GATE_HORIZON_S:-4.0}"

[[ -f "$DRIVER_PKG/driver.py" ]] || {
    echo "ERROR: no driver package at $DRIVER_PKG" >&2; exit 2; }
[[ -f "$MODEL_PY" ]] || {
    echo "ERROR: no model file at $MODEL_PY" >&2; exit 2; }
[[ -f "$AGENT_PY" ]] || {
    echo "ERROR: no agent file at $AGENT_PY" >&2; exit 2; }

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

# Reuse. When every replica this call would create is already running and has
# logged READY_MARKER, print their addresses and stop. A retry that is about
# the simulator side (a mount, a probe timeout) then skips the ten-plus
# minutes of loading sixteen checkpoints. Anything short of a complete, ready
# set falls through to a fresh start, so a half-dead pool is never reused.
# Identity is the container name only (prefix + gpu + index): use a new
# CONTAINER_PREFIX whenever the image or the driver environment changes.
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
            -v "${AGENT_PY}:/app/navsim/agents/drivesuprim/drivesuprim_agent.py:ro" \
            -p "127.0.0.1:${port}:${CONTAINER_PORT}" \
            -e "ALPASIM_DRIVER_HOST=0.0.0.0" \
            -e "ALPASIM_DRIVER_PORT=${CONTAINER_PORT}" \
            -e "ALPASIM_CONTESTANT_REPLICA_INDEX=${replica_index}" \
            -e "ALPASIM_CONTESTANT_REPLICAS=$((${#gpu_indices[@]} * REPLICAS_PER_GPU))" \
            -e "DRIVESUPRIM_ROUTE_CACHE=${ROUTE_CACHE}" \
            -e "DRIVESUPRIM_ROUTE_CORRIDOR_M=${ROUTE_CORRIDOR_M}" \
            -e "DRIVESUPRIM_ROUTE_GATE_HORIZON_S=${ROUTE_GATE_HORIZON_S}" \
            "$IMAGE" >/dev/null
        ports+=("$port"); names+=("$name")
        port=$((port + 1))
    done
done

echo "started ${#names[@]} driver replicas on GPUs ${GPU_INDICES_CSV}" \
     "(route_cache=${ROUTE_CACHE}, corridor=${ROUTE_CORRIDOR_M}m," \
     "horizon=${ROUTE_GATE_HORIZON_S}s)" >&2

READY_MARKER="${READY_MARKER:-policy load complete}"
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
