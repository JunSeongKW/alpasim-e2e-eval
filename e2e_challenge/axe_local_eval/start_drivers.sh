#!/usr/bin/env bash
# Start the AXE DriveSuprim driver replicas that the curated_val run talks to.
#
# The wizard treats the driver as an external service, so the replicas are
# started here and only their host:port list is handed to alpasim. Replicas are
# spread round-robin across GPU_INDICES_CSV, matching the competition EC2
# topology (16 contestant containers on GPUs 4-7).
set -euo pipefail

IMAGE="${IMAGE:-alpasim-e2e-drivesuprim-stage3:ep24-sweep}"
# sha256 of nurec_stage3_ep24/checkpoint/epoch=24-step=18150.ckpt
EXPECTED_CHECKPOINT_SHA256="${EXPECTED_CHECKPOINT_SHA256-1d9fa8d7b8bb4f6d5f732cdf203a3a44c603780c1086726be841eb050d1212dc}"
GPU_INDICES_CSV="${GPU_INDICES_CSV:-4,5,6,7}"
REPLICAS_PER_GPU="${REPLICAS_PER_GPU:-4}"
BASE_PORT="${BASE_PORT:-6860}"
CONTAINER_PREFIX="${CONTAINER_PREFIX:-axe-curatedval-drv}"
CONTAINER_PORT="${CONTAINER_PORT:-6789}"
# The competition runtime contract caps the driver's writable /tmp at 2 GiB.
TMPFS_SIZE="${TMPFS_SIZE:-2g}"
READY_TIMEOUT_SEC="${READY_TIMEOUT_SEC:-900}"

# Older submission images (axe-v4, axe-v5) predate the checkpoint label, so the
# guard has to be skippable: set EXPECTED_CHECKPOINT_SHA256= to opt out.
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

# Official evaluation hands the contestant container only these four variables,
# so everything else has to come from the image. Set OFFICIAL_ENV_ONLY=1 to
# reproduce that exactly: any DRIVESUPRIM_* flag still needed must be baked in,
# and a run that behaves differently under this mode would behave differently on
# the leaderboard.
OFFICIAL_ENV_ONLY="${OFFICIAL_ENV_ONLY:-0}"

IFS=',' read -r -a gpu_indices <<< "$GPU_INDICES_CSV"
ports=()
names=()

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
        if [[ "$OFFICIAL_ENV_ONLY" == 1 ]]; then
            env_args=(
                -e "ALPASIM_DRIVER_HOST=0.0.0.0"
                -e "ALPASIM_DRIVER_PORT=${CONTAINER_PORT}"
                -e "ALPASIM_CONTESTANT_REPLICA_INDEX=${replica_index}"
                -e "ALPASIM_CONTESTANT_REPLICAS=$((${#gpu_indices[@]} * REPLICAS_PER_GPU))"
            )
        else
            env_args=(
                -e "ALPASIM_DRIVER_HOST=0.0.0.0"
                -e "ALPASIM_DRIVER_PORT=${CONTAINER_PORT}"
                -e "DRIVESUPRIM_REQUIRE_POLICY_FOR_VERSION=1"
                -e "DRIVESUPRIM_SESSION_STARTUP_TIMEOUT_S=1200"
                -e "DRIVESUPRIM_MAX_BATCH_SIZE=1"
                -e "DRIVESUPRIM_INFERENCE_INTERVAL_US=100000"
                -e "DRIVESUPRIM_USE_FP16=0"
                -e "DRIVESUPRIM_NAVSIM_2HZ_HISTORY=1"
                -e "DRIVESUPRIM_REQUIRE_EXACT_CHECKPOINT=1"
                -e "DRIVESUPRIM_DISABLE_INFERENCE=0"
                -e "ALPASIM_DRIVER_GRPC_WORKERS=8"
                -e "DRIVESUPRIM_EGO_FOOTPRINT_FROM_API=${DRIVESUPRIM_EGO_FOOTPRINT_FROM_API:-1}"
                -e "DRIVESUPRIM_EGO_CENTER_OFFSET=${DRIVESUPRIM_EGO_CENTER_OFFSET:-0}"
            )
        fi
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
            -p "127.0.0.1:${port}:${CONTAINER_PORT}" \
            "${env_args[@]}" \
            "$IMAGE" >/dev/null
        ports+=("$port")
        names+=("$name")
        port=$((port + 1))
    done
done

echo "started ${#names[@]} driver replicas on GPUs ${GPU_INDICES_CSV}"

# Docker binds the published host port as soon as the container starts, well
# before the checkpoint is on the GPU, so a TCP connect is not a readiness
# signal. The driver logs a line once the policy is actually loaded.
READY_MARKER="${READY_MARKER:-policy load complete}"
deadline=$(( $(date +%s) + READY_TIMEOUT_SEC ))
for i in "${!names[@]}"; do
    name="${names[$i]}"
    until docker logs "$name" 2>&1 | grep -q "$READY_MARKER"; do
        if ! docker ps --filter "name=^${name}$" --format '{{.Names}}' | grep -q .; then
            echo "ERROR: ${name} exited before the policy loaded" >&2
            docker logs --tail 20 "$name" >&2 2>/dev/null || true
            exit 1
        fi
        if [[ $(date +%s) -ge $deadline ]]; then
            echo "ERROR: ${name} did not report '${READY_MARKER}' in time" >&2
            exit 1
        fi
        sleep 10
    done
    echo "  ready: ${name} on 127.0.0.1:${ports[$i]}"
done

# Hydra list literal for wizard.external_services.driver.
printf '['
for i in "${!ports[@]}"; do
    [[ $i -gt 0 ]] && printf ','
    printf '"localhost:%s"' "${ports[$i]}"
done
printf ']\n'
