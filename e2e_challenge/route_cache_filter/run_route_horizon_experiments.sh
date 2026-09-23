#!/usr/bin/env bash
# Run the two ten-clip route-gate horizon experiments sequentially, with video.
#
# This host has GPUs 0-7.  One driver and one renderer are placed on each GPU,
# while only four rollouts run concurrently. That keeps the per-GPU service
# footprint low while producing the same individual and combined videos as the
# earlier routecache4-after run.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/e2e_challenge/route_cache_filter"
LOG="$HERE/route_horizon_experiments.log"
GPU_INDICES_CSV="0,1,2,3,4,5,6,7"
MAX_START_USED_MIB="${MAX_START_USED_MIB:-12000}"

log() { echo "[$(date '+%F %H:%M:%S')] $*" | tee -a "$LOG"; }

wait_for_gpu_capacity() {
    local used count ready index mib
    while true; do
        mapfile -t used < <(
            nvidia-smi --query-gpu=index,memory.used \
                --format=csv,noheader,nounits
        )
        count="${#used[@]}"
        ready=1
        if [[ "$count" -ne 8 ]]; then
            ready=0
        else
            for row in "${used[@]}"; do
                index="${row%%,*}"
                mib="${row##*, }"
                if [[ ! "$index" =~ ^[0-7]$ || ! "$mib" =~ ^[0-9]+$ \
                      || "$mib" -gt "$MAX_START_USED_MIB" ]]; then
                    ready=0
                    break
                fi
            done
        fi
        if [[ "$ready" -eq 1 ]]; then
            log "GPU 0-7 are below ${MAX_START_USED_MIB} MiB; starting experiments"
            return 0
        fi
        log "waiting for GPU 0-7 capacity: ${used[*]}"
        sleep 60
    done
}

run_horizon() {
    local seconds="$1" tag="$2"
    log "START horizon=${seconds}s tag=${tag} (8 drivers/renderers, 4 workers)"
    if RUN_TAG="$tag" \
       ARMS=after \
       ROUTE_GATE_HORIZON_S="$seconds" \
       GPUS="$GPU_INDICES_CSV" \
       RENDER_GPUS="$GPU_INDICES_CSV" \
       ROLLOUT_WORKERS=4 \
       RENDERER_REPLICAS_PER_GPU=1 \
       "$HERE/run_ab_10clips.sh"; then
        log "DONE horizon=${seconds}s tag=${tag}"
    else
        local status=$?
        log "FAILED horizon=${seconds}s tag=${tag} exit=${status}"
        return "$status"
    fi
}

log "route horizon experiment queue created"
wait_for_gpu_capacity
run_horizon 1 routecache-h1 || exit $?
wait_for_gpu_capacity
run_horizon 2 routecache-h2 || exit $?
log "ALL DONE"
