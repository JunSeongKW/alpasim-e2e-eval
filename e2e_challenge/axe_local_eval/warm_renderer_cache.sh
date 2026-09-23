#!/usr/bin/env bash
# Seed the renderer's shared model cache once, so a run's sixteen renderers
# download nothing.
#
# Every nre-ga renderer fetches ~2.8 GB at startup into $HOME/.cache
# (nre/ harmonizer 1.4 GB + huggingface/ 1.4 GB). With one container per
# replica that is sixteen copies of the same download landing on the same disk
# at once, and the runtime's version probe gives up (1792 s) before they finish.
# run_curated_val.sh therefore mounts one host directory as every renderer's
# $HOME/.cache -- but sixteen replicas downloading into an EMPTY shared
# directory race on the final rename and die with "The provided filename ...
# does not exist". Hence this script: one renderer, alone, fills the directory;
# after that no renderer writes there.
#
# Adapted from another researcher's scripts/warm_renderer_cache.sh on this
# machine, which mounts at /tmp/.cache with HOME=/tmp. The nre-ga image has
# HOME=/home, and run_curated_val.sh keeps that, so the mount here is
# /home/.cache -- a cache at any other path is simply never read (verified:
# zero reads of a /tmp/.cache mount five minutes into a run).
#
# Usage
#   warm_renderer_cache.sh              # fills the cache only if it is empty
#   FORCE=1 warm_renderer_cache.sh      # refills it
#   RENDER_CACHE=... WARM_GPU=4 ...     # overrides
#
# Exit 0 = cache is seeded (already, or now). Non-zero = it is not, and a run
# launched against it would race.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RENDER_CACHE="${RENDER_CACHE:-$ROOT/.cache/renderer-shared}"
IMAGE="${RENDERER_IMAGE:-nvcr.io/nvidia/nre/nre-ga:26.04}"
WARM_GPU="${WARM_GPU:-4}"
SCENE_CACHE="${SCENE_CACHE:-$ROOT/data/nre-artifacts}"
# Any real sceneset works: the weights being fetched are scene-independent.
ARTIFACT_GLOB="${ARTIFACT_GLOB:-}"
# Same as `defines.renderer_entrypoint` / renderer command in base_config.yaml.
ENTRYPOINT="${RENDERER_ENTRYPOINT:-/app/internal/scripts/pycena/runtime/pycena_nrm_full}"
# "Seeded" means both download targets are present; either alone is a
# half-finished warm-up.
MIN_MB="${MIN_MB:-2000}"
WARM_TIMEOUT_SEC="${WARM_TIMEOUT_SEC:-2400}"

seeded_mb() {
    local mb=0 d
    for d in huggingface nre; do
        [[ -d "$RENDER_CACHE/$d" ]] || { echo 0; return; }
        mb=$(( mb + $(du -sm "$RENDER_CACHE/$d" 2>/dev/null | cut -f1) ))
    done
    echo "$mb"
}

mkdir -p "$RENDER_CACHE"
# Renderer containers run as root and the cache must stay writable by whoever
# seeds it next; a shared dir that one uid owns is the failure mode in waiting.
chmod 777 "$RENDER_CACHE" 2>/dev/null || true

have="$(seeded_mb)"
if [[ "${FORCE:-0}" != 1 && "$have" -ge "$MIN_MB" ]]; then
    echo "renderer cache seeded: ${have} MB in $RENDER_CACHE" >&2
    exit 0
fi

if [[ -z "$ARTIFACT_GLOB" ]]; then
    first="$(find "$SCENE_CACHE/scenesets" -mindepth 1 -maxdepth 1 -type d | head -1)"
    [[ -n "$first" ]] || { echo "ERROR: no sceneset under $SCENE_CACHE/scenesets" >&2; exit 2; }
    ARTIFACT_GLOB="/mnt/nre-data/scenesets/$(basename "$first")/**/*.usdz"
fi

name="warm-renderer-cache-$$"
trap 'docker rm -f "$name" >/dev/null 2>&1 || true' EXIT
echo "seeding renderer cache (${have} MB now) with one renderer on GPU $WARM_GPU: $RENDER_CACHE" >&2

# Command mirrors the wizard's renderer service exactly, so the same files are
# fetched. umask 0000 keeps what root writes readable/writable for everyone.
docker run -d --name "$name" \
    --gpus "device=${WARM_GPU}" \
    -e PYTHONDONTWRITEBYTECODE=1 \
    -e OMP_NUM_THREADS=1 \
    -v "$RENDER_CACHE:/home/.cache" \
    -v "$SCENE_CACHE:/mnt/nre-data" \
    -v "$SCENE_CACHE/ego-hoods:/mnt/ego-hoods" \
    "$IMAGE" \
    bash -c "umask 0000; exec $ENTRYPOINT serve-grpc --port=6000 --host=0.0.0.0 \
        --artifact-glob='$ARTIFACT_GLOB' --egocar-hood-dir=/mnt/ego-hoods \
        --no-enable-nrend --enable-harmonizer --download-cache-dir /tmp/nre-cache-dir \
        --cache-size=1 --max-workers=4 --enable-editing-actors" >/dev/null

deadline=$(( $(date +%s) + WARM_TIMEOUT_SEC ))
while :; do
    if docker logs "$name" 2>&1 | grep -q 'Serving on 0.0.0.0'; then
        have="$(seeded_mb)"
        if [[ "$have" -ge "$MIN_MB" ]]; then
            echo "renderer cache seeded: ${have} MB" >&2
            exit 0
        fi
        echo "ERROR: renderer bound but cache holds only ${have} MB (< ${MIN_MB}); wrong mount path?" >&2
        exit 1
    fi
    if ! docker ps --filter "name=^${name}$" --format '{{.Names}}' | grep -q .; then
        echo "ERROR: warm-up renderer exited:" >&2
        docker logs --tail 30 "$name" >&2 2>/dev/null || true
        exit 1
    fi
    if [[ $(date +%s) -ge $deadline ]]; then
        echo "ERROR: warm-up renderer not bound after ${WARM_TIMEOUT_SEC}s" >&2
        docker logs --tail 30 "$name" >&2 2>/dev/null || true
        exit 1
    fi
    sleep 10
done
