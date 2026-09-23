#!/usr/bin/env bash
# Experiment 1: MPC gain sweep on the three flat-ViT failure clips.
#
# Every variant runs the same checkpoint, the same ranking weights and the same
# renderer; only the controller gains differ, applied as hydra overrides so the
# wizard regenerates controller-config.yaml per run. Nothing is rebuilt.
#
#   ./run_mpc_gain_sweep.sh base track smooth
#
# Variant names and their overrides come from mpc_gain_variants.txt.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

IMAGE="${IMAGE:-alpasim-e2e-drivesuprim-flatvits-nodistill:ep19}"
SHA="${EXPECTED_CHECKPOINT_SHA256:-9f43a00619edfec15e5f44b5aa8942c312e382e5cd0fce8f756b270e44a22ef2}"
VARIANTS_FILE="${VARIANTS_FILE:-$SCRIPT_DIR/mpc_gain_variants.txt}"
CLIPS_FILE="${CLIPS_FILE:-$SCRIPT_DIR/rank3clips.txt}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/e2e_challenge_drivesuprim_debug/mpcgains_$(date +%H%M%S)}"

[[ $# -gt 0 ]] || { echo "usage: $0 <variant> [variant ...]" >&2; exit 2; }

scene_ids="$(paste -sd, "$CLIPS_FILE")"
mkdir -p "$OUT_ROOT"
pids=(); prefixes=()
# Lets a second sweep run alongside one already in flight without
# colliding on driver ports, wizard baseports or container names.
slot=${SLOT_OFFSET:-0}

for variant in "$@"; do
    overrides="$(awk -F'|' -v v="$variant" '$1==v {print $2; found=1} END{if(!found) exit 3}' "$VARIANTS_FILE")" \
        || { echo "ERROR: variant '$variant' not in $VARIANTS_FILE" >&2; exit 2; }

    # Distinct ports, container prefix and compose project per variant. The
    # compose project name is the RUN_DIR basename, so that must be unique too
    # or two stacks fight over the same container names.
    ports="$((6900 + slot*10)),$((6901 + slot*10)),$((6902 + slot*10))"
    gpus="$(( slot % 4 )),$(( (slot+1) % 4 )),$(( (slot+2) % 4 ))"
    baseport=$((6000 + slot*100))
    prefix="mg-${variant}"
    log="$OUT_ROOT/_logs/$variant"
    mkdir -p "$log"
    prefixes+=("$prefix")

    ( set -euo pipefail
      export IMAGE EXPECTED_CHECKPOINT_SHA256="$SHA"
      export DRIVESUPRIM_BACKBONE_TYPE=vits_flat
      export GPU_INDICES_CSV="$gpus" DRIVER_PORTS_CSV="$ports"
      export CONTAINER_PREFIX="$prefix" DRIVER_LOG_FILE="$log/driver.log"
      "$SCRIPT_DIR/run_axe_nurec_planonly_ep09_4gpu_drivers_foreground.sh" \
          > "$log/driver_stack.log" 2>&1 &
      driver_pid=$!
      for _ in $(seq 1 150); do
          ready=$(grep -c 'policy load complete' "$log/driver.log" 2>/dev/null) || ready=0
          [[ "$ready" -ge 3 ]] && break
          sleep 5
      done

      IMAGE="$IMAGE" EXPECTED_CHECKPOINT_SHA256="$SHA" \
      RUN_DIR="$OUT_ROOT/$variant" SCENE_IDS_CSV="$scene_ids" \
      DRIVER_PORTS_CSV="$ports" RENDER_GPUS_CSV="$gpus" \
      ROLLOUT_WORKERS=3 NRE_CACHE_SIZE=1 RENDER_VIDEO="${RENDER_VIDEO:-true}" \
      WIZARD_BASEPORT="$baseport" SERVICE_STARTUP_TIMEOUT_SEC=2400 \
      EXTRA_OVERRIDES="$overrides" \
          "$SCRIPT_DIR/run_axe_nurec_planonly_ep09_debug_video_eval.sh" \
          > "$log/eval.log" 2>&1 || echo "variant $variant eval exited $?" >&2

      kill "$driver_pid" 2>/dev/null || true
      docker rm -f $(docker ps -aq --filter "name=${prefix}") 2>/dev/null || true
    ) &
    pids+=($!)
    echo "launched $variant  ports=$ports gpus=$gpus baseport=$baseport"
    echo "    overrides: ${overrides:-<none, stock gains>}"
    slot=$((slot+1))
done

status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
for pre in "${prefixes[@]}"; do docker rm -f $(docker ps -aq --filter "name=${pre}") 2>/dev/null || true; done
echo "mpc sweep done (status $status): $OUT_ROOT"
