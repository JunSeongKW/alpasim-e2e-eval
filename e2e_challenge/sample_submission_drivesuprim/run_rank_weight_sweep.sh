#!/usr/bin/env bash
# Ranking-weight sweep on the three flat-ViT failure clips.
#
# Each variant differs only in the DriveSuprim ranking weights, mounted over the
# image's config via DRIVESUPRIM_CONFIG_OVERRIDE -- the checkpoint, the renderer
# and the controller are identical across variants, so a score difference is
# attributable to the weights and nothing else.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

IMAGE="${IMAGE:-alpasim-e2e-drivesuprim-flatvits-nodistill:ep19}"
SHA="${EXPECTED_CHECKPOINT_SHA256:-9f43a00619edfec15e5f44b5aa8942c312e382e5cd0fce8f756b270e44a22ef2}"
VARIANTS_DIR="$SCRIPT_DIR/assets/drivesuprim/flatvits_nodistill_ep19_eval/rank_variants"
CLIPS_FILE="${CLIPS_FILE:-$SCRIPT_DIR/rank3clips.txt}"
# Controller gains held fixed across every ranking variant. Defaults to the
# experiment-1 winner (lat_first), so a score difference here is attributable
# to the ranking weights and not to the controller.
MPC_OVERRIDES="${MPC_OVERRIDES:-controller.gains.long_position_weight=0.5 controller.gains.lat_position_weight=6.0 controller.gains.idx_start_penalty=3}"
STAMP="${STAMP:-$(date +%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/e2e_challenge_drivesuprim_debug/rankweights_${STAMP}}"

# variant : driver-ports : gpus : wizard-baseport : container-prefix
BATCH=("$@")
[[ ${#BATCH[@]} -gt 0 ]] || { echo "usage: $0 <variant:ports:gpus:baseport:prefix> ..." >&2; exit 2; }

scene_ids="$(paste -sd, "$CLIPS_FILE")"
mkdir -p "$OUT_ROOT"
pids=(); prefixes=()

for spec in "${BATCH[@]}"; do
    IFS=: read -r variant ports gpus baseport prefix <<< "$spec"
    cfg="$VARIANTS_DIR/${variant}.json"
    [[ -f "$cfg" ]] || { echo "ERROR: no variant config $cfg" >&2; exit 2; }
    log="$OUT_ROOT/_logs/${variant}"
    mkdir -p "$log"
    prefixes+=("$prefix")

    ( set -euo pipefail
      export IMAGE EXPECTED_CHECKPOINT_SHA256="$SHA"
      export DRIVESUPRIM_BACKBONE_TYPE=vits_flat
      export DRIVESUPRIM_CONFIG_OVERRIDE="$cfg"
      export GPU_INDICES_CSV="$gpus" DRIVER_PORTS_CSV="$ports"
      export CONTAINER_PREFIX="$prefix"
      export DRIVER_LOG_FILE="$log/driver.log"
      "$SCRIPT_DIR/run_axe_nurec_planonly_ep09_4gpu_drivers_foreground.sh" \
          > "$log/driver_stack.log" 2>&1 &
      driver_pid=$!

      # Drivers must answer before the wizard dials them.
      for _ in $(seq 1 150); do
          ready=$(grep -c 'policy load complete' "$log/driver.log" 2>/dev/null) || ready=0
          [[ "$ready" -ge 3 ]] && break
          sleep 5
      done

      IMAGE="$IMAGE" EXPECTED_CHECKPOINT_SHA256="$SHA" \
      RUN_DIR="$OUT_ROOT/$variant" SCENE_IDS_CSV="$scene_ids" \
      DRIVER_PORTS_CSV="$ports" RENDER_GPUS_CSV="$gpus" \
      ROLLOUT_WORKERS=3 NRE_CACHE_SIZE=1 RENDER_VIDEO=true \
      WIZARD_BASEPORT="$baseport" SERVICE_STARTUP_TIMEOUT_SEC=2400 \
      EXTRA_OVERRIDES="$MPC_OVERRIDES" \
          "$SCRIPT_DIR/run_axe_nurec_planonly_ep09_debug_video_eval.sh" \
          > "$log/eval.log" 2>&1 || echo "variant $variant eval exited $?" >&2

      kill "$driver_pid" 2>/dev/null || true
      docker rm -f $(docker ps -q --filter "name=${prefix}") 2>/dev/null || true
    ) &
    pids+=($!)
    echo "launched $variant  ports=$ports gpus=$gpus baseport=$baseport -> $log"
    echo "    mpc: $MPC_OVERRIDES"
done

status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
for pre in "${prefixes[@]}"; do docker rm -f $(docker ps -q --filter "name=${pre}") 2>/dev/null || true; done
echo "sweep done (status $status): $OUT_ROOT"
