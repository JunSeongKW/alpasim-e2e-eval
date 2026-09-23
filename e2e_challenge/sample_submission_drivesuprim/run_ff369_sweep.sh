#!/usr/bin/env bash
# Single-clip sweep on ff369d2c -- the one clip no configuration has yet passed
# on the 10-clip set. One rollout per variant, so many combinations fit at once.
#
# Variants come from ff369_variants.txt:  name | rank_json | mpc overrides
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

IMAGE="${IMAGE:-alpasim-e2e-drivesuprim-flatvits-nodistill:ep19}"
SHA="${EXPECTED_CHECKPOINT_SHA256:-9f43a00619edfec15e5f44b5aa8942c312e382e5cd0fce8f756b270e44a22ef2}"
VARIANTS_FILE="${VARIANTS_FILE:-$SCRIPT_DIR/ff369_variants.txt}"
RANK_DIR="$SCRIPT_DIR/assets/drivesuprim/flatvits_nodistill_ep19_eval/rank_variants"
CLIPS_FILE="${CLIPS_FILE:-$SCRIPT_DIR/ff369_clip.txt}"
OUT_ROOT="${OUT_ROOT:-$ROOT/runs/e2e_challenge_drivesuprim_debug/ff369_$(date +%H%M%S)}"

[[ $# -gt 0 ]] || { echo "usage: $0 <variant> [variant ...]" >&2; exit 2; }

scene_ids="$(paste -sd, "$CLIPS_FILE")"
mkdir -p "$OUT_ROOT"
pids=(); prefixes=(); slot=${SLOT_OFFSET:-0}

for variant in "$@"; do
    line="$(awk -F'|' -v v="$variant" '$1==v {print; found=1} END{if(!found) exit 3}' "$VARIANTS_FILE")" \
        || { echo "ERROR: variant '$variant' not in $VARIANTS_FILE" >&2; exit 2; }
    rank_json="$(cut -d'|' -f2 <<< "$line")"
    mpc_ov="$(cut -d'|' -f3 <<< "$line")"
    cfg="$RANK_DIR/${rank_json}.json"
    [[ -f "$cfg" ]] || { echo "ERROR: no rank config $cfg" >&2; exit 2; }

    port=$((6900 + slot)); gpu=$(( slot % 4 )); baseport=$((6000 + slot*20))
    prefix="ff-${variant}"; log="$OUT_ROOT/_logs/$variant"
    mkdir -p "$log"; prefixes+=("$prefix")

    ( set -euo pipefail
      export IMAGE EXPECTED_CHECKPOINT_SHA256="$SHA"
      export DRIVESUPRIM_BACKBONE_TYPE=vits_flat
      export DRIVESUPRIM_CONFIG_OVERRIDE="$cfg"
      export GPU_INDICES_CSV="$gpu" DRIVER_PORTS_CSV="$port"
      export CONTAINER_PREFIX="$prefix" DRIVER_LOG_FILE="$log/driver.log"
      "$SCRIPT_DIR/run_axe_nurec_planonly_ep09_4gpu_drivers_foreground.sh" \
          > "$log/driver_stack.log" 2>&1 &
      driver_pid=$!
      for _ in $(seq 1 150); do
          ready=$(grep -c 'policy load complete' "$log/driver.log" 2>/dev/null) || ready=0
          [[ "$ready" -ge 1 ]] && break
          sleep 5
      done

      IMAGE="$IMAGE" EXPECTED_CHECKPOINT_SHA256="$SHA" \
      RUN_DIR="$OUT_ROOT/$variant" SCENE_IDS_CSV="$scene_ids" \
      DRIVER_PORTS_CSV="$port" RENDER_GPUS_CSV="$gpu" \
      ROLLOUT_WORKERS=1 NRE_CACHE_SIZE=1 RENDER_VIDEO="${RENDER_VIDEO:-true}" \
      WIZARD_BASEPORT="$baseport" SERVICE_STARTUP_TIMEOUT_SEC=2400 \
      EXTRA_OVERRIDES="$mpc_ov" \
          "$SCRIPT_DIR/run_axe_nurec_planonly_ep09_debug_video_eval.sh" \
          > "$log/eval.log" 2>&1 || echo "variant $variant eval exited $?" >&2

      kill "$driver_pid" 2>/dev/null || true
      docker rm -f $(docker ps -aq --filter "name=${prefix}") 2>/dev/null || true
    ) &
    pids+=($!)
    echo "launched $variant  rank=$rank_json port=$port gpu=$gpu baseport=$baseport"
    slot=$((slot+1))
done

status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
for pre in "${prefixes[@]}"; do docker rm -f $(docker ps -aq --filter "name=${pre}") 2>/dev/null || true; done
echo "ff369 sweep done (status $status): $OUT_ROOT"
