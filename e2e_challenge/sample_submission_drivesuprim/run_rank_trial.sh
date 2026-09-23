#!/usr/bin/env bash
# One sweep trial: run a ranking variant over the screening clips and append its
# result to RANK_SWEEP_RESULTS.md.
#
#   ./run_rank_trial.sh R1a "fine imi 0.0"
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
NAME="${1:?variant name required}"
NOTE="${2:-}"

P="$SCRIPT_DIR/assets/drivesuprim/stage3_ep24_eval"
CFG="$P/rank_variants/${NAME}.json"
[[ -f "$CFG" ]] || { echo "ERROR: no variant config $CFG" >&2; exit 2; }

IMAGE="${IMAGE:-alpasim-e2e-drivesuprim-stage3:ep24-sweep}"
SHA="${EXPECTED_CHECKPOINT_SHA256:-1d9fa8d7b8bb4f6d5f732cdf203a3a44c603780c1086726be841eb050d1212dc}"
CLIPS="${CLIPS_FILE:-$SCRIPT_DIR/val150_clips.txt}"
# A fresh RUN_DIR name every time: compose derives its project name from the
# basename, and a name still being torn down cannot be reused.
RUN_DIR="${RUN_DIR:-$ROOT/runs/e2e_challenge_drivesuprim_debug/sweep_${NAME}_$(date +%H%M%S)}"

IMAGE="$IMAGE" EXPECTED_CHECKPOINT_SHA256="$SHA" \
DRIVESUPRIM_BACKBONE_TYPE=bevformer_m DRIVESUPRIM_CONFIG_OVERRIDE="$CFG" \
MPC_OVERRIDES="controller.gains.long_position_weight=0.5 controller.gains.lat_position_weight=6.0 controller.gains.idx_start_penalty=3" \
CLIPS_FILE="$CLIPS" WORKERS="${WORKERS:-16}" GPUS="${GPUS:-0 1 2 3}" \
BASE_PORT="${BASE_PORT:-6900}" BASEPORT="${BASEPORT:-6400}" PREFIX="${PREFIX:-sw}" \
RENDER_VIDEO=false SERVICE_STARTUP_TIMEOUT_SEC=5400 RUN_DIR="$RUN_DIR" \
    "$SCRIPT_DIR/run_val_sweep.sh"

python3 "$SCRIPT_DIR/record_rank_trial.py" --name "$NAME" --run-dir "$RUN_DIR" --note "$NOTE"

# A 150-clip trial leaves ~37 GB of rollout logs. The scores are already in
# aggregate/, and a full disk stops the sweep dead -- so drop the raw rollouts
# once they have been read.
if [[ -f "$RUN_DIR/aggregate/results-summary.json" ]]; then
    cp "$RUN_DIR/aggregate/results-summary.json" \
       "$SCRIPT_DIR/sweep_archive/$(basename "$RUN_DIR").json" 2>/dev/null || true
    rm -rf "$RUN_DIR/rollouts" "$RUN_DIR/telemetry" 2>/dev/null || true
fi
echo "trial $NAME done: $RUN_DIR"
