#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Keep challenge scoring/controller semantics while enabling only trusted,
# local post-rollout DriveSuprim visualization support.
export CHALLENGE_DEBUG_VISUALIZATION=1
export RENDER_VIDEO="${RENDER_VIDEO:-true}"
export GENERATE_COMBINED_VIDEO="${GENERATE_COMBINED_VIDEO:-true}"

if [[ -z "${RUN_DIR:-}" ]]; then
    RUN_ID="$(date +%Y%m%d_%H%M%S)"
    export RUN_DIR="$ROOT/runs/e2e_challenge_drivesuprim_debug/axe_nurec_planonly_fixed_step43590_10clips_video_${RUN_ID}"
fi

exec "$SCRIPT_DIR/run_axe_nurec_planonly_fixed_step43590_challenge_10clips_eval.sh"
