#!/usr/bin/env bash
# ep09 scoring run with the local DriveSuprim debug-visualization overlay.
#
# Keeps the challenge's score-affecting semantics (nonlinear MPC, force-GT
# handover, official runtime image) and additionally mounts the local evaluator
# so it can decode the driver's pickled debug payload and draw the
# "DriveSuprim perception + candidate BEV" panel into the videos.
# Diagnostic use only -- never point this at an untrusted driver image.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export CHALLENGE_DEBUG_VISUALIZATION=1
export RENDER_VIDEO="${RENDER_VIDEO:-true}"
export GENERATE_COMBINED_VIDEO="${GENERATE_COMBINED_VIDEO:-false}"

if [[ -z "${RUN_DIR:-}" ]]; then
    RUN_ID="$(date +%Y%m%d_%H%M%S)"
    export RUN_DIR="$ROOT/runs/e2e_challenge_drivesuprim_debug/axe_nurec_planonly_ep09_video_${RUN_ID}"
fi

exec "$SCRIPT_DIR/run_axe_nurec_planonly_ep09_10clips_eval.sh"
