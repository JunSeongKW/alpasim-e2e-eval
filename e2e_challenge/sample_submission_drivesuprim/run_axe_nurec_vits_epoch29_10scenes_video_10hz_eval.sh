#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IMAGE="${IMAGE:-alpasim-e2e-drivesuprim-axe-nurec-vits-driver:epoch29-step60040-h100}"
export NAVSIM_2HZ_HISTORY=0

exec "$SCRIPT_DIR/run_axe_nurec_vits_epoch29_10scenes_video_eval.sh"
