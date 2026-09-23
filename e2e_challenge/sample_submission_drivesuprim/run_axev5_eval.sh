#!/usr/bin/env bash
# Challenge-equivalent scoring run against the submitted axe-v5 image.
#
# Same as run_axe_nurec_planonly_ep09_10clips_eval.sh minus the checkpoint-label
# gate: axe-v5 predates that label, and its checkpoint identity is pinned by the
# image digest recorded in the run manifest instead.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export IMAGE="${IMAGE:-696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v5}"
export DRIVER_PORTS_CSV="${DRIVER_PORTS_CSV:-6860,6861,6862,6863}"
export RENDER_GPUS_CSV="${RENDER_GPUS_CSV:-0,1,2,3}"
export ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-4}"
export SCENE_IDS_CSV="${SCENE_IDS_CSV:?SCENE_IDS_CSV required}"
export SCENE_LIMIT=0
export SIM_STEPS="${SIM_STEPS:-199}"
export CHALLENGE_COMPAT_MODE=1
export CHALLENGE_DEBUG_VISUALIZATION="${CHALLENGE_DEBUG_VISUALIZATION:-0}"
export SERVICE_STARTUP_TIMEOUT_SEC="${SERVICE_STARTUP_TIMEOUT_SEC:-3600}"
export RENDER_VIDEO="${RENDER_VIDEO:-false}"
export GENERATE_COMBINED_VIDEO="${GENERATE_COMBINED_VIDEO:-false}"
# axe-v5 ignores DRIVESUPRIM_NAVSIM_2HZ_HISTORY; the value only labels the manifest.
export NAVSIM_2HZ_HISTORY=1

if [[ -z "${RUN_DIR:-}" ]]; then
    export RUN_DIR="$ROOT/runs/e2e_challenge_drivesuprim_debug/axev5_$(date +%Y%m%d_%H%M%S)"
fi

base_tag="alpasim-base:$(grep -m1 '^version' "$ROOT/pyproject.toml" | cut -d'"' -f2)"
expected_base="${EXPECTED_BASE_IMAGE:-alpasim-base:0.89.0-f012862-casadi372}"
base_id="$(docker image inspect "$base_tag" --format '{{.Id}}' 2>/dev/null || true)"
expected_base_id="$(docker image inspect "$expected_base" --format '{{.Id}}' 2>/dev/null || true)"
[[ -n "$base_id" && "$base_id" == "$expected_base_id" ]] || {
    echo "ERROR: $base_tag is not the current-code simulator image" >&2
    exit 2
}

exec "$SCRIPT_DIR/run_axe_nurec_vits_smoke_eval.sh"
