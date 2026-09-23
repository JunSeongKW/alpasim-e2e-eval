#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export IMAGE="${IMAGE:-alpasim-e2e-drivesuprim-axe-nurec-planonly-fixed-vits-driver:epoch29-step43590-h100}"
export EXPECTED_CHECKPOINT_SHA256="${EXPECTED_CHECKPOINT_SHA256:-b4bc194d81795f774ccaf4da30b403f0b233a23b03d5b4bf369c2a004faf0d33}"
export DRIVER_PORTS_CSV="${DRIVER_PORTS_CSV:-6860,6861,6862,6863}"
export RENDER_GPUS_CSV="${RENDER_GPUS_CSV:-0,1,2,3}"
export ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-4}"
export SCENE_IDS_CSV="${SCENE_IDS_CSV:-39875322-9804-49ef-afe2-93719b034d37,e3e470e4-1d47-4ae7-a543-ca18292cc1f8,f4e31360-9d31-4064-8cfb-efd7f3e072b7,c099d336-1152-4d57-b121-f325d8acb0e4,00169207-9da7-44c4-a67b-a925e77056ff,bdc3344d-4e43-45b9-807c-f7526199e4ef,ddc3e8df-b368-40f0-bb6a-9d9f4f568cd4,00442956-c080-4b2e-b9c7-647973969c86,161aff42-2c7f-4d9c-8a24-97a606bf3df8,c9047f0b-3765-4793-b6f8-e05eaadcad3b}"
export SCENE_LIMIT=0
export SIM_STEPS="${SIM_STEPS:-199}"
export NAVSIM_2HZ_HISTORY=1
export CHALLENGE_COMPAT_MODE=1
export SERVICE_STARTUP_TIMEOUT_SEC="${SERVICE_STARTUP_TIMEOUT_SEC:-900}"

# Public challenge scoring does not render videos. Override explicitly only for
# a separate diagnostic run; keep the score run at these defaults.
export RENDER_VIDEO="${RENDER_VIDEO:-false}"
export GENERATE_COMBINED_VIDEO="${GENERATE_COMBINED_VIDEO:-false}"

if [[ -z "${RUN_DIR:-}" ]]; then
    RUN_ID="$(date +%Y%m%d_%H%M%S)"
    export RUN_DIR="$ROOT/runs/e2e_challenge_drivesuprim_challenge/axe_nurec_planonly_fixed_step43590_10clips_${RUN_ID}"
fi

actual_checkpoint_sha256="$(docker image inspect "$IMAGE" --format '{{index .Config.Labels "org.alpasim.checkpoint.sha256"}}' 2>/dev/null || true)"
[[ "$actual_checkpoint_sha256" == "$EXPECTED_CHECKPOINT_SHA256" ]] || {
    echo "ERROR: $IMAGE is not the calibration-fixed checkpoint image" >&2
    echo "  expected: $EXPECTED_CHECKPOINT_SHA256" >&2
    echo "  actual:   ${actual_checkpoint_sha256:-missing image/label}" >&2
    exit 2
}

exec "$SCRIPT_DIR/run_axe_nurec_vits_smoke_eval.sh"
