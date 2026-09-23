#!/usr/bin/env bash
# Official-equivalent 10-clip scoring run for the ep09 checkpoint.
#
# Runtime/controller/eval code comes from the container image, which must be
# built from the current public e2e_challenge tip (f012862). The July
# alpasim-base:0.89.0 build predates the August dynamics/handover and
# progress/offroad scoring changes, so this script refuses to run against it.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export IMAGE="${IMAGE:-alpasim-e2e-drivesuprim-axe-nurec-planonly-ep09-driver:ep09-step4840-h100}"
export EXPECTED_CHECKPOINT_SHA256="${EXPECTED_CHECKPOINT_SHA256:-90ff4189f3e464ca681a1d17e7615b0d04b2de768429d21eb5aa85007fe2b76d}"
export DRIVER_PORTS_CSV="${DRIVER_PORTS_CSV:-6860,6861,6862,6863}"
export RENDER_GPUS_CSV="${RENDER_GPUS_CSV:-4,5,6,7}"
export ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-4}"
export SCENE_IDS_CSV="${SCENE_IDS_CSV:-39875322-9804-49ef-afe2-93719b034d37,e3e470e4-1d47-4ae7-a543-ca18292cc1f8,f4e31360-9d31-4064-8cfb-efd7f3e072b7,c099d336-1152-4d57-b121-f325d8acb0e4,00169207-9da7-44c4-a67b-a925e77056ff,bdc3344d-4e43-45b9-807c-f7526199e4ef,ddc3e8df-b368-40f0-bb6a-9d9f4f568cd4,00442956-c080-4b2e-b9c7-647973969c86,161aff42-2c7f-4d9c-8a24-97a606bf3df8,c9047f0b-3765-4793-b6f8-e05eaadcad3b}"
export SCENE_LIMIT=0
export SIM_STEPS="${SIM_STEPS:-199}"
export NAVSIM_2HZ_HISTORY=1
export CHALLENGE_COMPAT_MODE=1
export SERVICE_STARTUP_TIMEOUT_SEC="${SERVICE_STARTUP_TIMEOUT_SEC:-900}"

# Public challenge scoring renders no video and parses no debug info; both are
# forced off by CHALLENGE_COMPAT_MODE, which also disables host source mounts.
export RENDER_VIDEO="${RENDER_VIDEO:-false}"
export GENERATE_COMBINED_VIDEO="${GENERATE_COMBINED_VIDEO:-false}"

if [[ -z "${RUN_DIR:-}" ]]; then
    RUN_ID="$(date +%Y%m%d_%H%M%S)"
    export RUN_DIR="$ROOT/runs/e2e_challenge_drivesuprim_challenge/axe_nurec_planonly_ep09_10clips_${RUN_ID}"
fi

actual_checkpoint_sha256="$(docker image inspect "$IMAGE" --format '{{index .Config.Labels "org.alpasim.checkpoint.sha256"}}' 2>/dev/null || true)"
[[ "$actual_checkpoint_sha256" == "$EXPECTED_CHECKPOINT_SHA256" ]] || {
    echo "ERROR: $IMAGE is not the ep09 checkpoint image" >&2
    echo "  expected: $EXPECTED_CHECKPOINT_SHA256" >&2
    echo "  actual:   ${actual_checkpoint_sha256:-missing image/label}" >&2
    exit 2
}

# The simulator image the wizard resolves (alpasim-base:<repo-version>) must be
# the rebuild from the current checkout, not the stale July image.
base_tag="alpasim-base:$(grep -m1 '^version' "$ROOT/pyproject.toml" | cut -d'"' -f2)"
expected_base="${EXPECTED_BASE_IMAGE:-alpasim-base:0.89.0-f012862-casadi372}"
base_id="$(docker image inspect "$base_tag" --format '{{.Id}}' 2>/dev/null || true)"
expected_base_id="$(docker image inspect "$expected_base" --format '{{.Id}}' 2>/dev/null || true)"
[[ -n "$base_id" && "$base_id" == "$expected_base_id" ]] || {
    echo "ERROR: $base_tag is not the current-code simulator image" >&2
    echo "  $base_tag  -> ${base_id:-missing}" >&2
    echo "  $expected_base -> ${expected_base_id:-missing}" >&2
    echo "  rebuild with: docker build -t $expected_base -t $base_tag $ROOT" >&2
    exit 2
}

exec "$SCRIPT_DIR/run_axe_nurec_vits_smoke_eval.sh"
