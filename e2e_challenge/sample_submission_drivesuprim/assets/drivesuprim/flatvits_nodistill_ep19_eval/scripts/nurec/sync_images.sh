#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${NUREC_PREPARED_ROOT:?Set NUREC_PREPARED_ROOT to prepare_usdz.sh output}"
# The 2 Hz three-camera render is what the NuRec logs are sampled against;
# the 10 Hz path this used to default to was never produced.
NUREC_SENSOR_ROOT="${NUREC_SENSOR_ROOT:-/mnt/nfs/data/processed_dataset/alphasim/nurec2img/26.04_release_2hz_1920x1080_3cam_png}"
PYTHON_BIN="${PYTHON_BIN:-/rhome/satyam/.conda/envs/drivesuprim/bin/python}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

args=(
  --log-root "$NUREC_PREPARED_ROOT/navsim_logs/trainval"
  --image-root "$NUREC_SENSOR_ROOT"
)
if [[ "${NUREC_ALLOW_INCOMPLETE_IMAGES:-1}" == "1" ]]; then
  args+=(--allow-incomplete)
fi

"$PYTHON_BIN" -m navsim.planning.data.nurec_attach_images "${args[@]}"
