#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-alpasim-e2e-drivesuprim-driver:latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ASSET_DIR="${REPO_ROOT}/e2e_challenge/sample_submission_drivesuprim/assets/drivesuprim"

check_asset() {
  local path="$1"
  local min_bytes="$2"
  [[ -f "${path}" ]] || { echo "Missing required asset: ${path}" >&2; exit 1; }
  local size
  size="$(wc -c < "${path}")"
  (( size >= min_bytes )) || {
    echo "Asset is unexpectedly small: ${path} (${size} bytes)" >&2
    exit 1
  }
}

check_asset "${ASSET_DIR}/drivesuprim_vit.ckpt" 4000000000
check_asset "${ASSET_DIR}/da_vitl16.pth" 1000000000
check_asset "${ASSET_DIR}/test_8192_kmeans.npy" 1000000
[[ -d "${ASSET_DIR}/source/navsim" ]] || {
  echo "Missing staged DriveSuprim source. Run scripts/prepare_assets.sh first." >&2
  exit 1
}

docker build -t "${IMAGE}" \
  -f "${REPO_ROOT}/e2e_challenge/sample_submission_drivesuprim/Dockerfile" \
  "${REPO_ROOT}"
