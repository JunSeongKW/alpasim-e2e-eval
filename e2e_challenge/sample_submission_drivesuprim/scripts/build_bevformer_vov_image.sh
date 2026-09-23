#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-alpasim-e2e-drivesuprim-bevformer-vov-driver:jh-stage3-ep4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ASSET_DIR="${REPO_ROOT}/e2e_challenge/sample_submission_drivesuprim/assets/drivesuprim"

test -f "${ASSET_DIR}/drivesuprim_vov.ckpt"
test -f "${ASSET_DIR}/dd3d_det_final.pth"
test -f "${ASSET_DIR}/test_8192_kmeans.npy"
test -f "${ASSET_DIR}/source_bevformer/agents/backbones/bevformer/bevformer_backbone.py"

docker build -t "${IMAGE}" \
  -f "${REPO_ROOT}/e2e_challenge/sample_submission_drivesuprim/Dockerfile.bevformer-vov" \
  "${REPO_ROOT}"
