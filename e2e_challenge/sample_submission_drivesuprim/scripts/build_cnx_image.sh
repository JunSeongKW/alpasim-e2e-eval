#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-alpasim-e2e-drivesuprim-v2-cnx-deploy:latest}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
ASSET_DIR="${REPO_ROOT}/e2e_challenge/sample_submission_drivesuprim/assets/drivesuprim"

test -f "${ASSET_DIR}/drivesuprim_v2_cnx_deploy_random.ckpt"
test -f "${ASSET_DIR}/test_4096_kmeans.npy"
test -f "${ASSET_DIR}/source_jh/agents/backbones/bevformer/bevformer_backbone.py"

docker build -t "${IMAGE}" \
  -f "${REPO_ROOT}/e2e_challenge/sample_submission_drivesuprim/Dockerfile.cnx" \
  "${REPO_ROOT}"
