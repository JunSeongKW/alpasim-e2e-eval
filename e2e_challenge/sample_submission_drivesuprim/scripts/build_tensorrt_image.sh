#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-alpasim-e2e-drivesuprim-vov-driver:tensorrt}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

docker build -t "${IMAGE}" \
  -f "${REPO_ROOT}/e2e_challenge/sample_submission_drivesuprim/Dockerfile.tensorrt" \
  "${REPO_ROOT}"
