#!/usr/bin/env bash
# Start four driver replicas for the ep09 checkpoint (GPUs 4-7; 0-3 are in use).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export IMAGE="${IMAGE:-alpasim-e2e-drivesuprim-axe-nurec-planonly-ep09-driver:ep09-step4840-h100}"
export EXPECTED_CHECKPOINT_SHA256="${EXPECTED_CHECKPOINT_SHA256:-90ff4189f3e464ca681a1d17e7615b0d04b2de768429d21eb5aa85007fe2b76d}"
export GPU_INDICES_CSV="${GPU_INDICES_CSV:-4,5,6,7}"
export DRIVER_PORTS_CSV="${DRIVER_PORTS_CSV:-6860,6861,6862,6863}"
export NAVSIM_2HZ_HISTORY=1
export CONTAINER_PREFIX="${CONTAINER_PREFIX:-drivesuprim-axe-nurec-planonly-ep09-challenge}"

DRIVER_LOG_FILE="${DRIVER_LOG_FILE:-$ROOT/runs/driver_logs/axe_nurec_planonly_ep09_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "$DRIVER_LOG_FILE")"

echo "Driver log: $DRIVER_LOG_FILE"
exec > >(tee -a "$DRIVER_LOG_FILE") 2>&1

exec "$SCRIPT_DIR/run_axe_nurec_vits_epoch29_4gpu_drivers_foreground.sh"
