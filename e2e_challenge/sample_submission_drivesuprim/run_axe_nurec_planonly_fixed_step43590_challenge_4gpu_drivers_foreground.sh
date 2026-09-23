#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export IMAGE="${IMAGE:-alpasim-e2e-drivesuprim-axe-nurec-planonly-fixed-vits-driver:epoch29-step43590-h100}"
export EXPECTED_CHECKPOINT_SHA256="${EXPECTED_CHECKPOINT_SHA256:-b4bc194d81795f774ccaf4da30b403f0b233a23b03d5b4bf369c2a004faf0d33}"
export GPU_INDICES_CSV="${GPU_INDICES_CSV:-0,1,2,3}"
export DRIVER_PORTS_CSV="${DRIVER_PORTS_CSV:-6860,6861,6862,6863}"
export NAVSIM_2HZ_HISTORY=1
export CONTAINER_PREFIX="${CONTAINER_PREFIX:-drivesuprim-axe-nurec-planonly-fixed-step43590-challenge}"

DRIVER_LOG_FILE="${DRIVER_LOG_FILE:-$ROOT/runs/driver_logs/axe_nurec_planonly_fixed_step43590_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$(dirname "$DRIVER_LOG_FILE")"

echo "Driver log: $DRIVER_LOG_FILE"
exec > >(tee -a "$DRIVER_LOG_FILE") 2>&1

exec "$SCRIPT_DIR/run_axe_nurec_vits_epoch29_4gpu_drivers_foreground.sh"
