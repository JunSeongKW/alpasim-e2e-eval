#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export IMAGE="${IMAGE:-alpasim-e2e-drivesuprim-axe-nurec-pdms-vits-driver:epoch29-step43590-h100}"
export EXPECTED_CHECKPOINT_SHA256="${EXPECTED_CHECKPOINT_SHA256:-8aa875bcbd4f460791bd225d9c4ce57dd48e6affde6f9d4019a2afb5efd4e8f4}"
export GPU_INDICES_CSV="${GPU_INDICES_CSV:-0,1,2,3}"
export DRIVER_PORTS_CSV="${DRIVER_PORTS_CSV:-6860,6861,6862,6863}"
export NAVSIM_2HZ_HISTORY=1
export CONTAINER_PREFIX="${CONTAINER_PREFIX:-drivesuprim-axe-nurec-pdms-vits-step43590-challenge}"

exec "${SCRIPT_DIR}/run_axe_nurec_vits_epoch29_4gpu_drivers_foreground.sh"
