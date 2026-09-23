#!/usr/bin/env bash
# Build the driver image for good_planonly_from_stage1ep4_ep09 (epoch 9, step 4840).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="${PACKAGE_ROOT:-$SCRIPT_DIR/assets/drivesuprim/axe_nurec_planonly_ep09_eval}"

export PACKAGE_ROOT
export CHECKPOINT="${CHECKPOINT:-$PACKAGE_ROOT/checkpoint/good_planonly_from_stage1ep4_ep09.ckpt}"
export CONFIG="${CONFIG:-$PACKAGE_ROOT/axe_nurec_vits_config.json}"
export IMAGE="${IMAGE:-alpasim-e2e-drivesuprim-axe-nurec-planonly-ep09-driver:ep09-step4840-h100}"

exec "$SCRIPT_DIR/build_axe_nurec_vits_driver.sh"
