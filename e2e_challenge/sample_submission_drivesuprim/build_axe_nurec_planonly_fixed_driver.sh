#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="${PACKAGE_ROOT:-$SCRIPT_DIR/assets/drivesuprim/axe_nurec_planonly_fixed_vits_eval}"

export PACKAGE_ROOT
export CHECKPOINT="${CHECKPOINT:-$PACKAGE_ROOT/checkpoint/epoch=29-step=43590.ckpt}"
export CONFIG="${CONFIG:-$PACKAGE_ROOT/axe_nurec_vits_config.json}"
export IMAGE="${IMAGE:-alpasim-e2e-drivesuprim-axe-nurec-planonly-fixed-vits-driver:epoch29-step43590-h100}"

exec "$SCRIPT_DIR/build_axe_nurec_vits_driver.sh"
