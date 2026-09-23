#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAMPLE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DST="${SAMPLE_DIR}/assets/drivesuprim"
SRC="${1:-${DRIVESUPRIM_REPO:-}}"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -n "${SRC}" ]] || fail "usage: bash $0 /path/to/DriveSuprim"
[[ -d "${SRC}/navsim" ]] || fail "DriveSuprim navsim source not found under ${SRC}"

mkdir -p "${DST}/source"
cp -a "${SRC}/navsim" "${DST}/source/"
cp --reflink=auto "${SRC}/exp_v2/model_ckpt/drivesuprim_vit.ckpt" "${DST}/drivesuprim_vit.ckpt"
cp --reflink=auto "${SRC}/exp_v2/models/da_vitl16.pth" "${DST}/da_vitl16.pth"
cp --reflink=auto "${SRC}/traj_final/test_8192_kmeans.npy" "${DST}/test_8192_kmeans.npy"

echo "DriveSuprim source and assets staged in ${DST}"
