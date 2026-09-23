#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PACKAGE_ROOT="${PACKAGE_ROOT:-$ROOT/e2e_challenge/sample_submission_drivesuprim/assets/drivesuprim/axe_nurec_vits_eval}"
IMAGE="${IMAGE:-alpasim-e2e-axe-nurec-vits-driver:latest}"
VOCAB="$PACKAGE_ROOT/assets/nurec/vocab/nurec_train_kmeans_4096x40x3.npy"
CONFIG="${CONFIG:-$PACKAGE_ROOT/axe_nurec_vits_config.json}"
MSDA_PATCH="$ROOT/e2e_challenge/sample_submission_drivesuprim/patches/axe_nurec_vits_hopper_msda.patch"

cd "$ROOT"

if [[ -n "${CHECKPOINT:-}" ]]; then
    checkpoint="$CHECKPOINT"
else
    mapfile -t checkpoints < <(find "$PACKAGE_ROOT/checkpoint" -maxdepth 1 -type f -name '*.ckpt' -print | sort)
    [[ "${#checkpoints[@]}" -eq 1 ]] || {
        echo "ERROR: expected exactly one checkpoint; set CHECKPOINT explicitly" >&2
        exit 2
    }
    checkpoint="${checkpoints[0]}"
fi

for path in \
    "$PACKAGE_ROOT/README.md" \
    "$checkpoint" \
    "$VOCAB" \
    "$CONFIG" \
    "$MSDA_PATCH" \
    "$ROOT/e2e_challenge/sample_submission_drivesuprim/requirements.txt" \
    "$ROOT/e2e_challenge/sample_submission_drivesuprim/scripts/patch_grpc_py310.py"
do
    [[ -f "$path" ]] || { echo "ERROR: missing package file: $path" >&2; exit 2; }
done
[[ -d "$PACKAGE_ROOT/navsim" ]] || { echo "ERROR: missing navsim source" >&2; exit 2; }
[[ -d "$ROOT/src/grpc" ]] || { echo "ERROR: missing AlpaSim gRPC source" >&2; exit 2; }
[[ -d "$ROOT/e2e_challenge/sample_submission_vavam/vavam_challenge" ]] || {
    echo "ERROR: missing VAVAM challenge plumbing" >&2
    exit 2
}

context="$(mktemp -d "$PACKAGE_ROOT/.alpasim-build.XXXXXX")"
trap 'rm -rf "$context"' EXIT

cp -a "$PACKAGE_ROOT/navsim" "$context/navsim"
patch --batch --forward --directory="$context/navsim" --strip=1 < "$MSDA_PATCH"
cp -a "$ROOT/src/grpc" "$context/alpasim_grpc"
cp -a "$ROOT/e2e_challenge/sample_submission_drivesuprim/drivesuprim_challenge" "$context/drivesuprim_challenge"
cp -a "$ROOT/e2e_challenge/sample_submission_vavam/vavam_challenge" "$context/vavam_challenge"
cp "$ROOT/e2e_challenge/sample_submission_drivesuprim/requirements.txt" "$context/requirements.txt"
cp "$ROOT/e2e_challenge/sample_submission_drivesuprim/scripts/patch_grpc_py310.py" "$context/patch_grpc_py310.py"
cp --reflink=auto "$checkpoint" "$context/model.ckpt"
cp --reflink=auto "$VOCAB" "$context/nurec_train_kmeans_4096x40x3.npy"
cp "$CONFIG" "$context/axe_nurec_vits_config.json"
cp "$PACKAGE_ROOT/README.md" "$context/PACKAGE_README.md"

checkpoint_sha256="$(sha256sum "$checkpoint" | awk '{print $1}')"
vocab_sha256="$(sha256sum "$VOCAB" | awk '{print $1}')"

docker build \
    --file e2e_challenge/sample_submission_drivesuprim/Dockerfile.axe_nurec_vits \
    --label "org.alpasim.model=axe-nurec-vits-planonly" \
    --label "org.alpasim.checkpoint.sha256=$checkpoint_sha256" \
    --label "org.alpasim.vocab.sha256=$vocab_sha256" \
    --tag "$IMAGE" \
    "$context"

echo "Built: $IMAGE"
echo "Checkpoint: $checkpoint"
echo "Contract: K377 512x256, L0/F0/R0, 2Hz history, route=on, NuRec vocab"
