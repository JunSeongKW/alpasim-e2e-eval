#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

IMAGE="${IMAGE:-alpasim-e2e-drivesuprim-cnx-stage3-driver:epoch04-h100}"
ASSET_DIR="e2e_challenge/sample_submission_drivesuprim/assets/drivesuprim"
CHECKPOINT="$ASSET_DIR/drivesuprim_v2_convnextv2_stage3_epoch04.ckpt"
VOCAB="$ASSET_DIR/test_4096_kmeans.npy"
EXPECTED_CHECKPOINT_SHA256="9a8250144083af1b37e906a25584288f78bc14b43a2e5fb5ffd4fd03457ce662"
EXPECTED_VOCAB_SHA256="6f52b5fac9ff5debe5bae9dc82c3298b09ad0530aca63ba4285f7092301395cf"

[[ -d "$ASSET_DIR/source_cnx_stage3/navsim" ]] || { echo "ERROR: missing ConvNeXt source" >&2; exit 2; }
[[ "$(sha256sum "$CHECKPOINT" | awk '{print $1}')" == "$EXPECTED_CHECKPOINT_SHA256" ]] || { echo "ERROR: checkpoint SHA256 mismatch" >&2; exit 2; }
[[ "$(sha256sum "$VOCAB" | awk '{print $1}')" == "$EXPECTED_VOCAB_SHA256" ]] || { echo "ERROR: vocab SHA256 mismatch" >&2; exit 2; }

echo "Building $IMAGE"
echo "  checkpoint: $EXPECTED_CHECKPOINT_SHA256"
echo "  vocab:      $EXPECTED_VOCAB_SHA256"
echo "  input:      NAVSIM K=(412,366.2222,256,132.7407), 512x256"
echo "  history:    t-1.0s, t-0.5s, t"
echo "  H100 MSDA:  pure-PyTorch fallback"

docker build --pull=false \
    -f e2e_challenge/sample_submission_drivesuprim/Dockerfile.cnx_stage3 \
    -t "$IMAGE" \
    .

docker image inspect "$IMAGE" --format \
    'image={{.Id}} checkpoint={{index .Config.Labels "org.alpasim.checkpoint.sha256"}} vocab={{index .Config.Labels "org.alpasim.vocab.sha256"}}'
