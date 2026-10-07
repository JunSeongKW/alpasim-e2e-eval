#!/usr/bin/env bash
# CPU-only, offline build; never takes a GPU from a running evaluation.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
BUNDLE="${BUNDLE:-$ROOT/../models/stage3_5cam_ep05_20261007}"
BASE='696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe@sha256:85134c1ea9f09d140be610ca063c50ec60e99980c392e5bf81fe6563af503765'
IMAGE='alpasim-e2e-drivesuprim-stage3:5cam-ep05-20261007'
SHA='79c625f3a29c49da6b9605b1062de5e8e6ec5da8391362b25507e35725bb8b8e'
cd "$BUNDLE"
sha256sum -c --quiet SHA256SUMS
[[ "$(sha256sum AXEv1.0-NuRec/checkpoints/stage3_5cam_ep05.ckpt | cut -d' ' -f1)" == "$SHA" ]]
docker image inspect "$BASE" >/dev/null
mkdir -p "$ROOT/.cache"
STAGE="$(mktemp -d "$ROOT/.cache/5cam-ep05-build.XXXXXX")"
mkdir -p "$STAGE/app/assets/drivesuprim" "$STAGE/app/5cam_eval"
cp -a "$BUNDLE/AXEv1.0-NuRec/navsim" "$BUNDLE/AXEv1.0-NuRec/nurec_pf" "$STAGE/app/"
cp -a "$HERE/drivesuprim_challenge" "$STAGE/app/"
cp "$BUNDLE/AXEv1.0-NuRec/checkpoints/stage3_5cam_ep05.ckpt" "$STAGE/app/assets/drivesuprim/"
cp "$BUNDLE/AXEv1.0-NuRec/assets/nurec/vocab/nurec_train_kmeans_4096x40x3.npy" "$STAGE/app/assets/drivesuprim/"
cp "$HERE/smoke_cpu.py" "$STAGE/app/5cam_eval/"
docker run --rm --network none --entrypoint python \
    -v "$BUNDLE/AXEv1.0-NuRec:/bundle:ro" -v "$HERE:/tools:ro" \
    -v "$STAGE/app/assets/drivesuprim:/output" "$BASE" \
    /tools/compose_config.py --source /bundle --output /output/stage3_5cam_config.json
cmp "$HERE/stage3_5cam_config.json" "$STAGE/app/assets/drivesuprim/stage3_5cam_config.json"
SOURCE_SHA="$(sha256sum "$BUNDLE/SHA256SUMS" | cut -d' ' -f1)"
cp "$HERE/Dockerfile" "$STAGE/"
docker build --network none --build-arg "CHECKPOINT_SHA256=$SHA" \
    --build-arg "SOURCE_MANIFEST_SHA256=$SOURCE_SHA" -t "$IMAGE" "$STAGE"
cd "$ROOT"
OUT="$ROOT/runs/prepare-stage3-5cam-ep05"
mkdir -p "$OUT"
printf '%s\n' "$STAGE" > "$OUT/build-context.txt"
"$ROOT/.venv/bin/python" "$HERE/audit_scenes.py" --output "$OUT"
docker run --rm --network none --cpus 8 --memory 16g --entrypoint python \
    -v "$OUT:/validation" "$IMAGE" /app/5cam_eval/smoke_cpu.py \
    --fixture /validation/camera_fixture.json --report /validation/cpu_validation.json \
    > "$OUT/cpu-validation.log" 2>&1 || { tail -50 "$OUT/cpu-validation.log" >&2; exit 1; }
docker image inspect "$IMAGE" --format '{{.Id}}' > "$OUT/image-id.txt"
printf 'PASS: %s\nValidation: %s\n' "$IMAGE" "$OUT/cpu_validation.json"
