#!/usr/bin/env bash
# Verify a submission candidate before any quota is spent.
#
# Every check here is local and free. A submission is irreversible and the quota
# is small, so anything that can be checked without calling the API is checked
# here first. The API-side checks that remain (terms acceptance, image manifest
# resolution) do not consume quota either, which is why they are not duplicated.
#
# Usage:
#   e2e_challenge/axe_local_eval/preflight_submit.sh [IMAGE] [GAINS_JSON]
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

IMAGE="${1:-alpasim-e2e-drivesuprim-stage3:merged-route-ep30}"
GAINS="${2:-e2e_challenge/axe_local_eval/controller_gains_axe_v9.json}"

# sha256 of models/stage3_merged_route_ep30.ckpt, the weights scored over the
# 441-clip curated_val run in runs/leaderboard-merged-route-ep30. Override with
# EXPECTED_CKPT_SHA256=... to preflight a different candidate.
EXPECTED_CKPT_SHA256="${EXPECTED_CKPT_SHA256:-364c801e397e9d6e36ea1f9027dc4ca804ad2e429bb588cf4e84185ca851de3d}"
# Flags the 441-clip validation depended on. The evaluator sets no DRIVESUPRIM_*
# variable, so each of these must come from the image itself.
REQUIRED_ENV=(
    "DRIVESUPRIM_EGO_CENTER_OFFSET=1"
    "DRIVESUPRIM_EGO_FOOTPRINT_FROM_API=1"
    "DRIVESUPRIM_REQUIRE_POLICY_FOR_VERSION=1"
    "DRIVESUPRIM_REQUIRE_EXACT_CHECKPOINT=1"
    "DRIVESUPRIM_NAVSIM_2HZ_HISTORY=1"
    "DRIVESUPRIM_MAX_BATCH_SIZE=1"
    "DRIVESUPRIM_DISABLE_INFERENCE=0"
)
MAX_IMAGE_BYTES=$((40 * 1024 * 1024 * 1024))

fail=0
ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$*"; }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$*"; fail=1; }
warn() { printf '  \033[33mWARN\033[0m  %s\n' "$*"; }

echo "image : $IMAGE"
echo "gains : $GAINS"
echo

echo "[1] image exists"
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    ok "$IMAGE resolves locally"
else
    bad "$IMAGE not found locally"
    exit 1
fi

echo "[2] baked environment"
env_dump="$(docker image inspect "$IMAGE" --format '{{range .Config.Env}}{{println .}}{{end}}')"
for want in "${REQUIRED_ENV[@]}"; do
    if grep -qxF "$want" <<<"$env_dump"; then
        ok "$want"
    else
        got="$(grep "^${want%%=*}=" <<<"$env_dump" || echo '<unset>')"
        bad "$want   (image has: $got)"
    fi
done

echo "[3] checkpoint identity"
actual="$(docker run --rm --entrypoint sha256sum "$IMAGE" \
    /app/assets/drivesuprim/axe_nurec_vits.ckpt 2>/dev/null | awk '{print $1}')"
if [[ "$actual" == "$EXPECTED_CKPT_SHA256" ]]; then
    ok "checkpoint sha256 matches ${EXPECTED_CKPT_SHA256:0:16}..."
else
    bad "checkpoint sha256 mismatch: ${actual:-<none>}"
fi

echo "[4] size limit"
bytes="$(docker image inspect "$IMAGE" --format '{{.Size}}')"
if (( bytes <= MAX_IMAGE_BYTES )); then
    ok "$(numfmt --to=iec "$bytes") <= 40 GiB"
else
    bad "$(numfmt --to=iec "$bytes") exceeds the 40 GiB limit"
fi

echo "[5] tag is not 'latest'"
if [[ "${IMAGE##*:}" == latest ]]; then
    bad "the 'latest' tag is rejected by the submission CLI"
else
    ok "tag '${IMAGE##*:}'"
fi

echo "[6] entrypoint"
cmd="$(docker image inspect "$IMAGE" --format '{{json .Config.Cmd}}')"
if [[ "$cmd" == *drivesuprim_challenge.driver* ]]; then
    ok "CMD $cmd"
else
    bad "unexpected CMD $cmd"
fi

echo "[7] controller gains file"
if [[ -f "$GAINS" ]]; then
    if out="$(uv run python - "$GAINS" <<'PY' 2>&1
import pathlib, sys
sys.path.insert(0, "e2e_challenge/competitor_cli")
from alpasim_challenge import load_controller_gains
g = load_controller_gains(pathlib.Path(sys.argv[1]))
assert isinstance(g["idx_start_penalty"], int), "idx_start_penalty must be int"
print(" ".join(f"{k}={v}" for k, v in sorted(g.items())))
PY
    )"; then
        ok "validated by the real CLI validator"
        echo "        $out"
    else
        bad "rejected: $out"
    fi
else
    warn "$GAINS not found; submission would use evaluator defaults"
fi

echo "[8] driver starts with the official environment only"
probe=axe-preflight-probe
docker rm -f "$probe" >/dev/null 2>&1
docker run -d --name "$probe" --init --gpus "device=${PREFLIGHT_GPU:-4}" \
    --read-only --cap-drop ALL --security-opt no-new-privileges:true \
    --tmpfs /tmp:rw,nosuid,nodev,size=2g \
    --tmpfs /run:rw,nosuid,nodev,size=64m \
    -e ALPASIM_DRIVER_HOST=0.0.0.0 -e ALPASIM_DRIVER_PORT=6789 \
    -e ALPASIM_CONTESTANT_REPLICA_INDEX=0 -e ALPASIM_CONTESTANT_REPLICAS=16 \
    "$IMAGE" >/dev/null 2>&1
deadline=$((SECONDS + ${PREFLIGHT_TIMEOUT:-600}))
ready=0
while (( SECONDS < deadline )); do
    if docker logs "$probe" 2>&1 | grep -q "policy load complete"; then ready=1; break; fi
    docker inspect -f '{{.State.Running}}' "$probe" 2>/dev/null | grep -q true || break
    sleep 5
done
if (( ready )); then
    ok "policy loaded with no DRIVESUPRIM_* injected"
    vram="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | \
        awk -v p="$(docker inspect -f '{{.State.Pid}}' "$probe")" 'index($0,p)' | head -1)"
    [[ -n "$vram" ]] && echo "        VRAM: $vram"
else
    bad "driver did not report 'policy load complete'"
    docker logs --tail 15 "$probe" 2>&1 | sed 's/^/        /'
fi
docker rm -f "$probe" >/dev/null 2>&1

echo "[9] the tag about to be submitted is this exact image"
# The 2026-09-12 submission was lost to a retag from the wrong source: the ECR
# tag was created from :ep24-egobox instead of :submit-v1, so none of the baked
# ENV was present. `docker tag` never changes an image ID, so comparing IDs
# catches that before any quota is spent. `docker manifest inspect` does not --
# it only proves some image resolves under that name.
if [[ -n "${ECR_URI:-}" ]]; then
    want="$(docker image inspect "$IMAGE" --format '{{.ID}}')"
    if got="$(docker image inspect "$ECR_URI" --format '{{.ID}}' 2>/dev/null)"; then
        if [[ "$got" == "$want" ]]; then
            ok "$ECR_URI is the same image as $IMAGE"
            echo "        $want"
        else
            bad "$ECR_URI points at a DIFFERENT image"
            echo "        expected $want"
            echo "        actual   $got"
            echo "        fix: docker tag $IMAGE $ECR_URI"
        fi
    else
        bad "$ECR_URI not tagged locally; run: docker tag $IMAGE $ECR_URI"
    fi
else
    warn "ECR_URI not set; skipping. Run again as:"
    echo "        ECR_URI=696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/<team>:<tag> $0"
fi

echo
if (( fail )); then
    echo "PREFLIGHT FAILED - do not submit"
    exit 1
fi
echo "PREFLIGHT PASSED"
echo
echo "Remaining checks are API-side and consume no quota:"
echo "  alpasim_challenge.py terms status     both you and the captain must be current"
echo "  alpasim_challenge.py limits           confirm remaining submissions"
echo "  docker manifest inspect <ecr-uri>     after push, before submit"
