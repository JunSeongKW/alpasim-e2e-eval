#!/usr/bin/env bash
# Assemble the axe-v10 build context and verify it before docker touches it.
#
#   e2e_challenge/axe_local_eval/prepare_axe_v10_context.sh [CONTEXT_DIR]
#
# What goes in, and why each piece:
#   checkpoint.ckpt   stage3_aug_ep29_final, the weights the gamma ladder scored
#   stage/app/...     the seven reranker paths, laid out so one COPY places them
#   Dockerfile        the two-layer variant
#
# The hash check is the point of doing this in a script. A submission is
# irreversible, and the one previous failure in this project's history came from
# an image built off the wrong source -- so the checkpoint is verified here,
# before a two-hour build, rather than discovered afterwards in preflight.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
R="$ROOT/e2e_challenge/route_reranker"
CTX="${1:-/tmp/axe_v10_ctx_fast}"
CKPT="$ROOT/../models/stage3_aug_ep29_final.ckpt"
WANT_SHA=29c6fca63c4d4bd561fb90101ebbf39f7cb23b87a09251b1db3a766f05ffdef8

die() { echo "FAIL: $*" >&2; exit 1; }

[[ -f "$CKPT" ]] || die "checkpoint not found: $CKPT"
echo "hashing the checkpoint (763 MB)..."
got="$(ionice -c 3 sha256sum "$CKPT" | awk '{print $1}')"
[[ "$got" == "$WANT_SHA" ]] || die "checkpoint sha256 is $got, expected $WANT_SHA"
echo "  checkpoint sha256 OK  ${WANT_SHA:0:16}..."

rm -rf "$CTX"
mkdir -p "$CTX/stage/app/navsim/agents/drivesuprim"
cp "$CKPT" "$CTX/checkpoint.ckpt"

# The checkpoint also has to reach its runtime path; keeping it inside stage/
# would make the single COPY carry 763 MB, so it stays a separate small layer in
# the Dockerfile only if needed. Here it goes into stage/ as well, which is
# simpler and costs one commit either way.
mkdir -p "$CTX/stage/app/assets/drivesuprim"
cp "$CKPT" "$CTX/stage/app/assets/drivesuprim/axe_nurec_vits.ckpt"
rm -f "$CTX/checkpoint.ckpt"

cp -r "$R/drivesuprim_challenge" "$CTX/stage/app/drivesuprim_challenge"
rm -rf "$CTX/stage/app/drivesuprim_challenge/__pycache__"
for f in drivesuprim_model.py drivesuprim_config.py drivesuprim_agent.py \
         route_reranker.py route_inputs.py; do
    [[ -f "$R/$f" ]] || die "missing $R/$f"
    cp "$R/$f" "$CTX/stage/app/navsim/agents/drivesuprim/$f"
done
[[ -f "$R/rerank_variants.py" ]] || die "missing $R/rerank_variants.py"
cp "$R/rerank_variants.py" "$CTX/stage/app/rerank_variants.py"
cp "$ROOT/e2e_challenge/axe_local_eval/Dockerfile.axe_v10_fast" "$CTX/Dockerfile"

echo
echo "context: $CTX"
find "$CTX/stage" -type f | sed "s|$CTX/stage||" | sort | sed 's/^/  /'
echo
echo "python syntax check on every copied module:"
for f in $(find "$CTX/stage" -name '*.py'); do
    "$ROOT/.venv/bin/python" -m py_compile "$f" 2>/dev/null \
        && echo "  OK   ${f#$CTX/stage}" || die "syntax error in ${f#$CTX/stage}"
done
find "$CTX/stage" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true

cat <<EOF

build with:
  cd $CTX && docker build -t 696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v10 \\
    --build-arg CHECKPOINT_SHA256=$WANT_SHA \\
    --build-arg CHECKPOINT_SOURCE=$CKPT .
EOF
