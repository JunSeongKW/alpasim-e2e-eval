#!/usr/bin/env bash
# How much does the local PCS ranking depend on WHICH subjects are in the fit?
#
# The question this answers: our local leaderboard puts aug-cont-ep24 fifth
# although its scene score is the highest of all. Is that a property of the
# policy, or of our subject pool? ZOIB is fitted across subjects per scene, so
# scene difficulty and discrimination are estimated from whoever is in the fit.
# Our pool is almost all variants of one model family, which the competition's
# pool is not. If the ordering moves when the pool moves, the local rank cannot
# be read as a prediction of the official one.
#
# Each fit below holds the two subjects of interest fixed and changes only the
# company they keep. Scene scores and at-fault distances are pool-invariant, so
# any movement in PCS is pool effect alone.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/runs/pool-sensitivity"
mkdir -p "$OUT"
EV="$ROOT/e2e_challenge/local_evaluation/evaluate.py"

R_V9="$ROOT/runs/leaderboard-merged-route-ep30"
R_C24="$ROOT/runs/leaderboard-aug-cont-ep24"

fit() {
    local name="$1"; shift
    echo "=== $name ==="
    OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 UV_OFFLINE=1 \
    uv run --extra local-evaluation python "$EV" --track pai \
        --run "axe-v9=$R_V9" --run "aug-cont-ep24=$R_C24" "$@" \
        --output-dir "$OUT/$name" --device cpu > "$OUT/$name.log" 2>&1
    echo "  exit=$?"
}

# A: nothing but the two subjects and the eight published references.
fit poolA

# B: add three subjects that fail differently from our family.
fit poolB \
    --run "flatvits-ep25=$ROOT/runs/leaderboard-flatvits-ep25" \
    --run "axe-v4=$ROOT/runs/leaderboard-axe-v4" \
    --run "stage3_new_ep30=$ROOT/runs/leaderboard-stage3-ep30"

# C: add only near-neighbours from the same family -- the pool we actually have.
fit poolC \
    --run "aug-ep19=$ROOT/runs/leaderboard-aug-ep19" \
    --run "aug-ep29-final=$ROOT/runs/leaderboard-aug-ep29-final" \
    --run "disjoint-ep04=$ROOT/runs/leaderboard-disjoint-ep04" \
    --run "disjoint-ep29=$ROOT/runs/leaderboard-disjoint-ep29"

echo "=== done ==="
