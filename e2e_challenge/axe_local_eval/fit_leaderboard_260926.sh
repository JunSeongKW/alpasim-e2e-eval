#!/usr/bin/env bash
# Local Drive-IRT/ZOIB leaderboard, 19 local subjects + the 8 reference ones.
#
# This is the 260925c fit with the three runs finished since then added:
# aug-ep29-final, disjoint-ep04, disjoint-ep29. Every subject has to be fitted
# together, not appended: ZOIB is fitted across subjects per scene, so PCS
# values move when the subject set changes and only a single fit is comparable.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
OUT="$ROOT/runs/leaderboard-260926"
LOG="$ROOT/e2e_challenge/axe_local_eval/leaderboard_260926.log"
OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 \
UV_OFFLINE=1 uv run --extra local-evaluation python "$ROOT/e2e_challenge/local_evaluation/evaluate.py" \
    --track pai \
    --run axe-v9="$ROOT/runs/leaderboard-merged-route-ep30" \
    --run axe-v9+rerank-g0.02="$ROOT/runs/rr441-g0p02-cache-centre-max" \
    --run axe-v9+rerank-g0.1="$ROOT/runs/rr441-g0p1-cache-centre-max" \
    --run aug-ep04+rerank="$ROOT/runs/rr441-g0p02-augep04-cache-centre-max" \
    --run aug-ep19="$ROOT/runs/leaderboard-aug-ep19" \
    --run aug-ep29-final="$ROOT/runs/leaderboard-aug-ep29-final" \
    --run disjoint-ep04="$ROOT/runs/leaderboard-disjoint-ep04" \
    --run disjoint-ep29="$ROOT/runs/leaderboard-disjoint-ep29" \
    --run merged-ep29-bestmpc="$ROOT/runs/leaderboard-merged-ep29-bestmpc" \
    --run epzero-ep29-tuned="$ROOT/runs/leaderboard-epzero-ep29" \
    --run epzero-ep29-base="$ROOT/runs/leaderboard-epzero-ep29-basegains" \
    --run ep29-step30330="$ROOT/runs/leaderboard-260923-ep29-step30330" \
    --run stage3_new_ep30="$ROOT/runs/leaderboard-stage3-ep30" \
    --run flatvits-ep25="$ROOT/runs/leaderboard-flatvits-ep25" \
    --run routecache-hardgate="$ROOT/runs/routecache-val441" \
    --run routecache-center="$ROOT/runs/routecache-val441-center1s" \
    --run axe-ep24-egobox="$ROOT/runs/axe-ep24-curatedval-egobox" \
    --run axe-v5="$ROOT/runs/leaderboard-axe-v5" \
    --run axe-v4="$ROOT/runs/leaderboard-axe-v4" \
    --output-dir "$OUT" --device cpu > "$LOG" 2>&1
echo "exit=$? -> $OUT"
