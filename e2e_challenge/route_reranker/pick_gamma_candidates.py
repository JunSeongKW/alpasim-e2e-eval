"""Rank the swept gammas by their EXTRAPOLATED effect on the 441-clip set.

The 38-clip sweep is two-thirds corridor failures; the 441 set is 9% corridor
failures. A gamma that wins on the sweep wins by rescuing failures, and that
gain is diluted eleven-fold on the full set while the cost it imposes on the
clips that already worked is not. So the ranking that matters weights the two
groups the way the 441 set does, not the way the sweep does.

Prints the gammas best-first, one per line, so a shell loop can take the top N.
"""

import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent

fails = set(HERE.joinpath("clips_turnfail.txt").read_text().split())
ctrls = set(HERE.joinpath("clips_control.txt").read_text().split())


def load(tag):
    f = ROOT / f"runs/gen-{tag}-cache-centre-max/aggregate/results-summary.json"
    if not f.exists():
        return None
    return {r["clipgt_id"]: r for r in json.load(open(f))["rollouts"]
            if r.get("score") is not None}


base = load("g0")
if base is None:
    sys.exit("gamma=0 arm missing")

b441 = json.load(open(ROOT / "runs/leaderboard-merged-route-ep30/aggregate/results-summary.json"))
R441 = [r for r in b441["rollouts"] if r.get("score") is not None]
w = sum(1 for r in R441 if (r["metrics"].get("left_corridor_laterally") or 0) == 1) / len(R441)

ranked = []
for d in sorted((ROOT / "runs").glob("gen-g*-cache-centre-max")):
    tag = d.name[len("gen-"):-len("-cache-centre-max")]
    g = float(tag[1:].replace("p", "."))
    if g == 0:
        continue
    A = load(tag)
    if A is None:
        continue
    mf = statistics.fmean(A[c]["score"] - base[c]["score"] for c in fails if c in A)
    mc = statistics.fmean(A[c]["score"] - base[c]["score"] for c in ctrls if c in A)
    ranked.append((w * mf + (1 - w) * mc, g))

ranked.sort(reverse=True)
if "--table" in sys.argv:
    for est, g in ranked:
        print(f"gamma={g:<6} 441추정Δ={est:+.4f}")
else:
    for _, g in ranked:
        print(g)
