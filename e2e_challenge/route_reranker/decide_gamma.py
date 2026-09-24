"""Pick the gamma for the 441 run from the random-40 validation, paired to gamma=0.

Rule, in order:
  1. The noise floor is |mean paired delta of gamma=0 against the 441 baseline
     on these 40 clips| -- what the same code, same checkpoint, produces by
     itself. It is printed so the reader can see it.
  2. Among the candidates, take the largest mean paired delta against the
     gamma=0 ARM (not the 441 run: the arm is the reference at this exact
     configuration).
  3. If that delta does not clear the noise floor, say so loudly, but still
     return the extrapolation's leader (0.02) -- the user asked for a 441 score
     with the chosen gamma, and a null result on 441 is itself the answer.

Prints one line for the shell (the gamma) unless --explain is given.
"""

import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
HERE = pathlib.Path(__file__).resolve().parent
clips = set(HERE.joinpath("clips_random40.txt").read_text().split())


def load(path):
    f = pathlib.Path(path)
    if not f.exists():
        return None
    return {r["clipgt_id"]: r for r in json.load(open(f))["rollouts"]
            if r.get("score") is not None and r["clipgt_id"] in clips}


b441 = load(ROOT / "runs/leaderboard-merged-route-ep30/aggregate/results-summary.json")
arms = {}
for d in sorted((ROOT / "runs").glob("val-g*-cache-centre-max")):
    tag = d.name[len("val-"):-len("-cache-centre-max")]
    g = float(tag[1:].replace("p", "."))
    A = load(d / "aggregate" / "results-summary.json")
    if A:
        arms[g] = A

if 0.0 not in arms:
    sys.exit("gamma=0 validation arm missing")
base = arms[0.0]
common0 = [c for c in base if c in b441]
floor = statistics.fmean(base[c]["score"] - b441[c]["score"] for c in common0)
flips0 = sum(1 for c in common0 if (base[c]["score"] == 0) != (b441[c]["score"] == 0))

rows = []
for g, A in arms.items():
    if g == 0.0:
        continue
    common = [c for c in A if c in base]
    d = statistics.fmean(A[c]["score"] - base[c]["score"] for c in common)
    sd = statistics.pstdev(A[c]["score"] - base[c]["score"] for c in common)
    rows.append((d, g, len(common), sd))
rows.sort(reverse=True)

explain = "--explain" in sys.argv
if explain:
    print(f"잡음 바닥: gamma=0 arm vs 441 기준선, 같은 40클립 평균 Δ = {floor:+.4f}, 0점 판정 뒤집힘 {flips0}/{len(common0)}")
    for d, g, n, sd in rows:
        se = sd / max(n, 1) ** 0.5
        print(f"gamma={g:<6} Δ(vs gamma=0 arm) = {d:+.4f}  (n={n}, 표준오차 {se:.4f}, 잡음바닥 대비 {'초과' if d > abs(floor) else '이내'})")

if not rows:
    sys.exit("no candidate arms")
best_d, best_g, _, _ = rows[0]
choice = best_g if best_d > abs(floor) else 0.02
if explain:
    print(f"선택: gamma={choice}" + ("" if best_d > abs(floor) else "  (어느 후보도 잡음 바닥을 넘지 못해 441 추정 1위 0.02 로 진행)"))
else:
    print(choice)
