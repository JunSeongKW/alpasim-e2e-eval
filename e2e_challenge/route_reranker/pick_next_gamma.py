"""Choose the next gamma to evaluate on the 441 set, from the results so far.

A 441-clip run costs about four hours, so the search has to be frugal: every
proposal must be one whose answer changes what we would do next. The rule has
two phases.

Phase 1, a fixed ladder. 0.02, 0.05, 0.01, 0.1, 0.005 -- spread over the range
where gamma is known to do anything at all. Below 0.005 the selector barely
fires (on axe-v9, gamma 0.0005 changed 1 frame in 1892) and above 0.1 the route
cost starts overriding the model wherever the two disagree. The ladder is
ordered by prior plausibility, not by size, so that an early stop still leaves
the most informative points measured.

Phase 2, bisection on the winning bracket. With the ladder done, the best point
and its better-scoring neighbour bound an interval; the next proposal is their
geometric midpoint, because gamma acts multiplicatively (it is score points per
metre of route cost, and the costs span orders of magnitude). Bisecting in log
space keeps the proposals evenly informative. The search stops when the bracket
is narrower than a factor of RESOLUTION, at which point the remaining spread is
smaller than the simulator's own run-to-run noise and a further four hours would
not tell us anything.

The stopping rule is deliberately about resolution and not about score
differences. Two gammas half a factor apart can differ by less than the noise
floor while still being ordered consistently; what makes further search
pointless is that we could no longer act on the answer.

Prints one number -- the next gamma -- or DONE with the reason on stderr.
"""
import argparse
import json
import pathlib
import sys

LADDER = [0.02, 0.05, 0.01, 0.1, 0.005]
RESOLUTION = 1.35  # stop once the bracket is narrower than this factor
SAME = 1e-9


def load(ledger: pathlib.Path) -> list[dict]:
    if not ledger.exists():
        return []
    rows = []
    for line in ledger.read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    # Keep the last entry per gamma, in case a gamma was ever re-run.
    by_gamma = {}
    for r in rows:
        if r.get("score") is not None:
            by_gamma[round(float(r["gamma"]), 10)] = r
    return sorted(by_gamma.values(), key=lambda r: float(r["gamma"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--explain", action="store_true")
    a = ap.parse_args()

    done = load(pathlib.Path(a.ledger))
    measured = [float(r["gamma"]) for r in done]

    def already(g: float) -> bool:
        return any(abs(g - m) < max(SAME, m * 0.02) for m in measured)

    if a.explain:
        print("측정된 감마:", file=sys.stderr)
        for r in sorted(done, key=lambda r: -float(r["score"])):
            print(
                f"  γ={float(r['gamma']):<8.4g} 점수 {float(r['score']):.4f}"
                f"  0점률 {r.get('zero_rate', float('nan')):.3f}"
                f"  {r.get('run_dir', '')}",
                file=sys.stderr,
            )

    # Phase 1: the ladder, in order.
    for g in LADDER:
        if not already(g):
            if a.explain:
                print(f"→ 사다리 단계: γ={g}", file=sys.stderr)
            print(g)
            return

    # Phase 2: bisect between the best point and its better neighbour.
    ranked = sorted(done, key=lambda r: -float(r["score"]))
    best = float(ranked[0]["gamma"])
    scored = {float(r["gamma"]): float(r["score"]) for r in done}
    order = sorted(scored)
    i = order.index(best)
    left = order[i - 1] if i > 0 else None
    right = order[i + 1] if i + 1 < len(order) else None

    # The side to subdivide is the neighbour that scored higher; the optimum
    # lies between the best point and whichever side falls off more slowly.
    cand = [x for x in (left, right) if x is not None]
    if not cand:
        print("DONE: 측정점이 하나뿐", file=sys.stderr)
        return
    partner = max(cand, key=lambda x: scored[x])

    lo, hi = sorted((best, partner))
    if lo <= 0:
        # gamma=0 is the no-reranker baseline; bisecting toward it means
        # halving the smallest positive gamma instead.
        nxt = hi / 2.0
        if already(nxt) or nxt < 1e-4:
            print(f"DONE: 0 쪽으로 더 쪼갤 의미 없음 (하한 {hi:g})", file=sys.stderr)
            return
    else:
        if hi / lo < RESOLUTION:
            print(
                f"DONE: 구간 [{lo:g}, {hi:g}] 이 배율 {RESOLUTION} 보다 좁다"
                f" -- 남은 차이가 시뮬레이터 잡음보다 작다",
                file=sys.stderr,
            )
            return
        nxt = (lo * hi) ** 0.5

    if already(nxt):
        print(f"DONE: 다음 후보 {nxt:.4g} 가 이미 측정됨", file=sys.stderr)
        return
    if a.explain:
        print(
            f"→ 최고점 γ={best:g}(점수 {scored[best]:.4f}) 와"
            f" 더 높은 이웃 γ={partner:g}(점수 {scored[partner]:.4f}) 의"
            f" 기하 중점: γ={nxt:.4g}",
            file=sys.stderr,
        )
    print(f"{nxt:.4g}")


if __name__ == "__main__":
    main()
