#!/usr/bin/env bash
# The validation table: each gamma on the random 40 clips against the 441
# baseline restricted to the same 40, paired clip by clip.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec "$ROOT/.venv/bin/python" - <<'PY'
import json, pathlib, statistics, sys

clips = set(pathlib.Path("e2e_challenge/route_reranker/clips_random40.txt").read_text().split())
b441 = json.load(open("runs/leaderboard-merged-route-ep30/aggregate/results-summary.json"))
B = {r["clipgt_id"]: r for r in b441["rollouts"]
     if r.get("score") is not None and r["clipgt_id"] in clips}


def load(tag):
    f = pathlib.Path(f"runs/val-{tag}-cache-centre-max/aggregate/results-summary.json")
    if not f.exists():
        return None
    return {r["clipgt_id"]: r for r in json.load(open(f))["rollouts"]
            if r.get("score") is not None}


def m(rows, key):
    return statistics.fmean((r["metrics"].get(key) or 0) for r in rows)


tags = []
for d in sorted(pathlib.Path("runs").glob("val-g*-cache-centre-max")):
    t = d.name[len("val-"):-len("-cache-centre-max")]
    tags.append((float(t[1:].replace("p", ".")), t))
tags.sort()

rows_b = list(B.values())
hdr = (f"{'gamma':>7} {'클립':>5} {'평균점수':>9} {'Δ(짝)':>8} {'0점률':>7} "
       f"{'corridor':>9} {'충돌':>6} {'at-fault':>9} {'이탈':>6} {'거리m':>7} {'판정바뀜':>8}")
print(hdr); print("-" * len(hdr))
print(f"{'441기준':>7} {len(rows_b):>5} {statistics.fmean(r['score'] for r in rows_b):>9.4f} {'':>8} "
      f"{sum(1 for r in rows_b if r['score']==0)/len(rows_b):>7.3f} "
      f"{m(rows_b,'left_corridor_laterally'):>9.3f} {m(rows_b,'collision_any'):>6.3f} "
      f"{m(rows_b,'collision_at_fault'):>9.3f} {m(rows_b,'offroad'):>6.3f} {m(rows_b,'dist_traveled_m'):>7.1f} {'':>8}")
for g, tag in tags:
    A = load(tag)
    if not A:
        print(f"{g:>7}   (실행 중)")
        continue
    common = [c for c in A if c in B]
    rows = [A[c] for c in common]
    delta = statistics.fmean(A[c]["score"] - B[c]["score"] for c in common)
    flips = sum(1 for c in common if (A[c]["score"] == 0) != (B[c]["score"] == 0))
    print(f"{g:>7} {len(rows):>5} {statistics.fmean(r['score'] for r in rows):>9.4f} {delta:>+8.4f} "
          f"{sum(1 for r in rows if r['score']==0)/len(rows):>7.3f} "
          f"{m(rows,'left_corridor_laterally'):>9.3f} {m(rows,'collision_any'):>6.3f} "
          f"{m(rows,'collision_at_fault'):>9.3f} {m(rows,'offroad'):>6.3f} {m(rows,'dist_traveled_m'):>7.1f} {flips:>8}")

print("\nΔ(짝) = 같은 40클립에서 441 기준선과 클립별로 짝지은 평균 차이. gamma=0 의 Δ 가 0 에 가까우면")
print("마운트 코드가 16워커에서 기준선을 재현한다는 뜻이고, 그 값이 나머지 γ 의 오차 하한이다.")
PY
