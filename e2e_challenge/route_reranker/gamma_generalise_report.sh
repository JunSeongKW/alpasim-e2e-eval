#!/usr/bin/env bash
# One table for the gamma sweep: failures rescued, controls broken, per gamma.
#
# The baseline is the gamma=0 ARM of this same sweep, not the 441-clip run. That
# matters: gamma=0 and the 441 run are the same code and the same checkpoint, yet
# gamma=0 scored above zero on 8 of the 26 clips the 441 run scored at zero. The
# two differ only in concurrency -- 4 workers here against 16 there -- which
# shifts RPC timing and therefore which inference lands on which step. Those 8
# clips sit close enough to the 4 m corridor boundary that timing alone flips
# them. Measuring "rescued" against the 441 run would bank that drift as if the
# reranker had earned it.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
exec "$ROOT/.venv/bin/python" - "$@" <<'PY'
import json, pathlib, re, statistics, sys

HERE = pathlib.Path("e2e_challenge/route_reranker")
fails = set(HERE.joinpath("clips_turnfail.txt").read_text().split())
ctrls = set(HERE.joinpath("clips_control.txt").read_text().split())

# Turn-ness is a property of the scene, so it can be read from whichever arm
# logged it and applied to all of them. The driver only sees a session uuid; the
# runtime worker log is what pairs that uuid with a clip.
session_clip = {}
for w in pathlib.Path("runs").glob("gen-*-cache-centre-max/txt-logs/runtime_worker_*.log"):
    clip = None
    for line in w.read_text(errors="ignore").splitlines():
        m = re.search(r"(clipgt-[0-9a-f-]{36})", line)
        if m:
            clip = m.group(1)
        m = re.search(r"session=([0-9a-f]{8})", line)
        if m and clip:
            session_clip.setdefault(m.group(1), clip)

turn = {}
for log in pathlib.Path("runs").glob("gen-*-cache-centre-max.driver-*.log"):
    for line in log.read_text(errors="ignore").splitlines():
        m = re.search(r"route_turn_deg=([\d.]+) session=([0-9a-f]{8})", line)
        if not m:
            continue
        clip = session_clip.get(m.group(2))
        if clip:
            turn[clip] = max(turn.get(clip, 0.0), float(m.group(1)))

TURN_DEG = 20.0          # the bundle's own threshold for a turning frame


def load(tag):
    f = pathlib.Path(f"runs/gen-{tag}-cache-centre-max/aggregate/results-summary.json")
    if not f.exists():
        return None
    return {r["clipgt_id"]: r
            for r in json.load(open(f))["rollouts"] if r.get("score") is not None}


tags = []
for d in sorted(pathlib.Path("runs").glob("gen-g*-cache-centre-max")):
    t = d.name[len("gen-"):-len("-cache-centre-max")]
    tags.append((float(t[1:].replace("p", ".")), t))
tags.sort()
if not tags:
    print("아직 결과 없음")
    sys.exit()

base = load("g0")
if base is None:
    print("gamma=0 arm 이 아직 끝나지 않아 짝지은 비교를 할 수 없다")
    sys.exit()

hdr = (f"{'gamma':>7} {'클립':>5} {'해결':>8} {'파손':>8} {'회전해결':>9} {'직진해결':>9} "
       f"{'평균점수':>9} {'0점률':>7} {'corridor':>9} {'충돌':>6} {'거리m':>7}")
print(hdr)
print("-" * len(hdr))
for g, tag in tags:
    R = load(tag)
    if not R:
        print(f"{g:>7}   (아직 실행 중)")
        continue
    fixed = broken = 0
    t_fix = t_tot = s_fix = s_tot = 0
    for c, r in R.items():
        b = base.get(c)
        if b is None:
            continue
        is_turn = turn.get(c, 0.0) >= TURN_DEG
        if c in fails:
            (t_tot := t_tot + 1) if is_turn else (s_tot := s_tot + 1)
            if r["score"] > 0 and b["score"] == 0:
                fixed += 1
                if is_turn:
                    t_fix += 1
                else:
                    s_fix += 1
        if c in ctrls and b["score"] > 0.95 and r["score"] < 0.95:
            broken += 1
    n_f = sum(1 for c in R if c in fails)
    n_c = sum(1 for c in R if c in ctrls)
    print(f"{g:>7} {len(R):>5} {fixed:>3}/{n_f:<4} {broken:>3}/{n_c:<4} "
          f"{t_fix:>4}/{t_tot:<4} {s_fix:>4}/{s_tot:<4} "
          f"{statistics.fmean(r['score'] for r in R.values()):>9.4f} "
          f"{sum(1 for r in R.values() if r['score'] == 0) / len(R):>7.3f} "
          f"{statistics.fmean((r['metrics'].get('left_corridor_laterally') or 0) for r in R.values()):>9.3f} "
          f"{statistics.fmean((r['metrics'].get('collision_any') or 0) for r in R.values()):>6.3f} "
          f"{statistics.fmean((r['metrics'].get('dist_traveled_m') or 0) for r in R.values()):>7.1f}")

all_clips = fails | ctrls
known = sum(1 for c in all_clips if c in turn)
n_turn = sum(1 for c in all_clips if turn.get(c, 0) >= TURN_DEG)
print(f"\n회전 분류: {known}/{len(all_clips)} 클립에 route 기록 있음; "
      f"그중 {TURN_DEG:.0f}도 이상 회전 {n_turn}개")
if known < len(all_clips):
    print("  (session 기록은 07:05 이후 기동한 arm 부터 남으므로 그 arm 이 끝나면 채워진다)")

# The 441-clip run as a separate line, to show how much of any gap is drift.
b441 = pathlib.Path("runs/leaderboard-merged-route-ep30/aggregate/results-summary.json")
if b441.exists():
    B = {r["clipgt_id"]: r for r in json.load(open(b441))["rollouts"]
         if r.get("score") is not None}
    same = [c for c in base if c in B]
    flip = sum(1 for c in same if B[c]["score"] == 0 and base[c]["score"] > 0)
    print(f"\n동일 코드 재현성: gamma=0 이 441 실행과 다른 판정을 낸 클립 {flip}/{len(same)} "
          f"(워커 16 -> 4 로 바뀐 타이밍 차이. 4 m 경계에 걸친 클립이 뒤집힌다)")
    print(f"  441 실행 평균점수 {statistics.fmean(B[c]['score'] for c in same):.4f} "
          f"vs gamma=0 {statistics.fmean(base[c]['score'] for c in same):.4f}")
PY
