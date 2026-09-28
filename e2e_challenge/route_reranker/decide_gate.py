"""Pick a collision-gate setting from the sweep, on two objectives at once.

The request was for a combination that improves the scene score *and* at-fault
distance. Those can trade against each other: widening the gate drops the
candidates that would have collided, which lifts at-fault distance, but it also
drops candidates that were merely close, which costs progress and therefore both
distance travelled and score. So the choice has to be stated, not implied.

The rule: among settings whose at-fault distance is at least the baseline's, take
the highest scene score. A setting that buys score by giving up at-fault distance
is printed but never chosen -- at-fault distance is the axis the request named,
and it is also what the official capability score weights most heavily (a
checkpoint has already been seen to rank 4th on scene score and 12th on PCS on
the strength of that one column).

--combine prints "margin scale conf" for the stage-B run: the best value found
for each knob independently, taken together. It prints nothing when no knob
cleared the bar, which is itself an answer -- it means the gate as shipped is
already at a local optimum for this checkpoint.

A note on reading the output. The differences here are small and the simulator is
not deterministic: the same code and checkpoint disagreed with itself on 8 of 38
clips when only the worker count changed. Two settings within a few thousandths
of a point are not ordered by this data, and the printed ranking should not be
read as if they were.
"""
import argparse
import json
import pathlib
import sys

# Below this, a difference in scene score is within run-to-run noise.
NOISE_SCORE = 0.005


def load(path: pathlib.Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            r = json.loads(line)
            if r.get("score") is not None:
                rows.append(r)
    # Last entry wins per label, in case a run was repeated.
    by_label = {r["label"]: r for r in rows}
    return list(by_label.values())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--explain", action="store_true")
    ap.add_argument("--combine", action="store_true")
    a = ap.parse_args()

    rows = load(pathlib.Path(a.ledger))
    base = next((r for r in rows if r["label"] == "baseline"), None)
    if base is None:
        print("no baseline in ledger", file=sys.stderr)
        return
    others = [r for r in rows if r["label"] != "baseline"]

    def af(r):  # at-fault km; None when a run had no at-fault incident at all
        return r.get("at_fault_km") or 0.0

    if a.combine:
        # Best value per knob among the runs that varied only that knob and did
        # not lose at-fault distance.
        pick = {"margin": base["margin"], "scale": base["scale"], "conf": base["conf"]}
        for knob in ("margin", "scale", "conf"):
            cands = [
                r for r in others
                if r[knob] != base[knob]
                and all(r[k] == base[k] for k in pick if k != knob)
                and af(r) >= af(base)
                and r["score"] > base["score"]
            ]
            if cands:
                pick[knob] = max(cands, key=lambda r: (r["score"], af(r)))[knob]
        if all(pick[k] == base[k] for k in pick):
            return  # nothing helped; print nothing
        print(f"{pick['margin']} {pick['scale']} {pick['conf']}")
        return

    if a.explain:
        print(f"{'설정':<12}{'margin':>8}{'scale':>7}{'conf':>6}"
              f"{'점수':>9}{'at-fault km':>13}{'0점률':>8}{'at-fault충돌':>12}{'주행m':>8}")
        for r in sorted(rows, key=lambda r: -r["score"]):
            print(f"{r['label']:<12}{r['margin']:>8.2f}{r['scale']:>7.2f}{r['conf']:>6.2f}"
                  f"{r['score']:>9.4f}{af(r):>13.3f}{r.get('zero_rate', 0):>8.3f}"
                  f"{r.get('collision_at_fault', 0):>12.4f}{r.get('dist_traveled_m', 0):>8.1f}")

        print(f"\n기준선 대비 (점수 차이 {NOISE_SCORE} 미만은 잡음 범위)")
        for r in sorted(others, key=lambda r: -r["score"]):
            ds, da = r["score"] - base["score"], af(r) - af(base)
            verdict = ("둘 다 개선" if ds > NOISE_SCORE and da > 0 else
                       "at-fault 만 개선" if da > 0 and ds > -NOISE_SCORE else
                       "점수만 개선 (at-fault 손실)" if ds > NOISE_SCORE else
                       "개선 없음")
            print(f"  {r['label']:<12} 점수 {ds:+.4f}  at-fault {da:+.3f} km   {verdict}")

        # Pareto front: nothing else beats these on both axes at once.
        front = [r for r in rows
                 if not any(o["score"] > r["score"] and af(o) > af(r) for o in rows)]
        print("\n파레토 front (두 축 동시에 우월한 설정이 없는 것들)")
        for r in sorted(front, key=lambda r: -r["score"]):
            print(f"  {r['label']:<12} 점수 {r['score']:.4f}  at-fault {af(r):.3f} km")

    eligible = [r for r in rows if af(r) >= af(base)]
    best = max(eligible, key=lambda r: r["score"]) if eligible else base
    if best["label"] == base["label"]:
        print("\n결론: 기준선(margin 0.0, scale 1.1, conf 0.3)을 넘는 조합이 없다."
              " 배포된 게이트 설정이 이 체크포인트에서는 이미 국소 최적이다.", file=sys.stderr)
    else:
        print(f"\n결론: {best['label']}"
              f" (margin={best['margin']}, scale={best['scale']}, conf={best['conf']})"
              f" — 점수 {best['score']:.4f} ({best['score'] - base['score']:+.4f}),"
              f" at-fault {af(best):.3f} km ({af(best) - af(base):+.3f})", file=sys.stderr)
    print(f"{best['margin']} {best['scale']} {best['conf']}")


if __name__ == "__main__":
    main()
