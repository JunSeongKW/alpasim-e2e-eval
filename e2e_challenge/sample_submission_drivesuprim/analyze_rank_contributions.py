#!/usr/bin/env python3
"""Aggregate the ranking-score decomposition over a driver log.

The driver prints one RANKDBG line per inference:

    [DriveSuprim] RANKDBG k=256 sel=12 total=-9.8432 | il=-5.545(38.2%,r3) ...
                  ... nc=-0.001(0.4%,r91) ... | flip=ep,il

Each term reports three things about one decision:

  value       the term's contribution to the driven candidate's score
  influence   the term's spread over the candidates as a share of all spread --
              a term that barely varies cannot reorder them, however large
  solo_rank   where the driven candidate would rank under that term ALONE

and `flip` lists the terms whose removal changes which candidate is driven.
Influence says who could have decided; flip says who did.
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

TERM_RE = re.compile(
    r"(?P<name>[a-z_]+)=(?P<value>[-+][\d.]+)\((?P<influence>[\d.]+)%,r(?P<rank>\d+)\)"
)
LINE_RE = re.compile(r"RANKDBG k=(?P<k>\d+) sel=(?P<sel>\d+) total=(?P<total>[-+][\d.]+)")
FLIP_RE = re.compile(r"\| flip=(?P<flips>[a-z,]+|-)\s*$")

LABELS = {
    "il": "imitation (il)",
    "nc": "no_at_fault_collisions (nc)",
    "dac": "drivable_area_compliance (dac)",
    "gt": "gt_compliance (gt)",
    "ep": "ego_progress (ep)",
    "agg": "aggregate pdm_score (agg)",
}
ORDER = ("il", "nc", "dac", "gt", "ep", "agg")


def parse(paths: list[Path]) -> tuple[list[dict], int]:
    frames: list[dict] = []
    for path in paths:
        with path.open(errors="replace") as handle:
            for line in handle:
                if "RANKDBG" not in line:
                    continue
                head = LINE_RE.search(line)
                if not head:
                    continue
                terms = {
                    m.group("name"): {
                        "value": float(m.group("value")),
                        "influence": float(m.group("influence")),
                        "solo_rank": int(m.group("rank")),
                    }
                    for m in TERM_RE.finditer(line)
                }
                flip_match = FLIP_RE.search(line.rstrip())
                flips = []
                if flip_match and flip_match.group("flips") != "-":
                    flips = flip_match.group("flips").split(",")
                frames.append(
                    {
                        "k": int(head.group("k")),
                        "selected": int(head.group("sel")),
                        "total": float(head.group("total")),
                        "terms": terms,
                        "flips": flips,
                    }
                )
    return frames, len(paths)


def report(frames: list[dict]) -> None:
    if not frames:
        print("RANKDBG 라인이 없습니다. DRIVESUPRIM_RANK_LOG_EVERY 설정과 로그 경로를 확인하십시오.")
        return

    print(f"프레임(추론) 수: {len(frames)}   후보 수 k={frames[0]['k']}")
    print()
    print("항목별 기여도 — 무엇이 후보 순서를 실제로 바꾸는가")
    print(
        f"  {'term':32s} {'영향도%':>9s} {'중앙값%':>9s} {'값(평균)':>11s} "
        f"{'단독순위':>9s} {'결정률%':>9s}"
    )

    influence = defaultdict(list)
    values = defaultdict(list)
    solo = defaultdict(list)
    flip_counter: Counter = Counter()
    for frame in frames:
        for name, entry in frame["terms"].items():
            influence[name].append(entry["influence"])
            values[name].append(entry["value"])
            solo[name].append(entry["solo_rank"])
        for name in frame["flips"]:
            flip_counter[name] += 1

    for name in ORDER:
        if name not in influence:
            continue
        print(
            f"  {LABELS[name]:32s} "
            f"{statistics.fmean(influence[name]):9.1f} "
            f"{statistics.median(influence[name]):9.1f} "
            f"{statistics.fmean(values[name]):+11.3f} "
            f"{statistics.fmean(solo[name]):9.1f} "
            f"{100.0 * flip_counter[name] / len(frames):9.1f}"
        )

    print()
    print("  영향도%   후보들 사이 점수 퍼짐(std)에서 그 항목이 차지하는 비율")
    print("  단독순위  그 항목만으로 줄 세웠을 때 실제 주행한 후보의 등수 (1이면 그 항목이 곧 선택)")
    print("  결정률%   그 항목을 빼면 선택이 바뀌는 프레임 비율 — 실제로 결정한 비율")
    print()

    decided = sum(1 for frame in frames if frame["flips"])
    print(f"단일 항목 제거로 선택이 바뀌는 프레임: {decided} / {len(frames)} "
          f"({100.0 * decided / len(frames):.1f}%)")
    combos = Counter(
        ",".join(sorted(frame["flips"])) or "(없음)" for frame in frames
    )
    print("결정 항목 조합 상위:")
    for combo, count in combos.most_common(6):
        print(f"  {combo:24s} {count:6d}  ({100.0 * count / len(frames):5.1f}%)")

    dominant = Counter(
        max(frame["terms"].items(), key=lambda kv: kv[1]["influence"])[0]
        for frame in frames
        if frame["terms"]
    )
    print()
    print("프레임별 최대 영향도 항목:")
    for name, count in dominant.most_common():
        print(f"  {LABELS.get(name, name):32s} {count:6d}  ({100.0 * count / len(frames):5.1f}%)")


STOP_RE = re.compile(
    r"RANKSTOP k=(?P<k>\d+) stop=(?P<stop>\d+)(?: driven_end=(?P<driven>[\d.]+)m)?"
)
STOP_FULL_RE = re.compile(
    r"best_stop=#(?P<idx>\d+) stop_end=(?P<send>[\d.]+)m stop_rank=(?P<srank>\d+) "
    r"stop_il_rank=(?P<ilrank>\d+) gap=(?P<gap>[-+][\d.]+) \| (?P<shares>.*)$"
)
SHARE_RE = re.compile(r"(?P<name>[a-z_]+)=(?P<value>[-+][\d.]+)")


def report_stop(paths: list[Path]) -> None:
    """왜 멈추지 않았는가 -- 정지 후보와의 점수 차를 항목별로 분해."""

    rows = []
    no_stop = 0
    for path in paths:
        with path.open(errors="replace") as handle:
            for line in handle:
                if "RANKSTOP" not in line:
                    continue
                head = STOP_RE.search(line)
                if not head:
                    continue
                if head.group("stop") == "0":
                    no_stop += 1
                    continue
                full = STOP_FULL_RE.search(line.rstrip())
                if not full:
                    continue
                rows.append(
                    {
                        "stop_count": int(head.group("stop")),
                        "driven_end": float(head.group("driven") or 0.0),
                        "stop_end": float(full.group("send")),
                        "stop_rank": int(full.group("srank")),
                        "stop_il_rank": int(full.group("ilrank")),
                        "gap": float(full.group("gap")),
                        "shares": {
                            m.group("name"): float(m.group("value"))
                            for m in SHARE_RE.finditer(full.group("shares"))
                        },
                    }
                )

    print()
    print("=" * 72)
    print("왜 멈추지 않았는가 — 주행한 후보 vs 가장 나은 정지 후보")
    print("=" * 72)
    if not rows:
        print(f"  정지 후보가 후보군에 하나도 없던 프레임: {no_stop}")
        print("  (RANKSTOP 라인이 없으면 계측이 적용되지 않은 로그입니다)")
        return

    print(f"  정지 후보가 있던 프레임: {len(rows)}   없던 프레임: {no_stop}")
    print(f"  후보군 내 정지 후보 수 : 평균 {statistics.fmean(r['stop_count'] for r in rows):.1f}개")
    print(f"  주행한 계획의 종점     : 평균 {statistics.fmean(r['driven_end'] for r in rows):.1f} m")
    print(f"  정지 후보의 total 등수 : 평균 {statistics.fmean(r['stop_rank'] for r in rows):.1f}위")
    print(f"  정지 후보의 il 등수    : 평균 {statistics.fmean(r['stop_il_rank'] for r in rows):.1f}위")
    print(f"  점수 차(주행 - 정지)   : 평균 {statistics.fmean(r['gap'] for r in rows):+.3f}")
    print()
    print("  그 점수 차를 만든 항목 (양수 = 주행을 밀어줌, 음수 = 정지를 밀어줌)")
    names = ORDER
    agg = {name: [] for name in names}
    for row in rows:
        for name in names:
            if name in row["shares"]:
                agg[name].append(row["shares"][name])
    for name in names:
        if not agg[name]:
            continue
        mean = statistics.fmean(agg[name])
        pushed = 100.0 * sum(1 for v in agg[name] if v > 0) / len(agg[name])
        print(f"    {LABELS[name]:32s} {mean:+8.3f}   주행을 민 프레임 {pushed:5.1f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", type=Path, help="driver log file(s)")
    args = parser.parse_args()
    missing = [p for p in args.logs if not p.is_file()]
    if missing:
        print(f"ERROR: 로그를 찾을 수 없습니다: {missing}", file=sys.stderr)
        return 2
    frames, _ = parse(args.logs)
    report(frames)
    report_stop(args.logs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
