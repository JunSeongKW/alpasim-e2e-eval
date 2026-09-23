#!/usr/bin/env python3
"""Append one sweep trial's result to RANK_SWEEP_RESULTS.md."""
import argparse, collections, json, pathlib, sys

HERE = pathlib.Path(__file__).parent
PKG = HERE / "assets/drivesuprim/stage3_ep24_eval"
RESULTS = HERE / "RANK_SWEEP_RESULTS.md"
POOL = pathlib.Path("/tmp/claude-1000/-home-kaist5/f6b573e8-9153-4e4c-ba0a-564444b8bd06"
                    "/scratchpad/pool_both.json")
SHORT = {"no_at_fault_collisions": "NC", "drivable_area_compliance": "DAC",
         "ego_progress": "EP"}


def fmt_exp(d):
    if not d:
        return "1/1/1"
    return " ".join(f"{SHORT.get(k, k)}:{v:g}" for k, v in sorted(d.items()))


def band(t):
    if t < 0:
        return "미상"
    return "45+" if t >= 45 else "20~45" if t >= 20 else "<20"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    cfg = json.load(open(PKG / "rank_variants" / f"{a.name}.json"))
    summ = pathlib.Path(a.run_dir) / "aggregate/results-summary.json"
    if not summ.is_file():
        print(f"경고: 집계 없음 {summ}", file=sys.stderr)
        return 1
    rs = json.load(open(summ))["rollouts"]
    pool = json.load(open(POOL)) if POOL.is_file() else {}

    scores = [r["score"] for r in rs]
    avg = sum(scores) / len(scores)
    passed = sum(1 for s in scores if s > 0)
    reasons = collections.Counter(
        (r.get("failure_reason") or "pass").split(":")[0][:22] for r in rs)
    by = collections.defaultdict(list)
    for r in rs:
        cid = r["clipgt_id"].replace("clipgt-", "")
        by[band(pool.get(cid, {}).get("turn", -1))].append(r["score"])

    fr = cfg.get("pdm_rank_product_exponents_refine")
    fine_exp = fmt_exp(fr) if fr else "(coarse 와 동일)"
    lines = [
        f"\n### {a.name}" + (f" — {a.note}" if a.note else ""),
        "",
        "```",
        f"coarse   exp {fmt_exp(cfg.get('pdm_rank_product_exponents'))}"
        f"   imi {cfg['pdm_imi_rank_weight']}",
        f"fine     exp {fine_exp}   imi {cfg['pdm_imi_rank_weight_refine']}",
        "```",
        "",
        f"- **평균 {avg:.4f}   통과 {passed}/{len(scores)}**",
        "- 판정: " + "  ".join(f"{k} {v}" for k, v in reasons.most_common()),
        "- 회전각별: " + "   ".join(
            f"{b} {sum(v)/len(v):.3f}({len(v)})"
            for b in ("<20", "20~45", "45+") if (v := by.get(b))),
        f"- 위치: `{pathlib.Path(a.run_dir).name}`",
    ]
    txt = "\n".join(lines) + "\n"
    if not RESULTS.exists():
        RESULTS.write_text(
            "# 랭킹 지수·계수 탐색 결과\n\n"
            "계획: [`RANK_SWEEP_PLAN.md`](RANK_SWEEP_PLAN.md) · "
            "선별 클립 `val60_clips.txt` (458개에서 층화 추출)\n",
            encoding="utf-8")
    with RESULTS.open("a", encoding="utf-8") as f:
        f.write(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
