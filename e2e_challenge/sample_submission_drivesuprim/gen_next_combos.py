#!/usr/bin/env python3
"""Propose the next batch of ranking combinations and append them to the queue.

Called by the sweep daemon whenever the queue runs dry, so the search keeps
going without a human picking the next values. Half the batch perturbs the best
result so far (exploit), half samples the space at random (explore), and both
skip signatures already measured.
"""
import glob, itertools, json, os, pathlib, random, subprocess, sys

HERE = pathlib.Path(__file__).parent
RB = pathlib.Path("/home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim/runs"
                  "/e2e_challenge_drivesuprim_debug")
VAR = HERE / "assets/drivesuprim/stage3_ep24_eval/rank_variants"
QUEUE = HERE / "sweep_queue.txt"
SHORT = {"no_at_fault_collisions": "NC", "drivable_area_compliance": "DAC",
         "ego_progress": "EP"}
# coarse 를 바꾼 5건은 하나(imi 2.0)를 빼고 모두 기준선 이하였고, fine imi 는
# 0.02 와 0.1 이 동률에 나머지가 명확히 낮았다. 그래서 탐색은 fine 지수 3축으로
# 좁힌다 -- 축을 줄이면 같은 시행 수로 훨씬 촘촘히 볼 수 있다.
GRID = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.3, 1.7, 2.0, 2.5, 3.0]
IMI_C = [1.0]
IMI_F = [0.02]


def sig(cfg):
    def e(d):
        return tuple(sorted((SHORT.get(k, k), round(float(v), 3))
                            for k, v in (d or {}).items() if float(v) != 1.0))
    return (e(cfg.get("pdm_rank_product_exponents")),
            e(cfg.get("pdm_rank_product_exponents_refine")),
            round(float(cfg["pdm_imi_rank_weight"]), 3),
            round(float(cfg["pdm_imi_rank_weight_refine"]), 3))


def measured():
    """signature -> mean score, over every trial that produced a summary."""
    out = {}
    for d in sorted(glob.glob(str(RB / "sweep_*"))):
        name = pathlib.Path(d).name.split("_")[1]
        cfg_p = VAR / f"{name}.json"
        summ = pathlib.Path(d) / "aggregate/results-summary.json"
        if not (cfg_p.is_file() and summ.is_file()):
            continue
        rs = json.load(open(summ))["rollouts"]
        s = sum(r["score"] for r in rs) / max(len(rs), 1)
        out.setdefault(sig(json.load(open(cfg_p))), []).append(s)
    return {k: sum(v) / len(v) for k, v in out.items()}


def to_args(fine, coarse, ci, fi):
    a = []
    if coarse:
        a += ["--coarse-exp", ",".join(f"{k}:{v:g}" for k, v in coarse.items())]
    if fine:
        a += ["--fine-exp", ",".join(f"{k}:{v:g}" for k, v in fine.items())]
    a += ["--coarse-imi", str(ci), "--fine-imi", str(fi)]
    return a


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    seen = measured()
    if not seen:
        print("측정 결과가 없어 제안을 만들 수 없다", file=sys.stderr)
        return 1
    best_sig = max(seen, key=seen.get)
    rng = random.Random()
    bc, bf, bci, bfi = best_sig
    best_fine = {k: v for k, v in bf}
    best_coarse = {k: v for k, v in bc}

    tried = set(seen)
    props, guard = [], 0
    while len(props) < n and guard < 4000:
        guard += 1
        if len(props) < n // 2:                      # exploit: 최고 조합 주변
            fine = dict(best_fine) or {"DAC": 0.5}
            coarse = dict(best_coarse)
            for k in rng.sample(["NC", "DAC", "EP"], rng.choice([1, 1, 2])):
                fine[k] = rng.choice(GRID)
            ci, fi = bci, bfi
        else:                                        # explore: fine 지수 무작위
            fine = {k: rng.choice(GRID) for k in ("NC", "DAC", "EP")
                    if rng.random() < 0.8}
            coarse = {}
            ci, fi = 1.0, 0.02
        fine = {k: v for k, v in fine.items() if v != 1.0}
        coarse = {k: v for k, v in coarse.items() if v != 1.0}
        s = (tuple(sorted(coarse.items())), tuple(sorted(fine.items())),
             round(ci, 3), round(fi, 3))
        if s in tried:
            continue
        tried.add(s)
        props.append((fine, coarse, ci, fi))

    existing = QUEUE.read_text() if QUEUE.exists() else ""
    idx = 1
    lines = []
    for fine, coarse, ci, fi in props:
        while f"G{idx}|" in existing or f"G{idx}." in existing:
            idx += 1
        name = f"G{idx}"; idx += 1
        subprocess.run([sys.executable, str(HERE / "make_rank_variant.py"), name]
                       + to_args(fine, coarse, ci, fi),
                       check=True, stdout=subprocess.DEVNULL)
        desc = (f"fine {fine or '1/1/1'} coarse {coarse or '1/1/1'} "
                f"imi c{ci}/f{fi}")
        lines.append(f"{name}|자동제안: {desc}")
    with QUEUE.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"{len(lines)}개 제안 추가 (최고 {seen[best_sig]:.4f} 기준)")
    for l in lines:
        print("  " + l)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
