# SPDX-License-Identifier: Apache-2.0
"""Score the same rollouts under the three AlpaSim scoring conventions in the literature.

    UV_OFFLINE=1 .venv/bin/python e2e_challenge/drivor_eval/rescore_protocols.py \
        --run ours=runs/drivor-ours-hugsim-final-441x3-<date> \
        --run original=runs/drivor-drivor-original-441x3-<date> \
        [--check-against runs/<official aggregate dir or interim results-summary.json per run>] \
        --out runs/drivor-protocols-<stamp>

The simulator always runs the full 20 s; every convention below is a choice made
when the per-timestep metrics (rollouts/*/*/metrics.parquet) are reduced. All
three start at the first eval_relevant step and stop counting once the ego is
4 m from the logged trajectory (rendering-artifact guard); they differ in what
else ends the count and what a failure is:

  challenge  AlpaSim e2e_challenge code, also OPTED (arXiv 2609.20756) and ECO
             (arXiv 2609.31383) on the same 441 scenes. Count stops at the first
             collision of any kind (rear-end included) or off-road. Scene score
             = 0 on an at-fault (front/lateral) collision, off-road or a lateral
             4 m corridor exit, else min(progress / 0.8, 1).
  alpamayo   Alpamayo-R1 (arXiv 2511.00088, Sec. 6.1), also Qwen-Drive 1.0. The
             simulation ends at the first close encounter or off-road event;
             events past 4 m are excluded. Rates of close encounter (all and
             at-fault), off-road, progress, and km between events (all and
             at-fault). Close encounter is taken as the simulator's collision.
  road       RoaD (arXiv 2512.01993, Sec. 5.2.1 and App. B.1). Rear-end
             collisions by log-replayed followers are ignored, so only an
             at-fault collision or off-road is an incident and ends the count.
             A follower that rear-ends the ego drives on through it and the
             same contact turns front/lateral a few frames later; the whole
             contact episode that starts as rear-only is therefore ignored
             (at_fault_outside_rear_episodes).
             Driving score = km between incidents, at-fault collision rate,
             off-road rate, distance travelled.

Our runs are recomputed from the per-timestep data; the challenge recomputation
is checked rollout by rollout against the official aggregation. The organizer
reference runs only ship per-rollout aggregates computed the challenge way, so
they appear under challenge and alpamayo (which truncate identically) and not
under road.
"""

import argparse
import collections
import json
import math
import statistics
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "e2e_challenge/local_evaluation/data/pai"
REFERENCES = {"alpamayo1": "Alpamayo 1 (기준)", "vavam-linear": "VaVAM 선형 MPC (기준)",
              "vavam-nonlinear": "VaVAM 비선형 MPC (기준)"}
CORRIDOR_M = 4.0
SATURATION = 0.8
SHORT_GT_M = 5.0


def at_fault_outside_rear_episodes(w: pd.DataFrame) -> pd.Series:
    """Front/lateral contact that does not continue a rear-end contact.

    A log-replayed follower that runs into a slow ego keeps driving through it,
    so the same contact is classified rear first and front/lateral a few frames
    later (in our runs ~90% of such hits, median 0.7 s after the rear contact).
    RoaD ignores rear-end collisions caused by following vehicles; here a
    contiguous collision episode whose first frame is a rear-only contact is
    ignored as a whole, and only front/lateral contact outside such episodes
    counts as at-fault.
    """
    coll = (w["collision_any"] > 0).to_numpy()
    rear = (w["collision_rear"] > 0).to_numpy()
    fl = ((w["collision_front"] > 0) | (w["collision_lateral"] > 0)).to_numpy()
    keep = fl.copy()
    i = 0
    while i < len(coll):
        if coll[i]:
            j = i
            while j < len(coll) and coll[j]:
                j += 1
            if rear[i] and not fl[i]:
                keep[i:j] = False
            i = j
        else:
            i += 1
    return pd.Series(keep, index=w.index)


def rollout_record(path: Path, stop_on_any_collision: bool) -> dict:
    d = pd.read_parquet(path, columns=["name", "timestamps_us", "values"])
    w = d.pivot_table(index="timestamps_us", columns="name", values="values", aggfunc="first").sort_index()
    rel = w.index[w["eval_relevant"] > 0]
    if len(rel):
        w = w.loc[w.index >= rel[0]]

    def cut(mask):
        nonlocal w
        hit = w.index[mask.reindex(w.index).fillna(False).astype(bool)]
        if len(hit):
            w = w.loc[w.index <= hit[0]]

    cut(w["dist_to_gt_trajectory"] >= CORRIDOR_M)
    if stop_on_any_collision:
        at_fault = (w["collision_front"] > 0) | (w["collision_lateral"] > 0)
        cut((w["collision_any"] > 0) | (w["offroad"] > 0))
        at_fault = (w["collision_front"] > 0) | (w["collision_lateral"] > 0)
    else:
        at_fault = at_fault_outside_rear_episodes(w)
        cut(at_fault | (w["offroad"] > 0))
        at_fault = at_fault_outside_rear_episodes(w)
    rec = {
        "clip": path.parents[1].name, "rollout": path.parent.name,
        "collision_any": float((w["collision_any"] > 0).any()),
        "collision_at_fault": float(at_fault.any()),
        "collision_rear": float((w["collision_rear"] > 0).any()),
        "offroad": float((w["offroad"] > 0).any()),
        "left_corridor_laterally": float((w["left_corridor_laterally"] > 0).any()),
        "dist_m": float(w["dist_traveled_m"].iloc[-1]),
        "gt_dist_m": float(w["gt_dist_traveled_m"].iloc[-1]),
        "progress": float(w["progress_rel_to_total"].iloc[-1]),
    }
    progress_score = 1.0 if rec["gt_dist_m"] < SHORT_GT_M else min(max(rec["progress"], 0.0), 1.0) / SATURATION
    progress_score = min(progress_score, 1.0)
    failed = rec["collision_at_fault"] or rec["offroad"] or rec["left_corridor_laterally"]
    rec["scene_score"] = 0.0 if failed else progress_score
    return rec


def load_run(run: Path, stop_on_any_collision: bool) -> list:
    out = []
    for m in sorted(run.glob("rollouts/*/*/metrics.parquet")):
        if (m.parent / "_complete").exists():
            out.append(rollout_record(m, stop_on_any_collision))
    return out


def load_reference(key: str) -> list:
    s = json.loads((REF / key / "aggregate/results-summary.json").read_text())
    out = []
    for r in s["rollouts"]:
        m = r["metrics"]
        out.append({"clip": r["clipgt_id"], "rollout": r["rollout_id"], "scene_score": r["score"],
                    "collision_any": float(m["collision_any"] > 0), "collision_at_fault": float(m["collision_at_fault"] > 0),
                    "collision_rear": float(m["collision_rear"] > 0), "offroad": float(m["offroad"] > 0),
                    "left_corridor_laterally": float(m["left_corridor_laterally"] > 0),
                    "dist_m": float(m["dist_traveled_m"]), "gt_dist_m": float(m["gt_dist_traveled_m"]),
                    "progress": float(m["progress_rel_to_total"])})
    return out


def by_clip(rows):
    d = collections.defaultdict(list)
    for r in rows:
        d[r["clip"]].append(r)
    return d


def summarize(rows, clips):
    rs = [r for c in clips for r in rows[c]]
    mean = lambda k: 100 * statistics.mean(r[k] for r in rs)  # noqa: E731
    km = sum(r["dist_m"] for r in rs) / 1000
    inc_all = sum(1 for r in rs if r["collision_any"] or r["offroad"])
    inc_af = sum(1 for r in rs if r["collision_at_fault"] or r["offroad"])
    return {
        "scenes": len(clips), "rollouts": len(rs),
        "scene_score": 100 * statistics.mean(statistics.mean(r["scene_score"] for r in rows[c]) for c in clips),
        "zero_rate": 100 * statistics.mean(r["scene_score"] == 0 for r in rs),
        "collision_any": mean("collision_any"), "collision_at_fault": mean("collision_at_fault"),
        "collision_rear": mean("collision_rear"), "offroad": mean("offroad"),
        "corridor_exit": mean("left_corridor_laterally"), "progress": mean("progress"),
        "dist_m": statistics.mean(r["dist_m"] for r in rs),
        "km_per_event_all": km / inc_all if inc_all else math.inf,
        "km_per_event_at_fault": km / inc_af if inc_af else math.inf,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--check-against", action="append", default=[], metavar="NAME=results-summary.json",
                    help="official aggregation of the same run, to verify the challenge recomputation")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    runs = dict(x.split("=", 1) for x in a.run)
    challenge = {n: by_clip(load_run(Path(p), True)) for n, p in runs.items()}
    road = {n: by_clip(load_run(Path(p), False)) for n, p in runs.items()}
    refs = {k: by_clip(load_reference(k)) for k in REFERENCES}

    checks = {}
    for item in a.check_against:
        name, path = item.split("=", 1)
        official = {r["rollout_id"]: r for r in json.loads(Path(path).read_text())["rollouts"]}
        ours = {r["rollout"]: r for v in challenge[name].values() for r in v}
        common_ids = [i for i in ours if i in official]
        diffs = [abs(ours[i]["scene_score"] - official[i]["score"]) for i in common_ids]
        checks[name] = {"rollouts_compared": len(common_ids), "max_abs_scene_score_diff": max(diffs) if diffs else None}
        assert diffs and max(diffs) < 1e-6, f"challenge recomputation differs from the official aggregation: {checks[name]}"

    full = set.intersection(*[{c for c, v in rows.items() if len(v) >= 3} for rows in challenge.values()])
    clips = sorted(full)
    tables = {"challenge": {}, "alpamayo": {}, "road": {}}
    for name, rows in challenge.items():
        tables["challenge"][name] = tables["alpamayo"][name] = summarize(rows, clips)
    for name, rows in road.items():
        tables["road"][name] = summarize(rows, clips)
    for key, label in REFERENCES.items():
        s = summarize(refs[key], clips)
        tables["challenge"][label] = tables["alpamayo"][label] = s

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "protocols.json").write_text(json.dumps({"scenes": clips, "checks": checks, "tables": tables},
                                                     indent=1, ensure_ascii=False) + "\n")
    f1, f2 = (lambda x: f"{x:.1f}"), (lambda x: "∞" if math.isinf(x) else f"{x:.2f}")
    md = [f"# 같은 롤아웃, 세 가지 채점 방식 ({len(clips)}개 장면, 모델별 장면당 3롤아웃)", ""]
    if checks:
        md += ["대회 방식 재계산 검증: " + ", ".join(f"{k} {v['rollouts_compared']}개 롤아웃 최대 차이 {v['max_abs_scene_score_diff']:.1e}"
                                         for k, v in checks.items()), ""]
    md += ["## 1. 대회 방식 (AlpaSim e2e challenge · OPTED · ECO)", "",
           "| 주체 | 장면 점수(%)↑ | 0점(%)↓ | 책임 충돌(%)↓ | 도로 이탈(%)↓ | 4 m 이탈(%)↓ | 진행률(%)↑ | 책임 사고당 km↑ |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    md += [f"| {n} | {f1(s['scene_score'])} | {f1(s['zero_rate'])} | {f1(s['collision_at_fault'])} | {f1(s['offroad'])} | "
           f"{f1(s['corridor_exit'])} | {f1(s['progress'])} | {f2(s['km_per_event_at_fault'])} |" for n, s in tables["challenge"].items()]
    md += ["", "## 2. Alpamayo 방식 (Alpamayo-R1 · Qwen-Drive)", "",
           "| 주체 | 근접충돌 전체(%)↓ | 근접충돌 책임(%)↓ | 도로 이탈(%)↓ | 진행률(%)↑ | AlpaSim score 전체(km)↑ | AlpaSim score 책임(km)↑ |",
           "|---|---:|---:|---:|---:|---:|---:|"]
    md += [f"| {n} | {f1(s['collision_any'])} | {f1(s['collision_at_fault'])} | {f1(s['offroad'])} | {f1(s['progress'])} | "
           f"{f2(s['km_per_event_all'])} | {f2(s['km_per_event_at_fault'])} |" for n, s in tables["alpamayo"].items()]
    md += ["", "## 3. RoaD 방식 (후방 추돌 무시: 후방으로 시작된 접촉 구간 전체를 무시)", "",
           "| 주체 | Driving score(km/사고)↑ | 책임 충돌(%)↓ | 도로 이탈(%)↓ | 주행 거리(m)↑ | 진행률(%)↑ |",
           "|---|---:|---:|---:|---:|---:|"]
    md += [f"| {n} | {f2(s['km_per_event_at_fault'])} | {f1(s['collision_at_fault'])} | {f1(s['offroad'])} | "
           f"{f1(s['dist_m'])} | {f1(s['progress'])} |" for n, s in tables["road"].items()]
    md += ["", "기준 실행(Alpamayo 1, VaVAM)은 롤아웃별 집계값만 공개돼 RoaD 방식으로 다시 계산할 수 없다."]
    (a.out / "protocols.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
