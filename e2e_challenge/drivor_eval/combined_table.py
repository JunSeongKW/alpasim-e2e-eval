# SPDX-License-Identifier: Apache-2.0
"""One AlpaSim 441 table: our DrivoR runs, the organizer reference runs, and published numbers.

    .venv/bin/python e2e_challenge/drivor_eval/combined_table.py \
        --original runs/drivor-drivor-original-441x3-<date> --ours runs/drivor-ours-hugsim-final-441x3-<date> \
        [--fit runs/leaderboard-drivor-8ref-<date>] --out runs/drivor-combined-alpasim441-<date>

Three groups of rows, kept apart because they were not produced the same way:
  1. our runs           -- this pipeline (e2e_challenge/drivor_eval, dev contract, 441 x 3)
  2. organizer references -- e2e_challenge/local_evaluation/data/pai (same dev contract, 441 x 3)
  3. published          -- published_alpasim441.json, copied from the papers, not run by us
Groups 1 and 2 are computed identically from results-summary.json: scene score is
the mean over scenes of the mean rollout score; the rates are the aggregation's
own metrics_results (collision_at_fault, offroad, left_corridor_laterally,
progress_clipped_rel) and km per incident is avg_dist_between_incidents_at_fault.
These are the same quantities the papers print (ECO's VaVAM row reproduces the
reference run exactly). The +- on our scene score is the std over three
evaluation passes formed by the three rollouts of each scene, the counterpart of
OPTED's three evaluation runs. PCS exists only for subjects in our Drive-IRT
fit (8 references + 2 DrivoR) and is left empty for published rows.
"""

import argparse
import collections
import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
REF = ROOT / "e2e_challenge/local_evaluation/data/pai"

NOTE_KO = (
    "※ '논문 보고값' 행은 우리가 직접 실행한 결과가 아니라 각 논문(OPTED, ECO)의 표에 보고된 수치를 그대로 옮긴 것입니다. "
    "같은 441개 장면(AlpaSim 대회 curated_val)과 같은 대회 채점 지표를 사용했지만, 렌더러 설정(harmonizer), "
    "초기 강제 GT 구간의 드라이버 호출 여부, 장면당 롤아웃 수, 각 모델을 시뮬레이터에 연결한 방식(카메라 구성·추론 주기·전처리) 등 "
    "평가 조건이 우리 실행과 완전히 동일하지 않을 수 있으므로, 이 값들과의 비교는 참고용으로 해석해야 합니다. "
    "예를 들어 같은 VaVAM이라도 OPTED 보고값은 3.9%, ECO 보고값과 공식 기준 실행은 11.0%로, 연결 방식에 따라 크게 달라집니다. "
    "'공식 기준 실행' 행(Alpamayo 1, VaVAM)은 대회 주최 측이 공개한 결과 파일로, 우리와 같은 dev 계약"
    "(카메라 6대 10 Hz, 1.7 s 강제 GT, 기본 비선형 MPC, 441장면×3롤아웃)에서 실행되었습니다. "
    "PCS는 공식 기준 8종과 DrivoR 2종을 함께 적합한 로컬 값(alpamayo1=2000, alternative_2=600 기준)이므로 논문 보고값 행에는 없습니다. "
    "± 는 논문의 경우 평가 3회 또는 학습 seed 3개의 표준편차이고, 우리 값은 장면당 롤아웃 3회를 3개의 평가 회차로 나눈 표준편차입니다. "
    "또한 모든 행이 같은 학습 조건은 아닙니다. '학습 도메인' 열의 제로샷 행만 평가 도메인(PAI 차량·NuRec 장면·AlpaSim)의 데이터를 학습에 쓰지 않았고, "
    "Alpamayo는 같은 차량 플랫폼의 대규모 내부 데이터로, Qwen-Drive는 PAI-AV 일부로 학습했으며, "
    "BC·RoaD-recovery·OPTED 행은 AlpaSim 안에서 NuRec 학습 장면(평가 441과 분리)으로 미세조정한 결과입니다."
)
NOTE_EN = (
    "Rows marked 'published' were not run by us; they are copied from the tables of the cited papers (OPTED, ECO). "
    "They use the same 441-scene split and the same challenge metrics, but evaluation details such as the renderer "
    "harmonizer, whether the driver is queried during the initial forced-GT segment, the number of rollouts per scene, "
    "and how each model is integrated with the simulator (camera set, planning rate, preprocessing) may differ from our "
    "runs, so comparisons with these rows are indicative only (e.g. the same VaVAM model scores 3.9% in OPTED but 11.0% "
    "in ECO and in the organizer reference run). Organizer reference rows (Alpamayo 1, VaVAM) are the published "
    "challenge reference runs under the same dev contract as ours (6 cameras at 10 Hz, 1.7 s forced GT, default "
    "nonlinear MPC, 441 scenes x 3 rollouts). PCS is our local Drive-IRT fit of the 8 references and the 2 DrivoR "
    "runs (alpamayo1 = 2000, alternative_2 = 600) and is not available for published rows. Training conditions also "
    "differ: only rows marked zero-shot used no data from the evaluation domain (PAI vehicle, NuRec scenes, AlpaSim); "
    "Alpamayo is pre-trained on large-scale internal data from the same vehicle platform, Qwen-Drive includes part of "
    "PAI-AV, and the BC / RoaD-recovery / OPTED rows are fine-tuned inside AlpaSim on NuRec training scenes disjoint "
    "from the 441 evaluation scenes."
)
COLUMNS = ["group", "model", "train_domain", "controller", "rollouts_per_scene", "scene_score_pct", "km_per_incident",
           "atfault_collision_pct", "offroad_pct", "corridor_exit_pct", "progress_pct", "pcs", "source"]
HEADER_KO = ["구분", "모델", "학습 도메인", "제어기", "롤아웃/장면", "장면 점수(%)↑", "책임 사고당 km↑", "책임 충돌(%)↓",
             "도로 이탈(%)↓", "4 m 이탈(%)↓", "진행률(%)↑", "PCS↑ (순위 구간)", "출처"]


def summary_path(p: Path) -> Path:
    return p if p.suffix == ".json" else p / "aggregate/results-summary.json"


def measured(path: Path):
    s = json.loads(path.read_text())
    m = s["metrics_results"][0]
    by = collections.defaultdict(list)
    for r in s["rollouts"]:
        if r.get("score") is not None:
            by[r["clipgt_id"]].append(r)
    per_scene = [statistics.mean(r["score"] for r in v) for v in by.values()]
    counts = collections.Counter(len(v) for v in by.values())
    passes = max(counts)
    pass_scores = []
    if passes > 1 and set(counts) == {passes}:
        for k in range(passes):
            pass_scores.append(100 * statistics.mean(sorted(v, key=lambda r: r["rollout_id"])[k]["score"]
                                                     for v in by.values()))
    score = 100 * statistics.mean(per_scene)
    sd = f"±{statistics.pstdev(pass_scores):.1f}" if pass_scores else ""
    return {
        "rollouts_per_scene": "/".join(str(c) for c in sorted(counts)) + f" ({len(by)} scenes)",
        "scene_score_pct": f"{score:.1f}{sd}",
        "km_per_incident": f"{m['avg_dist_between_incidents_at_fault']:.2f}",
        "atfault_collision_pct": f"{100 * m['collision_at_fault']:.1f}",
        "offroad_pct": f"{100 * m['offroad']:.1f}",
        "corridor_exit_pct": f"{100 * m['left_corridor_laterally']:.1f}",
        "progress_pct": f"{100 * m['progress_clipped_rel']:.1f}",
    }


def load_pcs(fit: Path):
    if not fit or not (fit / "capability_ranking.csv").exists():
        return {}
    with (fit / "capability_ranking.csv").open(newline="") as f:
        return {r["subject_id"]: f"{round(float(r['policy_capability_score']))} "
                                 f"({int(float(r['rank_lo']))}-{int(float(r['rank_hi']))})"
                for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--original", type=Path, required=True)
    ap.add_argument("--ours", type=Path, required=True)
    ap.add_argument("--fit", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    pcs = load_pcs(a.fit)
    rows = []
    for key, name, run in (("ours_hugsim_final", "Ours (DrivoR + Z_safe, A8/B12/D10)", a.ours),
                           ("drivor_original", "DrivoR (original V1)", a.original)):
        domain = ("제로샷 (NAVSIM navtrain 학습, 모델 선택은 HUGSIM)" if key == "ours_hugsim_final"
                  else "제로샷 (NAVSIM navtrain 학습)")
        rows.append({"group": "직접 평가 (우리)", "model": name, "train_domain": domain, "controller": "기본 비선형 MPC",
                     **measured(summary_path(run)), "pcs": pcs.get(key, ""),
                     "source": f"runs/{run.name if run.is_dir() else run.stem}"})
    for key, name, ctrl in (("alpamayo1", "Alpamayo 1 (R1 10B)", "기본 비선형 MPC"),
                            ("vavam-nonlinear", "VaVAM", "기본 비선형 MPC"),
                            ("vavam-linear", "VaVAM", "선형 MPC")):
        domain = ("같은 도메인 사전학습 (NVIDIA 내부 8만 h, PAI-AV 출처)" if key == "alpamayo1"
                  else "제로샷 (OpenDV·nuPlan·nuScenes 학습)")
        rows.append({"group": "공식 기준 실행", "model": name, "train_domain": domain, "controller": ctrl,
                     **measured(REF / key / "aggregate/results-summary.json"), "pcs": pcs.get(key, ""),
                     "source": f"AlpaSim local_evaluation reference `{key}`"})
    published = json.loads((HERE / "published_alpasim441.json").read_text())
    ctrl_ko = {"default MPC": "기본 MPC", "nonlinear MPC (default)": "기본 비선형 MPC", "linear MPC": "선형 MPC"}
    for r in published["rows"]:
        rows.append({"group": "논문 보고값", "model": r["model"], "train_domain": r["train_domain"], "controller": ctrl_ko.get(r["controller"], r["controller"]),
                     "rollouts_per_scene": "3 runs" if r["source"] == "OPTED" else "미기재",
                     **{k: r[k] for k in ("scene_score_pct", "km_per_incident", "atfault_collision_pct",
                                          "offroad_pct", "corridor_exit_pct", "progress_pct")},
                     "pcs": "",
                     "source": f"{r['source']} (arXiv {published['sources'][r['source']]['url'].rsplit('/', 1)[-1]})"})
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "combined_alpasim441.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEADER_KO)
        for r in rows:
            w.writerow([r[c] for c in COLUMNS])
    md = ["# AlpaSim 441 장면 (curated_val) 통합 결과", "",
          "| " + " | ".join(HEADER_KO) + " |", "|" + "---|" * len(HEADER_KO)]
    md += ["| " + " | ".join(str(r[c]) for c in COLUMNS) + " |" for r in rows]
    md += ["", NOTE_KO, "", "English note for the paper:", "", NOTE_EN, "",
           "출처:", *[f"- {v['citation']} — {v['url']}" for v in published["sources"].values()],
           "- 공식 기준 실행: NVlabs/alpasim e2e_challenge/local_evaluation/data/pai"]
    (a.out / "combined_alpasim441.md").write_text("\n".join(md) + "\n")
    (a.out / "note.txt").write_text(NOTE_KO + "\n\n" + NOTE_EN + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
