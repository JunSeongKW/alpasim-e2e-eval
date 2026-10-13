"""Does the TensorRT image drive like vits512_g112 on PyTorch? (run by run_parity.sh)

    .venv/bin/python e2e_challenge/g112_trt/compare_parity.py

Same 441 clips, same dev preset and gains; the PyTorch side is the two 441 runs
of the bundle (10-11 and the 10-12 repeat). The question is whether the TRT run
is inside the run-to-run noise those two already show: mean scene score with a
paired scene bootstrap CI against each, and how many clips change pass/fail
(TRT vs either PyTorch run, next to PyTorch vs PyTorch). Writes
runs/<trt run>/parity.json and prints a table.
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = {
    "trt": "leaderboard-stage3-vits512-g112-kv28-trt-20261013",
    "torch_1": "leaderboard-stage3-vits512-g112-kv28-20261010",
    "torch_2": "leaderboard-stage3-vits512-g112-kv28-20261010-rerun-20261012",
}
RATES = ["collision_at_fault", "offroad", "left_corridor_laterally"]


def load(run):
    s = json.loads((ROOT / "runs" / run / "aggregate/results-summary.json").read_text())
    rows = {r["clipgt_id"]: r for r in s["rollouts"]}
    assert len(rows) == len(s["rollouts"]) == 441, f"{run}: {len(s['rollouts'])} rollouts"
    return rows, s["metrics_results"][0]


def boot(diff, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n, len(diff)))
    m = diff[idx].mean(1)
    return float(diff.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    data = {k: load(v) for k, v in RUNS.items()}
    clips = sorted(data["trt"][0])
    for k, (rows, _) in data.items():
        assert sorted(rows) == clips, f"{k}: different clip set"
    score = {k: np.array([rows[c]["score"] for c in clips]) for k, (rows, _) in data.items()}
    passed = {k: np.array([bool(rows[c]["passed"]) for c in clips]) for k, (rows, _) in data.items()}
    infra = {k: sum(1 for c in clips if "rpc" in str(rows[c].get("failure_reason") or "").lower())
             for k, (rows, _) in data.items()}
    out = {"runs": RUNS, "clips": len(clips), "per_run": {}, "pairs": {}}
    for k, (rows, agg) in data.items():
        out["per_run"][k] = {
            "scene_score": round(float(score[k].mean()), 4),
            "pass_rate": round(float(passed[k].mean()), 4),
            "km_between_at_fault": round(agg["avg_dist_between_incidents_at_fault"], 3),
            **{r: round(agg[r], 4) for r in RATES},
            "drive_rpc_mean_s": round(agg.get("driver_drive_rpc_duration_mean_s") or float("nan"), 3),
            "infra_failures": infra[k],
        }
    for a, b in (("trt", "torch_1"), ("trt", "torch_2"), ("torch_2", "torch_1")):
        mean, lo, hi = boot(score[a] - score[b])
        out["pairs"][f"{a}-{b}"] = {"diff": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)],
                                    "pass_flips": int((passed[a] != passed[b]).sum()),
                                    "score_changed": int((np.abs(score[a] - score[b]) > 1e-9).sum())}
    avg = (score["torch_1"] + score["torch_2"]) / 2
    mean, lo, hi = boot(score["trt"] - avg)
    out["pairs"]["trt-torch_avg"] = {"diff": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)]}
    # Inside the noise: the CI against the PyTorch average covers 0, and TRT flips no
    # more clips against a PyTorch run than the two PyTorch runs flip against each other (+50%).
    noise = out["pairs"]["torch_2-torch_1"]["pass_flips"]
    flips = max(out["pairs"]["trt-torch_1"]["pass_flips"], out["pairs"]["trt-torch_2"]["pass_flips"])
    out["within_noise"] = bool(lo <= 0 <= hi and flips <= 1.5 * noise + 5)
    (ROOT / "runs" / RUNS["trt"] / "parity.json").write_text(json.dumps(out, indent=2) + "\n")

    print("| run | scene score | pass | km/at-fault | at-fault | offroad | corridor | Drive RPC mean (s) | infra |")
    print("|---|---|---|---|---|---|---|---|---|")
    for k, r in out["per_run"].items():
        print(f"| {k} | {r['scene_score']:.4f} | {r['pass_rate']:.3f} | {r['km_between_at_fault']:.3f} | "
              f"{r['collision_at_fault']:.3f} | {r['offroad']:.3f} | {r['left_corridor_laterally']:.3f} | "
              f"{r['drive_rpc_mean_s']} | {r['infra_failures']} |")
    for k, p in out["pairs"].items():
        print(f"{k}: {p['diff']:+.4f} [{p['ci95'][0]:+.4f}, {p['ci95'][1]:+.4f}]"
              + (f"  pass flips {p['pass_flips']}, score changed {p['score_changed']}" if "pass_flips" in p else ""))
    print("within run-to-run noise:", out["within_noise"])
    return 0 if out["within_noise"] else 1


if __name__ == "__main__":
    sys.exit(main())
