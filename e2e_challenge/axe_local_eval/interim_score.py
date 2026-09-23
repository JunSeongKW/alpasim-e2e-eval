"""Score the finished rollouts of a run that is still going, using AlpaSim's own aggregation.

Hand-rolling the reduction from metrics.parquet looked simple and was wrong: it
inflated offroad by up to 1.9x and dist_to_gt_trajectory by up to 4.9x against
the official numbers, by different factors per run, so interim tables built that
way could not be compared to anything. The aggregation applies at-fault
attribution, eval-relevant truncation and the scene-score gates, none of which
survive a groupby-and-mean.

Calling the real aggregation on whatever rollouts have finished avoids all of
that. The result is exact for the clips it covers; it is a subset of the scene
set, not an approximation of the full one.
"""

import argparse
import contextlib
import json
import pathlib
import sys

from omegaconf import OmegaConf

from eval.aggregation.main import _run_aggregation_core
from eval.schema import EvalConfig

METRICS = [
    "avg_dist_between_incidents_at_fault",
    "collision_at_fault",
    "collision_any",
    "offroad",
    "left_corridor_laterally",
    "wrong_lane",
    "dist_traveled_m",
    "dist_to_gt_trajectory",
    "duration_frac_20s",
]


def score(run_dir: pathlib.Path, tmp: pathlib.Path) -> dict:
    done = run_dir / "aggregate" / "results-summary.json"
    if done.exists():                      # the run finished; use its own summary
        return json.loads(done.read_text())
    cfg = OmegaConf.merge(
        OmegaConf.structured(EvalConfig), OmegaConf.load(run_dir / "eval-config.yaml")
    )
    cfg.video.render_video = False
    tmp.mkdir(parents=True, exist_ok=True)
    # Aggregation prints metric tables to stdout; keep this process's stdout
    # clean so the caller can parse it.
    with contextlib.redirect_stdout(sys.stderr):
        _run_aggregation_core([run_dir], tmp, cfg)
    return json.loads((tmp / "results-summary.json").read_text())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+", help="run directories to score")
    ap.add_argument("--tmp", default="/tmp/interim-score")
    args = ap.parse_args()

    out = {}
    for i, r in enumerate(args.runs):
        run = pathlib.Path(r)
        summary = score(run, pathlib.Path(args.tmp) / f"{i}-{run.name}")
        rollouts = [x for x in summary.get("rollouts", []) if x.get("score") is not None]
        m = summary["metrics_results"][0]
        out[run.name] = {
            "clips": len({x["clipgt_id"] for x in rollouts}),
            "rollouts": len(rollouts),
            "avg_scene_score": (
                sum(x["score"] for x in rollouts) / len(rollouts) if rollouts else None
            ),
            "zero_rate": (
                sum(1 for x in rollouts if x["score"] == 0.0) / len(rollouts)
                if rollouts
                else None
            ),
            **{k: m.get(k) for k in METRICS},
        }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
