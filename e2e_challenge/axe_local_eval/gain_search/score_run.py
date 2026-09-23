"""Score one gain-search run with the stratified estimator the subsets were built for.

The subset over-samples clips that vary and under-samples the ones that always
score 1.0, so a plain mean over it is not the full-set mean. Weighting each
stratum back to its population share is, and it is what makes numbers from this
subset comparable to the 441-clip runs.

Prints one JSON line so the search loop can append it to results.jsonl without
parsing prose.
"""

import argparse
import collections
import contextlib
import json
import math
import pathlib
import statistics
import sys

from omegaconf import OmegaConf

from eval.aggregation.main import _run_aggregation_core
from eval.schema import EvalConfig

HERE = pathlib.Path(__file__).parent


def aggregate(run_dir: pathlib.Path, out: pathlib.Path) -> dict:
    """Re-score with AlpaSim's own aggregation when the wizard left none."""
    done = run_dir / "aggregate" / "results-summary.json"
    if done.exists():
        return json.loads(done.read_text())
    schema = OmegaConf.structured(EvalConfig)
    cfg = OmegaConf.merge(schema, OmegaConf.load(run_dir / "eval-config.yaml"))
    cfg.video.render_video = False
    out.mkdir(parents=True, exist_ok=True)
    # Aggregation prints metric tables to stdout. The caller parses this
    # process's stdout as a single JSON object, so send that chatter to stderr;
    # otherwise every run the wizard did not finish itself is scored as failed.
    with contextlib.redirect_stdout(sys.stderr):
        _run_aggregation_core([run_dir], out, cfg)
    return json.loads((out / "results-summary.json").read_text())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", required=True)
    p.add_argument("--subset", default="search", choices=("search", "holdout"))
    p.add_argument("--tmp", default="/tmp/gain-score")
    args = p.parse_args()

    meta = json.loads((HERE / "subsets.json").read_text())
    weights = meta["stratum_weights"]
    stratum_of = {
        clip: name for name, ids in meta[args.subset].items() for clip in ids
    }

    summary = aggregate(pathlib.Path(args.run), pathlib.Path(args.tmp))

    per_stratum = collections.defaultdict(list)
    reasons = collections.Counter()
    seen = set()
    for r in summary.get("rollouts", []):
        if r.get("score") is None:
            continue
        clip = r["clipgt_id"]
        name = stratum_of.get(clip)
        if name is None:            # a clip outside the subset; ignore it
            continue
        seen.add(clip)
        per_stratum[name].append(r["score"])
        reasons[r.get("failure_reason") or "(pass)"] += 1

    if not per_stratum:
        print(json.dumps({"error": "no scored rollout matched the subset"}))
        return 1

    # Weighted mean, and its standard error from the within-stratum variances.
    # Strata the run has not reached yet are dropped and the remaining weights
    # renormalised, so a partial run still reports a comparable number.
    present = {k: weights[k] for k in per_stratum}
    wsum = sum(present.values())
    mean = sum(w / wsum * statistics.fmean(per_stratum[k]) for k, w in present.items())
    var = 0.0
    for k, w in present.items():
        v = per_stratum[k]
        if len(v) > 1:
            var += (w / wsum) ** 2 * statistics.variance(v) / len(v)
    se = math.sqrt(var)

    flat = [s for v in per_stratum.values() for s in v]
    out = {
        "run": pathlib.Path(args.run).name,
        "subset": args.subset,
        "clips": len(seen),
        "rollouts": len(flat),
        "score": round(mean, 5),
        "se": round(se, 5),
        "unweighted": round(statistics.fmean(flat), 5),
        "full_rate": round(sum(1 for s in flat if s >= 0.999) / len(flat), 4),
        "zero_rate": round(sum(1 for s in flat if s == 0.0) / len(flat), 4),
        "reasons": dict(reasons),
        "per_stratum": {
            k: {"n": len(v), "mean": round(statistics.fmean(v), 4)}
            for k, v in sorted(per_stratum.items())
        },
    }
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
