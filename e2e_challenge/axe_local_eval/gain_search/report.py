"""Print every controller-gain measurement taken so far as one ranked table.

The runs come from two searches -- ep24 under the ec2 preset, ep29 under dev --
but splitting the report by search makes the reader merge two rankings by hand
to answer the only question that matters, which is what the best gain set is
right now. So they share one table, ordered by score, and the search each row
came from is a column rather than a heading.

The scores are not interchangeable across those two searches: a different
checkpoint and a different preset both move the absolute number. The `src`
column is what carries that warning, and comparisons within one `src` are the
ones that mean something.
"""

import json
import pathlib
import statistics

HERE = pathlib.Path(__file__).parent
GAINS = [
    "lat_position_weight",
    "long_position_weight",
    "idx_start_penalty",
    "heading_weight",
    "rel_front_steering_angle_weight",
    "rel_acceleration_weight",
    "acceleration_weight",
]
SOURCES = [
    ("ep24", HERE / "results.jsonl", "ec2", (6.0, 0.5, 3)),
    ("ep29", HERE / "ep29/results.jsonl", "dev", (3.0, 0.25, 3)),
]


def collect():
    """Unique (source, gain set) -> mean score over its repeats."""
    out = {}
    failed = {}
    for src, path, preset, anchor in SOURCES:
        rows = (
            [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
            if path.exists()
            else []
        )
        failed[src] = sum(1 for r in rows if r.get("score") is None)
        for r in rows:
            if r.get("score") is None:
                continue
            key = (src, json.dumps({k: r["gains"][k] for k in sorted(GAINS)}, sort_keys=True))
            e = out.setdefault(
                key,
                {"src": src, "preset": preset, "anchor": anchor,
                 "gains": r["gains"], "tag": r["tag"], "scores": []},
            )
            e["scores"].append(r["score"])
    for e in out.values():
        e["n"] = len(e["scores"])
        e["mean"] = statistics.fmean(e["scores"])
    return list(out.values()), failed


def main() -> None:
    rows, failed = collect()
    rows.sort(key=lambda e: -e["mean"])

    head = (f"{'#':>3} {'src':<5}{'tag':<22}{'mean':>8}{'n':>3}   "
            f"{'lat':>4}{'lon':>6}{'idx':>4}{'hdg':>5}{'rst':>5}{'rac':>5}{'acc':>5}")
    print(head)
    print("-" * len(head))
    for i, e in enumerate(rows, 1):
        g = e["gains"]
        here = (g["lat_position_weight"], g["long_position_weight"], g["idx_start_penalty"])
        mark = "  <- anchor" if here == e["anchor"] else ""
        print(
            f"{i:>3} {e['src']:<5}{e['tag']:<22}{e['mean']:>8.4f}{e['n']:>3}   "
            f"{g['lat_position_weight']:>4g}{g['long_position_weight']:>6g}"
            f"{g['idx_start_penalty']:>4g}{g['heading_weight']:>5g}"
            f"{g['rel_front_steering_angle_weight']:>5g}"
            f"{g['rel_acceleration_weight']:>5g}{g['acceleration_weight']:>5g}{mark}"
        )

    print()
    for src, _, preset, anchor in SOURCES:
        n = sum(1 for e in rows if e["src"] == src)
        runs = sum(e["n"] for e in rows if e["src"] == src)
        print(f"  {src}: {runs} runs / {n} unique, preset={preset}"
              + (f", {failed[src]} failed" if failed.get(src) else ""))


if __name__ == "__main__":
    main()
