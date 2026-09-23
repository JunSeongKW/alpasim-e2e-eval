"""Attach extracted NuRec maps and expert routes to existing NAVSIM log pickles."""

from __future__ import annotations

import argparse
import json
import pickle
import random
from bisect import bisect_left
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


# A prepared NuRec frame is 0.5 s. Beyond half a frame the nearest map ego
# sample is ambiguous, so the attached route suffix can belong to a different
# part of the clip. That corrupts ego_progress and driving_direction_compliance
# in EPDMS without any visible error, so refuse to attach instead of reporting
# a large number nobody reads. Pass a non-positive value to disable the check.
DEFAULT_MAX_TIMESTAMP_ERROR_US = 250_000

# A clip whose ego never matches a lane yields an empty route, and
# scene_filter.has_route drops every one of its windows anyway. Dropping such
# a log keeps the prepared set scoreable instead of failing validation on data
# that can never produce a token. Stop outright if many logs are affected,
# because that means map extraction itself regressed rather than a few clips
# leaving the reconstructed area.
DEFAULT_MAX_ROUTELESS_FRACTION = 0.01

# _route_suffixes holds the last known route suffix across ego poses that match
# no lane. That is right for a short gap at the head or tail of a clip, but a
# clip matched only sporadically carries a stale route over most of its length,
# scoring ego_progress and driving_direction_compliance against a centerline the
# ego has long left. Observed clips sit either above 75% matched or below 20%,
# so half is a wide margin on both sides.
DEFAULT_MIN_EGO_MATCH_FRACTION = 0.5


class RouteAlignmentError(ValueError):
    """Raised when no map ego sample is close enough to a log frame."""


def _clip_id(frames: Sequence[Mapping[str, Any]], source: Path) -> str:
    first = frames[0]
    for value in (first.get("clip_id"), first.get("log_name"), source.stem):
        if value:
            value = str(value)
            return value[len("nurec-") :] if value.startswith("nurec-") else value
    raise ValueError(f"cannot infer NuRec clip id from {source}")


def _route_suffixes(bundle: Mapping[str, Any]) -> Tuple[List[int], List[List[str]]]:
    route = list(bundle["route_roadblock_ids"])
    timestamps: List[int] = []
    suffixes: List[List[str]] = []
    route_index = 0
    for match in bundle["matched_ego"]:
        roadblock_id = match.get("roadblock_id")
        if roadblock_id:
            try:
                route_index = route.index(roadblock_id, route_index)
            except ValueError:
                # Revisited roadblocks are uncommon; retaining the current
                # suffix is safer than jumping backwards in the route.
                pass
        timestamps.append(int(match["timestamp_us"]))
        suffixes.append(route[route_index:])
    return timestamps, suffixes


def attach_map(
    frames: List[Dict[str, Any]],
    bundle: Mapping[str, Any],
    max_timestamp_error_us: Optional[int] = DEFAULT_MAX_TIMESTAMP_ERROR_US,
) -> Dict[str, int]:
    timestamps, suffixes = _route_suffixes(bundle)
    if not timestamps:
        raise ValueError(f"{bundle['clip_id']}: map bundle has no ego timestamps")
    observed_error_us = 0
    frames_with_route = 0
    for frame in frames:
        timestamp = int(frame["timestamp"])
        index = bisect_left(timestamps, timestamp)
        candidates = [candidate for candidate in (index - 1, index) if 0 <= candidate < len(timestamps)]
        nearest = min(candidates, key=lambda candidate: abs(timestamps[candidate] - timestamp))
        error = abs(timestamps[nearest] - timestamp)
        if max_timestamp_error_us and max_timestamp_error_us > 0 and error > max_timestamp_error_us:
            raise RouteAlignmentError(
                f"{bundle['clip_id']}: frame timestamp {timestamp} is {error}us from the nearest map "
                f"ego sample, above the {max_timestamp_error_us}us limit; the route suffix would be unreliable"
            )
        observed_error_us = max(observed_error_us, error)
        frame["map_location"] = bundle["map_name"]
        frame["roadblock_ids"] = list(suffixes[nearest])
        frames_with_route += bool(frame["roadblock_ids"])
    matched_ego = bundle["matched_ego"]
    num_matched = sum(1 for match in matched_ego if match.get("roadblock_id"))
    return {
        "frames": len(frames),
        "frames_with_route": frames_with_route,
        "max_timestamp_error_us": observed_error_us,
        "ego_match_fraction": num_matched / len(matched_ego) if matched_ego else 0.0,
    }


def process_logs(
    input_root: Path,
    map_root: Path,
    output_root: Path,
    max_timestamp_error_us: Optional[int] = DEFAULT_MAX_TIMESTAMP_ERROR_US,
    min_ego_match_fraction: float = DEFAULT_MIN_EGO_MATCH_FRACTION,
) -> Dict[str, Any]:
    if input_root.resolve() == output_root.resolve():
        raise ValueError("input and output roots must differ; attach-maps never overwrites source logs")
    files = sorted(input_root.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"no .pkl logs found in {input_root}")
    summary = {
        "input_logs": len(files),
        "logs": 0,
        "log_names": [],
        "frames": 0,
        "frames_with_route": 0,
        "missing_maps": [],
        "routeless_maps": [],
        "unreliable_route_maps": [],
        "max_timestamp_error_us": 0,
        "max_timestamp_error_limit_us": max_timestamp_error_us,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    for source in files:
        with source.open("rb") as stream:
            frames = pickle.load(stream)
        clip_id = _clip_id(frames, source)
        map_path = map_root / f"{clip_id}.pkl"
        if not map_path.is_file():
            summary["missing_maps"].append(clip_id)
            continue
        with map_path.open("rb") as stream:
            bundle = pickle.load(stream)
        stats = attach_map(frames, bundle, max_timestamp_error_us)
        dropped = None
        if stats["frames_with_route"] == 0:
            dropped = "routeless_maps"
        elif stats["ego_match_fraction"] < min_ego_match_fraction:
            dropped = "unreliable_route_maps"
        if dropped:
            summary[dropped].append(clip_id)
            # Re-runs must not leave a previously written log behind, or the
            # prepared set would keep data the index no longer lists.
            (output_root / source.name).unlink(missing_ok=True)
            continue
        with (output_root / source.name).open("wb") as stream:
            pickle.dump(frames, stream, protocol=pickle.HIGHEST_PROTOCOL)
        summary["logs"] += 1
        summary["log_names"].append(source.stem)
        summary["frames"] += stats["frames"]
        summary["frames_with_route"] += stats["frames_with_route"]
        summary["max_timestamp_error_us"] = max(summary["max_timestamp_error_us"], stats["max_timestamp_error_us"])
    return summary


def write_index(
    summary: Mapping[str, Any],
    map_root: Path,
    sensor_root: Path,
    index_path: Path,
    val_fraction: float = 0.10,
    split_seed: int = 2026,
) -> None:
    """Write an all-train index with a sampled validation subset."""
    if not 0.0 <= val_fraction <= 1.0:
        raise ValueError("val_fraction must be between zero and one")
    names = sorted(summary["log_names"])
    num_val = int(round(len(names) * val_fraction))
    if names and val_fraction > 0:
        num_val = max(1, num_val)
    num_val = min(num_val, len(names))
    val = sorted(random.Random(split_seed).sample(names, num_val))
    index = {
        "format_version": 2,
        # All data remains in training. Validation is a reproducible random
        # subset of training rather than a held-out partition.
        "logs": {"train": names, "val": val},
        "validation_sampling": {
            "fraction": val_fraction,
            "seed": split_seed,
            "overlaps_train": True,
        },
        "num_logs": len(names),
        "num_frames": int(summary["frames"]),
        "frame_interval_us": 500_000,
        "sensor_root": str(sensor_root.resolve()),
        "map_root": str(map_root.resolve()),
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-log-root", type=Path, required=True)
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--output-log-root", type=Path, required=True)
    parser.add_argument("--index-path", type=Path)
    parser.add_argument("--sensor-root", type=Path)
    parser.add_argument("--val-fraction", type=float, default=0.10)
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument(
        "--max-timestamp-error-us",
        type=int,
        default=DEFAULT_MAX_TIMESTAMP_ERROR_US,
        help="Reject a clip whose frame is farther than this from the nearest map ego sample "
             "(non-positive disables the check)",
    )
    parser.add_argument(
        "--max-routeless-fraction",
        type=float,
        default=DEFAULT_MAX_ROUTELESS_FRACTION,
        help="Abort if more than this fraction of logs yield no usable route",
    )
    parser.add_argument(
        "--min-ego-match-fraction",
        type=float,
        default=DEFAULT_MIN_EGO_MATCH_FRACTION,
        help="Drop a log whose ego matches a lane in less than this fraction of its poses",
    )
    parser.add_argument("--allow-missing-maps", action="store_true")
    args = parser.parse_args()
    summary = process_logs(
        args.input_log_root,
        args.map_root,
        args.output_log_root,
        args.max_timestamp_error_us,
        args.min_ego_match_fraction,
    )
    unusable = summary["routeless_maps"] + summary["unreliable_route_maps"]
    if len(unusable) > int(summary["input_logs"] * args.max_routeless_fraction):
        raise ValueError(
            f"{len(unusable)} of {summary['input_logs']} logs have no usable route, above the "
            f"{args.max_routeless_fraction:.1%} limit; map extraction is likely broken. First={unusable[0]}"
        )
    if summary["missing_maps"] and not args.allow_missing_maps:
        first = summary["missing_maps"][0]
        raise FileNotFoundError(
            f"{len(summary['missing_maps'])} logs have no extracted map bundle; first={first}"
        )
    if args.index_path:
        if args.sensor_root is None:
            parser.error("--sensor-root is required when --index-path is used")
        write_index(
            summary,
            args.map_root,
            args.sensor_root,
            args.index_path,
            args.val_fraction,
            args.split_seed,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
