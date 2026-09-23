"""Merge generated route sidecars into prepared NuRec logs.

``nurec_route`` writes one sidecar per log so routes can be built while a
long-running job still reads the logs. This step folds them in, after which
``EgoStatus.route_waypoints`` is populated straight from the scene dict.

Run this only when nothing else is reading the log root: it rewrites every log.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np

from navsim.planning.data.nurec_route import ROUTE_SLOTS


def attach_routes(frames: list, routes: Dict[str, np.ndarray]) -> Dict[str, int]:
    attached = 0
    for frame in frames:
        waypoints = routes.get(str(frame["token"]))
        if waypoints is None:
            frame["route_waypoints"] = np.full((ROUTE_SLOTS, 3), np.nan, dtype=np.float32)
            continue
        waypoints = np.asarray(waypoints, dtype=np.float32)
        if waypoints.shape != (ROUTE_SLOTS, 3):
            raise ValueError(f"{frame['token']}: expected ({ROUTE_SLOTS}, 3), got {waypoints.shape}")
        frame["route_waypoints"] = waypoints
        attached += int(np.isfinite(waypoints[:, 0]).any())
    return {"frames": len(frames), "frames_with_route": attached}


def process_logs(log_root: Path, route_root: Path) -> Dict[str, Any]:
    files = sorted(log_root.glob("*.pkl"))
    if not files:
        raise FileNotFoundError(f"no prepared logs in {log_root}")
    summary: Dict[str, Any] = {"logs": 0, "frames": 0, "frames_with_route": 0, "missing_sidecars": []}
    for source in files:
        sidecar = route_root / source.name
        if not sidecar.is_file():
            summary["missing_sidecars"].append(source.stem)
            continue
        with source.open("rb") as stream:
            frames = pickle.load(stream)
        with sidecar.open("rb") as stream:
            routes = pickle.load(stream)
        stats = attach_routes(frames, routes)
        # Atomic: a log truncated mid-write would take out both the route and
        # everything else the training split needs from it.
        part = source.with_suffix(f".pkl.{os.getpid()}.part")
        with part.open("wb") as stream:
            pickle.dump(frames, stream, protocol=pickle.HIGHEST_PROTOCOL)
        part.replace(source)
        summary["logs"] += 1
        summary["frames"] += stats["frames"]
        summary["frames_with_route"] += stats["frames_with_route"]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--route-root", type=Path, required=True)
    args = parser.parse_args()
    summary = process_logs(args.log_root, args.route_root)
    if summary["missing_sidecars"]:
        raise FileNotFoundError(
            f"{len(summary['missing_sidecars'])} logs have no route sidecar; "
            f"first={summary['missing_sidecars'][0]}"
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
