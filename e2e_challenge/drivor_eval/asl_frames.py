# SPDX-License-Identifier: Apache-2.0
"""Pull the driver's inputs for a few steps out of a rollout.asl (alpasim .venv).

    UV_OFFLINE=1 .venv/bin/python e2e_challenge/drivor_eval/asl_frames.py ROLLOUT.asl OUT_DIR --steps 20 100

Writes OUT_DIR/session_request.json (the DriveSessionRequest, incl.
available_cameras) and OUT_DIR/step<k>/<logical_id>.jpg: the k-th image the
driver received from each camera, bytes exactly as sent. Used for offline
render checks of the PAI -> nuPlan view mapping without starting a simulation.
"""

import argparse
import asyncio
import json
from pathlib import Path

from alpasim_utils.logs import async_read_pb_log
from google.protobuf.json_format import MessageToDict


async def main(asl: str, out: Path, steps):
    out.mkdir(parents=True, exist_ok=True)
    seen = {}
    async for entry in async_read_pb_log(asl):
        kind = entry.WhichOneof("log_entry")
        if kind == "driver_session_request":
            d = MessageToDict(entry.driver_session_request, preserving_proto_field_name=True)
            (out / "session_request.json").write_text(json.dumps(d, indent=1) + "\n")
        elif kind == "driver_camera_image":
            image = entry.driver_camera_image.camera_image
            k = seen.get(image.logical_id, 0)
            if k in steps:
                (out / f"step{k:03d}").mkdir(exist_ok=True)
                (out / f"step{k:03d}" / f"{image.logical_id}.jpg").write_bytes(image.image_bytes)
            seen[image.logical_id] = k + 1
    print(out, seen)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("asl")
    ap.add_argument("out", type=Path)
    ap.add_argument("--steps", type=int, nargs="+", default=[20, 100])
    a = ap.parse_args()
    asyncio.run(main(a.asl, a.out, set(a.steps)))
