# SPDX-License-Identifier: Apache-2.0
"""gRPC conformance check: replay a rollout.asl's driver traffic against a driver (alpasim .venv).

    UV_OFFLINE=1 .venv/bin/python e2e_challenge/drivor_eval/replay_asl.py ROLLOUT.asl \
        --address 127.0.0.1:6800 [--max-drives 40] [--report out.json]

Sends, in logged order, exactly the requests the runtime sent to its driver in
that rollout: start_session, every camera image, egomotion, route and drive,
then close_session. Open loop: the logged ego motion (another policy's) is
replayed, so this tests the service contract, not driving. Each DriveResponse
must be empty (not ready) or carry the current pose followed by 8 poses at
time_now + k * 0.5 s with finite values; the report lists every violation and
the predicted paths in the rig frame for inspection.
"""

import argparse
import asyncio
import json
import math
import pickle
import time

import grpc
from alpasim_grpc.v0 import common_pb2, egodriver_pb2, egodriver_pb2_grpc
from alpasim_utils.logs import async_read_pb_log


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


async def main(a):
    channel = grpc.aio.insecure_channel(a.address, options=[("grpc.max_send_message_length", 64 << 20),
                                                            ("grpc.max_receive_message_length", 64 << 20)])
    stub = egodriver_pb2_grpc.EgodriverServiceStub(channel)
    version = await stub.get_version(common_pb2.Empty())
    report = {"version": version.version_id, "drives": 0, "empty": 0, "violations": [], "paths": [],
              "latency_ms": []}
    uuid, last_pose, drives = None, None, 0
    async for e in async_read_pb_log(a.asl):
        kind = e.WhichOneof("log_entry")
        if kind == "driver_session_request":
            uuid = e.driver_session_request.session_uuid
            await stub.start_session(e.driver_session_request)
        elif uuid is None:
            continue
        elif kind == "driver_camera_image":
            await stub.submit_image_observation(e.driver_camera_image)
        elif kind == "driver_ego_trajectory":
            await stub.submit_egomotion_observation(e.driver_ego_trajectory)
            if e.driver_ego_trajectory.trajectory.poses:
                last_pose = e.driver_ego_trajectory.trajectory.poses[-1]
        elif kind == "route_request":
            await stub.submit_route(e.route_request)
        elif kind == "driver_request":
            req = e.driver_request
            t0 = time.time()
            try:
                resp = await stub.drive(req)
            except grpc.aio.AioRpcError as err:
                report.setdefault("drive_errors", []).append({"time_now_us": req.time_now_us,
                                                              "error": err.details()[:200]})
                drives += 1
                if a.max_drives and drives >= a.max_drives:
                    break
                continue
            report["latency_ms"].append(round(1000 * (time.time() - t0), 1))
            report["drives"] += 1
            poses = resp.trajectory.poses
            if not poses:
                report["empty"] += 1
            else:
                bad = []
                if len(poses) != 9:
                    bad.append(f"{len(poses)} poses")
                if poses[0] != last_pose:
                    bad.append("first pose is not the latest egomotion pose")
                want = [req.time_now_us + k * 500_000 for k in range(1, 9)]
                if [p.timestamp_us for p in poses[1:]] != want:
                    bad.append("timestamps")
                vals = [v for p in poses for v in (p.pose.vec.x, p.pose.vec.y, p.pose.vec.z, p.pose.quat.w, p.pose.quat.z)]
                if not all(math.isfinite(v) for v in vals):
                    bad.append("non-finite")
                if bad:
                    report["violations"].append({"time_now_us": req.time_now_us, "problems": bad})
                a0 = poses[0].pose
                c, s, yaw = math.cos(yaw_of(a0.quat)), math.sin(yaw_of(a0.quat)), yaw_of(a0.quat)
                rig = [[round(c * (p.pose.vec.x - a0.vec.x) + s * (p.pose.vec.y - a0.vec.y), 2),
                        round(-s * (p.pose.vec.x - a0.vec.x) + c * (p.pose.vec.y - a0.vec.y), 2),
                        round(math.degrees(yaw_of(p.pose.quat) - yaw), 1)] for p in poses[1:]]
                report["paths"].append({"time_now_us": req.time_now_us, "rig_xy_yawdeg": rig,
                                        "debug": pickle.loads(resp.debug_info.unstructured_debug_info)
                                        if resp.debug_info.unstructured_debug_info else None})
            drives += 1
            if a.max_drives and drives >= a.max_drives:
                break
    if uuid is not None:
        await stub.close_session(egodriver_pb2.DriveSessionCloseRequest(session_uuid=uuid))
    lat = sorted(report["latency_ms"][report["empty"]:]) or [0]
    report["latency_ms_p50_max"] = [lat[len(lat) // 2], lat[-1]]
    summary = {k: v for k, v in report.items() if k not in ("paths", "latency_ms", "drive_errors")}
    summary["n_drive_errors"] = len(report.get("drive_errors", []))
    summary["first_errors"] = report.get("drive_errors", [])[:3]
    print(json.dumps(summary, indent=1))
    for p in report["paths"][:: max(1, len(report["paths"]) // 5)]:
        print(p["time_now_us"], p["rig_xy_yawdeg"][::2])
    if a.report:
        with open(a.report, "w") as f:
            json.dump(report, f, indent=1)
    await channel.close()
    return 1 if report["violations"] or report.get("drive_errors") else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("asl")
    ap.add_argument("--address", required=True)
    ap.add_argument("--max-drives", type=int, default=0)
    ap.add_argument("--report")
    raise SystemExit(asyncio.run(main(ap.parse_args())))
