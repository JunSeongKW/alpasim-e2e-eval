# SPDX-License-Identifier: Apache-2.0
"""DrivoR egodriver for the AlpaSim e2e challenge PAI track (drivor-eval env, py3.9).

    $ENV/bin/python drivor_driver.py --policy ours_hugsim_final --port 6800 \
        --device cuda:0 --precision fp16 --log-dir runs/<run>/drivers/<name>

One process = one model on one GPU, serving the official EgodriverService
(alpasim_grpc protos as generated in src/grpc, imported unchanged). The model
is mingyu's protected bundle (DRIVOR_ALPASIM_NUREC441_HANDOFF_20261008_2005):
runtime/policy.py loads and verifies the checkpoints and its own canonical
NAVSIM feature builder (build_model_inputs) turns our AgentInput into tensors,
so resize / normalisation / slot order are exactly the training ones.

What this adapter adds, identically for both policies (frozen before any score):
  cameras   pai_views.py: the six PAI f-theta renders -> nuPlan cam_f0, cam_b0,
            cam_l0, cam_r0 (1920x1080, distorted), from each session's own
            available_cameras calibration.
  command   alpasim_driver.navigation.determine_command_from_route, with the
            official VaVAM / Alpamayo route settings (threshold 3 m, lookahead
            20 m; dev preset routes start 40 m ahead). calibrate_command.py
            shows 3 m is within 0.005 balanced accuracy of the best setting
            against NAVSIM's training command.
  ego       NAVSIM ego_status rows (pose relative to now, velocity and
            acceleration in the rear-axle frame from DynamicState, command
            one-hot [left, straight, right, unknown]), 4 rows at 2 Hz.
  output    the selected 8 x (x, y, heading) at 0.5 .. 4.0 s, anchored on the
            latest egomotion pose and stamped time_now + k * 0.5 s, the current
            pose prepended -- as alpasim_driver.main converts predictions. No
            smoothing, interpolation or extrapolation; the controller
            interpolates. A failure raises (gRPC error); there is no fallback.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import pickle
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent import futures
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ALPASIM = HERE.parents[1]
KIT_DEFAULT = ALPASIM.parent / "mingyu/DRIVOR_ALPASIM_NUREC441_HANDOFF_20261008_2005"

# Route -> command: official navigation rule with the VaVAM / Alpamayo settings.
COMMAND_THRESHOLD_M = 3.0
MIN_LOOKAHEAD_M = 20.0
LEFT, STRAIGHT, RIGHT, UNKNOWN = 0, 1, 2, 3        # alpasim_driver DriveCommand == NAVSIM one-hot index
HISTORY_STEPS_US = (1_500_000, 1_000_000, 500_000, 0)  # NAVSIM: 4 frames at 2 Hz, last = now
OUTPUT_DT_US = 500_000
# --ego-clamp navsim: ego velocity / acceleration limited to the range DrivoR
# saw in training (min / max over 227k NAVSIM trainval frames, 400 logs). The
# simulator's first step carries an initialisation spike (e.g. -21.6 m/s^2) and
# MPC braking reaches -9 m/s^2, outside NAVSIM's -5.2 .. 3.4.
NAVSIM_RANGE = {"vx": (-2.03, 19.56), "vy": (-0.48, 0.41), "ax": (-5.15, 3.41), "ay": (-4.16, 5.57)}


def setup_paths(kit: Path) -> None:
    sys.path[:0] = [str(kit / "runtime"), str(HERE), str(HERE / "grpc_meta"), str(ALPASIM / "src/grpc")]
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TORCH_HOME", str(kit / "runtime/offline_empty_cache"))
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/drivor-mpl")


def yaw_of(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def command_from_route(waypoints) -> int:
    """alpasim_driver.navigation.determine_command_from_route, verbatim logic."""
    if len(waypoints) < 1:
        return UNKNOWN
    target = None
    for wp in waypoints:
        if math.hypot(wp.x, wp.y) >= MIN_LOOKAHEAD_M:
            target = wp
            break
    if target is None:
        return STRAIGHT
    if target.y > COMMAND_THRESHOLD_M:
        return LEFT
    if target.y < -COMMAND_THRESHOLD_M:
        return RIGHT
    return STRAIGHT


class Session:
    def __init__(self, uuid: str, scene: str, specs: dict, views):
        self.uuid = uuid
        self.scene = scene
        self.specs = specs
        self.views = views
        self.latest = {}          # logical_id -> (frame_end_us, jpeg bytes)
        self.poses = {}           # t_us -> PoseAtTime (local frame)
        self.dynamic = {}         # t_us -> (vx, vy, ax, ay) rig frame
        self.command = STRAIGHT   # alpasim_driver Session default until a route arrives
        self.counts = Counter()
        self.lock = threading.Lock()
        self.started = time.time()


class DrivorDriver:
    def __init__(self, args, pb):
        import numpy as np
        import torch
        from pai_views import NuplanRig
        from policy import build_model_inputs, load_policy

        self.np, self.torch, self.pb = np, torch, pb
        self.args = args
        self.build_model_inputs = build_model_inputs
        self.sessions = {}
        self.model_lock = threading.Lock()
        self.log_dir = Path(args.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.rig = NuplanRig()
        t0 = time.time()
        self.rays = self.rig.pixel_rays()
        logging.info("nuPlan pixel rays ready in %.1fs", time.time() - t0)
        t0 = time.time()
        self.policy = load_policy(args.policy, args.device, args.precision)
        logging.info("policy %s on %s/%s loaded in %.1fs; load audit %s", args.policy, args.device,
                     args.precision, time.time() - t0, json.dumps(self.policy.load_audit, default=str)[:2000])
        kit = Path(args.kit).resolve()
        outside = [m for m, mod in list(sys.modules.items())
                   if m.split(".")[0] in ("policy", "drivor", "navsim", "nuplan", "omegaconf")
                   and getattr(mod, "__file__", None) and kit not in Path(mod.__file__).resolve().parents]
        if outside:
            raise RuntimeError(f"model code imported from outside the bundle: {outside[:5]}")
        # Static camera metadata for the AgentInput (not read by the network,
        # INPUT_CONTRACT.json, but kept consistent with the synthesised views).
        self.camera_meta = {}
        for name in ("cam_f0", "cam_b0", "cam_l0", "cam_r0", "cam_l1", "cam_l2", "cam_r1", "cam_r2"):
            self.camera_meta[name] = dict(intrinsics=self.rig.K.copy(),
                                          sensor2lidar_rotation=self.rig.rotation(name),
                                          sensor2lidar_translation=self.rig.position(name))
        self.version = (f"drivor-{args.policy}-{args.precision}-clamp{args.ego_clamp}-pai-adapter-"
                        + subprocess.run(["git", "-C", str(ALPASIM), "rev-parse", "--short", "HEAD"],
                                         capture_output=True, text=True).stdout.strip())
        logging.info("policy load complete: %s", self.version)

    # ---- gRPC methods --------------------------------------------------------------------
    def start_session(self, request, context):
        from google.protobuf.json_format import MessageToDict
        from pai_views import SessionViews

        if request.session_uuid in self.sessions:
            context.abort(self.grpc.StatusCode.ALREADY_EXISTS, f"session {request.session_uuid} exists")
        specs = {c.logical_id: MessageToDict(c, preserving_proto_field_name=True)
                 for c in request.rollout_spec.vehicle.available_cameras}
        scene = request.debug_info.scene_id if request.HasField("debug_info") else request.session_uuid
        views = SessionViews(specs, self.rig, self.rays)   # raises if a needed camera is missing
        self.sessions[request.session_uuid] = Session(request.session_uuid, scene, specs, views)
        logging.info("start session %s scene %s cameras %s", request.session_uuid, scene, sorted(specs))
        return self.pb.common.SessionRequestStatus()

    def close_session(self, request, context):
        s = self.sessions.pop(request.session_uuid, None)
        if s is None:
            context.abort(self.grpc.StatusCode.NOT_FOUND, f"session {request.session_uuid} unknown")
        receipt = {"session": s.uuid, "scene": s.scene, "policy": self.args.policy, "version": self.version,
                   "wall_s": round(time.time() - s.started, 1), **s.counts,
                   "coverage": {t: round(v.coverage, 4) for t, v in (s.views.views or {}).items()},
                   "edge_filled": {t: round(v.edge_filled, 4) for t, v in (s.views.views or {}).items()}}
        with (self.log_dir / "sessions.jsonl").open("a") as f:
            f.write(json.dumps(receipt) + "\n")
        logging.info("close session %s %s", s.uuid, json.dumps(receipt))
        return self.pb.common.Empty()

    def get_version(self, request, context):
        from alpasim_grpc import API_VERSION_MESSAGE
        return self.pb.common.VersionId(version_id=self.version, git_hash=self.version.rsplit("-", 1)[-1],
                                        grpc_api_version=API_VERSION_MESSAGE)

    def submit_image_observation(self, request, context):
        s = self._session(request.session_uuid, context)
        img = request.camera_image
        with s.lock:
            prev = s.latest.get(img.logical_id)
            if prev is None or img.frame_end_us >= prev[0]:
                s.latest[img.logical_id] = (img.frame_end_us, img.image_bytes)
            s.counts["images"] += 1
        return self.pb.common.Empty()

    def submit_egomotion_observation(self, request, context):
        s = self._session(request.session_uuid, context)
        poses = request.trajectory.poses
        if len(poses) != len(request.dynamic_states):
            context.abort(self.grpc.StatusCode.INVALID_ARGUMENT, "poses / dynamic_states length mismatch")
        with s.lock:
            for pose, ds in zip(poses, request.dynamic_states):
                s.poses[pose.timestamp_us] = pose
                s.dynamic[pose.timestamp_us] = (ds.linear_velocity.x, ds.linear_velocity.y,
                                                ds.linear_acceleration.x, ds.linear_acceleration.y)
        return self.pb.common.Empty()

    def submit_route(self, request, context):
        s = self._session(request.session_uuid, context)
        with s.lock:
            # alpasim_driver.Session.update_command_from_route: keep the previous
            # command when the route is empty or no pose has arrived yet.
            if len(request.route.waypoints) >= 1 and s.poses:
                s.command = command_from_route(request.route.waypoints)
            s.counts[f"route_cmd_{s.command}"] += 1
        return self.pb.common.Empty()

    def submit_recording_ground_truth(self, request, context):
        return self.pb.common.Empty()

    def drive(self, request, context):
        s = self._session(request.session_uuid, context)
        np, pb = self.np, self.pb
        from pai_views import NEEDED_CAMERAS
        with s.lock:
            ready = all(c in s.latest for c in NEEDED_CAMERAS) and s.poses
            if not ready:
                s.counts["drive_not_ready"] += 1
                return pb.egodriver.DriveResponse(trajectory=pb.common.Trajectory())
            frames = {c: s.latest[c] for c in NEEDED_CAMERAS}
            t_anchor = max(s.poses)
            anchor = s.poses[t_anchor]
            command = s.command
            ego_rows = self._ego_rows(s, t_anchor, command)
        t0 = time.time()
        try:
            from PIL import Image
            images = {c: np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for c, (_, b) in frames.items()}
            views = s.views.render_all(images)
            cams = {name: SimpleNamespace(image=views.get(name), **meta) for name, meta in self.camera_meta.items()}
            agent_input = SimpleNamespace(cameras=[SimpleNamespace(**cams)], ego_statuses=ego_rows)
            inputs = self.build_model_inputs(agent_input, self.policy)
            t1 = time.time()
            with self.model_lock:
                out = self.policy.predict_from_model_inputs(inputs)
                traj = out["trajectory"][0].float().cpu().numpy()
                query = int(out["selected_query"][0])
            t2 = time.time()
        except Exception:
            s.counts["errors"] += 1
            logging.exception("drive failed for session %s at %s", s.uuid, request.time_now_us)
            if "inputs" in locals() and s.counts["errors"] <= 3:
                dump = self.log_dir / f"failed_{s.scene}_{request.time_now_us}.pt"
                self.torch.save({k: v.cpu() for k, v in inputs.items()}, dump)
                logging.info("saved failing model inputs to %s", dump)
            raise
        if traj.shape != (8, 3) or not np.isfinite(traj).all():
            s.counts["errors"] += 1
            context.abort(self.grpc.StatusCode.INTERNAL, f"bad model output {traj.shape}")
        response = pb.egodriver.DriveResponse(trajectory=self._to_local(traj, anchor, request.time_now_us))
        stale = request.time_now_us - min(t for t, _ in frames.values())
        with s.lock:
            s.counts["forwards"] += 1
            s.counts[f"cmd_{command}"] += 1
            s.counts["max_image_age_us"] = max(s.counts["max_image_age_us"], int(stale))
            s.counts["max_pose_age_us"] = max(s.counts["max_pose_age_us"], int(request.time_now_us - t_anchor))
            s.counts["prep_ms_total"] += int(1000 * (t1 - t0))
            s.counts["model_ms_total"] += int(1000 * (t2 - t1))
        response.debug_info.unstructured_debug_info = pickle.dumps(
            {"command": command, "selected_query": query, "prep_ms": round(1000 * (t1 - t0), 1),
             "model_ms": round(1000 * (t2 - t1), 1)}, protocol=4)
        return response

    # ---- helpers ---------------------------------------------------------------------------
    def _session(self, uuid, context):
        s = self.sessions.get(uuid)
        if s is None:
            context.abort(self.grpc.StatusCode.NOT_FOUND, f"session {uuid} unknown")
        return s

    def _ego_rows(self, s, t_now, command):
        """NAVSIM ego_statuses: 4 frames at 2 Hz ending now; poses relative to now."""
        np = self.np
        times = sorted(s.poses)
        cur = s.poses[t_now].pose
        cx, cy, cyaw = cur.vec.x, cur.vec.y, yaw_of(cur.quat)
        c, si = math.cos(cyaw), math.sin(cyaw)
        onehot = np.zeros(4, dtype=np.float32)
        onehot[command] = 1.0
        rows = []
        for back in HISTORY_STEPS_US:
            want = t_now - back
            # nearest recorded time not after `want`; before the session's first
            # pose, the first pose (only the last row reaches the network).
            t = max([x for x in times if x <= want], default=times[0])
            p = s.poses[t].pose
            dx, dy = p.vec.x - cx, p.vec.y - cy
            rel = np.array([c * dx + si * dy, -si * dx + c * dy,
                            math.atan2(math.sin(yaw_of(p.quat) - cyaw), math.cos(yaw_of(p.quat) - cyaw))])
            vx, vy, ax, ay = s.dynamic[t]
            if self.args.ego_clamp == "navsim":
                raw = (vx, vy, ax, ay)
                vx, vy, ax, ay = (min(max(val, NAVSIM_RANGE[k][0]), NAVSIM_RANGE[k][1])
                                  for k, val in zip(("vx", "vy", "ax", "ay"), raw))
                if back == 0 and (vx, vy, ax, ay) != raw:
                    s.counts["ego_clamped"] += 1
            rows.append(SimpleNamespace(ego_pose=rel.astype(np.float32),
                                        ego_velocity=np.array([vx, vy], dtype=np.float32),
                                        ego_acceleration=np.array([ax, ay], dtype=np.float32),
                                        driving_command=onehot.copy()))
        return rows

    def _to_local(self, traj, anchor, time_now_us):
        """alpasim_driver.main._convert_prediction_to_alpasim_trajectory for 2 Hz output."""
        pb = self.pb
        out = pb.common.Trajectory()
        out.poses.append(anchor)
        yaw = yaw_of(anchor.pose.quat)
        c, s = math.cos(yaw), math.sin(yaw)
        x0, y0, z0 = anchor.pose.vec.x, anchor.pose.vec.y, anchor.pose.vec.z
        for k, (x, y, h) in enumerate(traj.tolist(), start=1):
            gy = yaw + h
            out.poses.append(pb.common.PoseAtTime(
                pose=pb.common.Pose(vec=pb.common.Vec3(x=x0 + c * x - s * y, y=y0 + s * x + c * y, z=z0),
                                    quat=pb.common.Quat(w=math.cos(gy / 2), x=0.0, y=0.0, z=math.sin(gy / 2))),
                timestamp_us=time_now_us + k * OUTPUT_DT_US))
        return out


def make_servicer(driver, pb):
    base = pb.egodriver_grpc.EgodriverServiceServicer
    methods = ("start_session", "close_session", "submit_image_observation", "submit_egomotion_observation",
               "submit_route", "submit_recording_ground_truth", "drive", "get_version")
    return type("DrivorServicer", (base,), {m: (lambda self, r, c, _m=m: getattr(driver, _m)(r, c))
                                            for m in methods})()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", required=True, choices=["drivor_original", "ours_hugsim_final"])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--kit", default=str(KIT_DEFAULT))
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ego-clamp", default="none", choices=["none", "navsim"])
    args = ap.parse_args()
    setup_paths(Path(args.kit))
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(Path(args.log_dir) / "driver.log"),
                                  logging.StreamHandler()])
    import cv2
    import grpc
    import torch
    from alpasim_grpc.v0 import common_pb2, egodriver_pb2, egodriver_pb2_grpc

    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    pb = SimpleNamespace(common=common_pb2, egodriver=egodriver_pb2, egodriver_grpc=egodriver_pb2_grpc)
    driver = DrivorDriver(args, pb)
    driver.grpc = grpc
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8),
                         options=[("grpc.max_receive_message_length", 64 << 20),
                                  ("grpc.max_send_message_length", 64 << 20)])
    egodriver_pb2_grpc.add_EgodriverServiceServicer_to_server(make_servicer(driver, pb), server)
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()
    logging.info("serving %s on %s:%d", driver.version, args.host, args.port)
    server.wait_for_termination()


if __name__ == "__main__":
    main()
