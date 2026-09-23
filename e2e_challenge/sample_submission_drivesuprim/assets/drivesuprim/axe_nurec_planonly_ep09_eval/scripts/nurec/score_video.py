#!/usr/bin/env python3
"""Play back a scored clip: what the model drove, and the four terms it was graded on.

score_epdms.sh prints one average. This shows the frames that average is made of.
Each frame carries the cameras with the model's trajectory drawn on them, the
same trajectory rolled out through the controller in bird's eye, and the score
broken into NC x DAC x GT x EP -- so a zero can be read off the picture that
caused it rather than inferred from a column.

Runs off artefacts, not the model: the trajectory comes from the prediction
pickle the scoring job saved, and the scores from its CSV, so what is drawn is
exactly what was graded.

    source setup_nurec.sh
    python scripts/nurec/score_video.py \\
        --predictions <ckpt-dir>/<ckpt>.pkl \\
        --scores <exp>/<timestamp>.csv \\
        --log nurec-<clip> --out scored.mp4
"""
from __future__ import annotations

import argparse
import json
import lzma
import os
import shutil
import subprocess
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from shapely.geometry import LineString, Point, Polygon

from navsim.evaluate.pdm_score import get_trajectory_as_array, transform_trajectory
from navsim.planning.simulation.planner.nurec_controller.nurec_simulator import NuRecSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    GT_LATERAL_LIMIT_M,
    ROAD_EDGE_CONTACT_M,
    VEHICLE_CORNER_ROUNDNESS,
    _round_corners,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    state_array_to_coords_array,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import BBCoordsIndex, StateIndex
from scripts.nurec.verify_rectified_projection import lidar2img, project

CAMS = ("CAM_L0", "CAM_F0", "CAM_R0")
from navsim.planning.data.nurec_rectify.target import (
    TARGET_HEIGHT, TARGET_K, TARGET_WIDTH,
)

RECTIFIED_WH = (1920, 1080)   # the source render, for un-rectified logs only
"""Resolution `cam_intrinsic` is expressed in.

The images on disk are 512x256, which is that resolution squashed -- the
aspect is not preserved, so the projection has to be scaled per axis or the
path lands off-frame. calibration.json is read for the real number and this
is only the fallback."""
TERMS = ("no_at_fault_collisions", "drivable_area_compliance", "gt_compliance", "ego_progress")
TERM_LABEL = {
    "no_at_fault_collisions": "NC",
    "drivable_area_compliance": "DAC",
    "gt_compliance": "GT",
    "ego_progress": "EP",
}

# BGR, since OpenCV
INK = (232, 232, 232)
DIM = (150, 150, 150)
GROUND = (30, 30, 32)
PANEL = (24, 24, 26)
LANE = (58, 58, 62)
LANE_EDGE = (78, 78, 84)
ROAD_EDGE = (120, 120, 132)
ACTOR = (108, 116, 122)
HUMAN = (150, 225, 150)
MODEL = (70, 140, 255)
BAND = (96, 150, 96)
LEGEND_BAND = (120, 180, 120)
BAD = (70, 70, 235)
CULPRIT = (80, 80, 255)
GOOD = (140, 210, 140)
F = cv2.FONT_HERSHEY_SIMPLEX


FFMPEG_CANDIDATES = (
    "ffmpeg",
    "/home1/irteam/miniconda3/envs/catk/bin/ffmpeg",
    "/home1/irteam/miniconda3/envs/planr1/bin/ffmpeg",
)


def _find_ffmpeg() -> Optional[str]:
    for candidate in FFMPEG_CANDIDATES:
        path = shutil.which(candidate) if os.path.basename(candidate) == candidate else candidate
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


def _h264_encoder(ffmpeg: str) -> Optional[str]:
    """libx264 if the build has it, otherwise OpenH264. Both play in a browser."""
    try:
        listed = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=30, check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for name in ("libx264", "libopenh264"):
        if name in listed:
            return name
    return None


class VideoWriter:
    """Writes H.264 through ffmpeg, falling back to OpenCV's mp4v.

    cv2.VideoWriter's portable fourcc is 'mp4v' -- MPEG-4 Part 2, which pip's
    OpenCV can encode but no browser will decode. That is why the first cut of
    these videos opened to a black frame in VS Code and in Chrome alike: the
    file was fine, nothing on the machine could play it. H.264 is what those
    players actually support.
    """

    def __init__(self, path: Path, fps: float, size, quiet: bool = False):
        width, height = size
        # H.264 in yuv420p needs even dimensions; pad rather than resize so no
        # pixel of the render moves.
        self.pad = ((width % 2), (height % 2))
        self.size = (width, height)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._process = None
        self._cv2 = None

        ffmpeg = _find_ffmpeg()
        encoder = _h264_encoder(ffmpeg) if ffmpeg else None
        if encoder is None:
            if not quiet:
                print("no ffmpeg with an H.264 encoder found; writing mp4v, which "
                      "most players cannot open", flush=True)
            self._cv2 = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, self.size
            )
            self.codec = "mp4v"
            return

        self.codec = encoder
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "-s", f"{width + self.pad[0]}x{height + self.pad[1]}",
            "-r", f"{fps}", "-i", "-",
            "-an", "-c:v", encoder, "-pix_fmt", "yuv420p",
            # faststart so a player can begin before the whole file is read
            "-movflags", "+faststart",
        ]
        # libx264 takes a quality target; OpenH264 only understands a bitrate.
        command += ["-crf", "20", "-preset", "medium"] if encoder == "libx264" else ["-b:v", "8M"]
        command.append(str(path))
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE)

    def write(self, frame):
        if self._cv2 is not None:
            self._cv2.write(frame)
            return
        if any(self.pad):
            frame = cv2.copyMakeBorder(
                frame, 0, self.pad[1], 0, self.pad[0], cv2.BORDER_CONSTANT, value=PANEL
            )
        self._process.stdin.write(np.ascontiguousarray(frame).tobytes())

    def release(self):
        if self._cv2 is not None:
            self._cv2.release()
            return
        self._process.stdin.close()
        if self._process.wait() != 0:
            raise RuntimeError(f"ffmpeg failed writing {self.path}")


def _rectified_size(clip: str, cam: dict = None) -> tuple:
    """The resolution the log's own ``cam_intrinsic`` is written for.

    Order matters here, and the obvious order is wrong. ``NUREC_CALIBRATION``
    describes the *source render* -- 1920x1080, the f-theta frames the rectifier
    consumed -- while the log beside the stored files carries TARGET_K, already
    in 512x256 units. Preferring the calibration table therefore scales a
    rectified projection by 512/1920 a second time and drops the overlay off the
    bottom of the image. The log wins; the table is only consulted for a log
    that is not rectified, where nuPlan's principal point sits 20 px below
    centre and 2*(cx, cy) would say 1120 instead of 1080.
    """
    def _from_intrinsic():
        if cam is None:
            return RECTIFIED_WH
        K = np.asarray(cam["cam_intrinsic"], dtype=np.float64)
        return 2.0 * K[0, 2], 2.0 * K[1, 2]

    if cam is not None:
        K = np.asarray(cam["cam_intrinsic"], dtype=np.float64)
        if np.allclose(K, TARGET_K, atol=1e-6):
            return float(TARGET_WIDTH), float(TARGET_HEIGHT)
    path = os.environ.get("NUREC_CALIBRATION")
    if not path or not Path(path).is_file():
        return _from_intrinsic()
    try:
        cameras = json.load(open(path))["clips"][clip]
    except (KeyError, ValueError, OSError):
        return _from_intrinsic()
    for name in ("CAM_F0", *CAMS):
        try:
            width, height = cameras[name]["camera_model"]["parameters"]["resolution"]
            return int(width), int(height)
        except (KeyError, TypeError, ValueError):
            continue
    return _from_intrinsic()


def _ground(xy: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(xy, np.float64), np.zeros((len(xy), 1))], axis=1)


def _polyline(canvas, xy, projection, colour, label, width=3):
    """Draws an ego-frame ground path onto a camera image."""
    if len(xy) < 2:
        return
    dense = np.concatenate([np.linspace(xy[i], xy[i + 1], 10) for i in range(len(xy) - 1)])
    pixels, in_front = project(_ground(dense), projection)
    pts = [tuple(p.astype(int)) for p, ok in zip(pixels, in_front) if ok]
    h, w = canvas.shape[:2]
    pts = [p for p in pts if -w < p[0] < 2 * w and -h < p[1] < 2 * h]
    for a, b in zip(pts, pts[1:]):
        cv2.line(canvas, a, b, colour, width, cv2.LINE_AA)
    if pts and label:
        cv2.putText(canvas, label, (pts[-1][0] + 6, pts[-1][1]), F, 0.55, colour, 2, cv2.LINE_AA)


class Bev:
    """A metres-to-pixels canvas centred on the ego at t=0, ego heading up."""

    def __init__(self, size: int, span_m: float, origin_xy, heading: float):
        self.size = size
        self.px_per_m = size / span_m
        self.origin = np.asarray(origin_xy, np.float64)
        # rotate so the ego's heading points up the image
        c, s = np.cos(-heading + np.pi / 2), np.sin(-heading + np.pi / 2)
        self.rot = np.array([[c, -s], [s, c]])
        self.canvas = np.full((size, size, 3), GROUND, np.uint8)

    def to_px(self, xy) -> np.ndarray:
        local = (np.asarray(xy, np.float64).reshape(-1, 2) - self.origin) @ self.rot.T
        px = local * self.px_per_m
        # x right, y up -> image column right, image row down; ego sits low
        return np.stack([px[:, 0] + self.size / 2, self.size * 0.72 - px[:, 1]], axis=1)

    def poly(self, xy, fill=None, edge=None, width=1):
        pts = self.to_px(xy).astype(np.int32)
        if len(pts) < 3:
            return
        if fill is not None:
            cv2.fillPoly(self.canvas, [pts], fill, cv2.LINE_AA)
        if edge is not None:
            cv2.polylines(self.canvas, [pts], True, edge, width, cv2.LINE_AA)

    def line(self, xy, colour, width=2, closed=False, dashed=False):
        pts = self.to_px(xy).astype(np.int32)
        if len(pts) < 2:
            return
        if not dashed:
            cv2.polylines(self.canvas, [pts], closed, colour, width, cv2.LINE_AA)
            return
        for i in range(0, len(pts) - 1, 2):
            cv2.line(self.canvas, tuple(pts[i]), tuple(pts[i + 1]), colour, width, cv2.LINE_AA)


def _offset_band(path_xy: np.ndarray, half_width: float) -> Optional[np.ndarray]:
    """The +-half_width corridor around a path, as one polygon ring."""
    if len(path_xy) < 2:
        return None
    line = LineString(path_xy)
    if line.length <= 0:
        return None
    band = line.buffer(half_width, cap_style=2, join_style=2)
    if band.is_empty:
        return None
    if band.geom_type == "MultiPolygon":
        band = max(band.geoms, key=lambda g: g.area)
    return np.asarray(band.exterior.coords)[:, :2]


def _first_actor_contact(coords, observation, radius):
    """First at-fault contact: (timestep, other's polygon).

    At fault means the front bumper or a flank reached the other body, which is
    the rule the score uses -- being run into from behind is not the ego's.
    """
    for step in range(coords.shape[0]):
        rect = coords[step, :4]
        body = _round_corners(Polygon(rect), radius)
        front = LineString([rect[BBCoordsIndex.FRONT_LEFT], rect[BBCoordsIndex.FRONT_RIGHT]])
        rear = LineString([rect[BBCoordsIndex.REAR_LEFT], rect[BBCoordsIndex.REAR_RIGHT]])
        for other_token in observation[step].tokens:
            if observation.red_light_token in other_token:
                continue
            obj = observation.unique_objects.get(other_token)
            if obj is None:
                continue
            other = _round_corners(
                observation[step][other_token],
                VEHICLE_CORNER_ROUNDNESS * min(obj.box.length, obj.box.width) / 2.0,
            )
            if not body.intersects(other):
                continue
            if other.intersects(rear) and not other.intersects(front):
                continue  # rear-ended; not the ego's fault
            return step, other
    return None, None


def _first_edge_contact(coords, road_edges, radius):
    """First timestep whose body reaches a road edge, and the edge it reached."""
    if not road_edges:
        return None, None
    geoms = []
    for edge in road_edges:
        geom = getattr(edge, "geometry", edge)
        xy = np.asarray(getattr(geom, "coords", []), float)
        if len(xy) >= 2:
            geoms.append(LineString(xy[:, :2]))
    for step in range(coords.shape[0]):
        body = _round_corners(Polygon(coords[step, :4]), radius)
        for geom in geoms:
            if body.distance(geom) <= ROAD_EDGE_CONTACT_M:
                return step, np.asarray(geom.coords)
    return None, None


def _bar(canvas, x, y, w, h, value, label, detail):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (44, 44, 48), -1)
    filled = int(round(w * float(np.clip(value, 0.0, 1.0))))
    colour = BAD if value < 0.999 else GOOD
    if filled > 0:
        cv2.rectangle(canvas, (x, y), (x + filled, y + h), colour, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (86, 86, 92), 1)
    cv2.putText(canvas, label, (x, y - 7), F, 0.52, INK, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{value:.3f}", (x + w + 10, y + h - 3), F, 0.52, colour, 1, cv2.LINE_AA)
    if detail:
        cv2.putText(canvas, detail, (x + w + 74, y + h - 3), F, 0.42, DIM, 1, cv2.LINE_AA)


def _panel(width, height, row, series, index, token):
    """The score readout: four terms, their product, and the clip's run of scores."""
    canvas = np.full((height, width, 3), PANEL, np.uint8)
    cv2.putText(canvas, "score  =  NC  x  DAC  x  GT  x  EP", (24, 38), F, 0.72, INK, 1, cv2.LINE_AA)
    cv2.putText(canvas, token[:16], (24, 62), F, 0.46, DIM, 1, cv2.LINE_AA)

    y = 96
    for term in TERMS:
        value = float(row[term])
        detail = ""
        if term == "gt_compliance" and value < 0.999:
            detail = f"drifted past {GT_LATERAL_LIMIT_M:.0f} m"
        elif term == "ego_progress" and value < 0.999:
            detail = "short of the recorded run"
        elif value < 0.999:
            detail = "failed"
        _bar(canvas, 24, y, 230, 22, value, TERM_LABEL[term], detail)
        y += 54

    score = float(row["score"])
    cv2.line(canvas, (24, y - 12), (width - 24, y - 12), (60, 60, 66), 1)
    colour = BAD if score < 0.001 else (GOOD if score > 0.999 else INK)
    cv2.putText(canvas, "score", (24, y + 26), F, 0.66, INK, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"{score:.3f}", (150, y + 26), F, 0.9, colour, 2, cv2.LINE_AA)

    # the clip's scores, with a cursor on this frame
    top, bottom, left, right = y + 58, height - 34, 24, width - 24
    cv2.rectangle(canvas, (left, top), (right, bottom), (40, 40, 44), -1)
    n = max(len(series), 1)
    step = (right - left) / n
    for i, value in enumerate(series):
        x0 = int(left + i * step)
        x1 = int(left + (i + 1) * step) - 1
        h = int((bottom - top) * float(np.clip(value, 0.0, 1.0)))
        col = BAD if value < 0.001 else (GOOD if value > 0.999 else (150, 150, 160))
        # A zero is a result, not a gap: give it a floor tick so the strip reads
        # as "scored 0" rather than "no frame here".
        cv2.rectangle(canvas, (x0, bottom - max(h, 3)), (max(x1, x0 + 1), bottom), col, -1)
    cx = int(left + (index + 0.5) * step)
    cv2.line(canvas, (cx, top), (cx, bottom), (255, 255, 255), 1)
    cv2.putText(canvas, f"clip score {np.mean(series):.3f}   frame {index + 1}/{n}",
                (left, height - 14), F, 0.46, DIM, 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", type=Path, required=True, help="pickle saved by the scoring job")
    ap.add_argument("--scores", type=Path, required=True, help="the scoring job's CSV")
    ap.add_argument("--log", required=True, help="nurec-<clip>")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--cam-width", type=int, default=760)
    ap.add_argument("--span", type=float, default=70.0, help="BEV width in metres")
    args = ap.parse_args()

    prepared = Path(os.environ["NUREC_PREPARED_ROOT"])
    image_root = Path(os.environ["NUREC_SENSOR_ROOT"])
    cache_root = prepared / os.environ.get("NUREC_METRIC_CACHE_DIRNAME", "metric_cache")
    clip = args.log.removeprefix("nurec-")

    predictions = pickle.load(args.predictions.open("rb"))
    scores = pd.read_csv(args.scores)
    scores = scores[scores.token != "average_all_frames"].set_index("token")

    frames: List[Dict] = pickle.load((prepared / "navsim_logs" / "trainval" / f"{args.log}.pkl").open("rb"))
    order = {f["token"]: i for i, f in enumerate(frames)}
    tokens = sorted(
        (t for t in predictions if t in scores.index and t in order),
        key=lambda t: order[t],
    )
    if not tokens:
        print(f"no scored frames for {args.log}", flush=True)
        return
    print(f"{args.log}: {len(tokens)} scored frames", flush=True)

    # Pass the frame's own camera: the log's cam_intrinsic is what lidar2img is
    # built from, so it -- not the calibration table's source resolution -- is
    # what the projection has to be scaled against.
    rect_w, rect_h = _rectified_size(clip, frames[order[tokens[0]]]["cams"]["CAM_F0"])
    print(f"intrinsics resolution {rect_w:.0f}x{rect_h:.0f}", flush=True)

    lanes = None
    bundle_path = prepared / "maps" / f"{clip}.pkl"
    if bundle_path.is_file():
        lanes = pickle.load(bundle_path.open("rb")).get("lanes", {})

    sampling = TrajectorySampling(num_poses=40, interval_length=0.1)
    simulator = NuRecSimulator(sampling)
    series = [float(scores.loc[t, "score"]) for t in tokens]

    writer = None
    for index, token in enumerate(tokens):
        row = scores.loc[token]
        frame = frames[order[token]]

        cache_file = cache_root / args.log / "unknown" / token / "metric_cache.pkl"
        with lzma.open(cache_file, "rb") as fh:
            cache = pickle.load(fh)
        ego = cache.ego_state
        params = ego.car_footprint.vehicle_parameters
        radius = VEHICLE_CORNER_ROUNDNESS * min(params.length, params.width) / 2.0

        # The scored rollout: the model's trajectory through the same controller
        # the scoring job used, so the picture and the number come from one path.
        pred_states = simulator.simulate_proposals(
            np.stack([_states_from(predictions[token]["trajectory"], ego, sampling)]), ego
        )[0]
        coords = state_array_to_coords_array(pred_states[None, ...], params)[0]
        model_xy = pred_states[:, [StateIndex.X, StateIndex.Y]]

        # Sampled the same way the scorer samples it, so the drawn corridor is
        # the one gt_compliance measured against.
        human_xy = _states_from(cache.human_trajectory, ego, sampling)[:, [StateIndex.X, StateIndex.Y]]

        # ---------------------------------------------------------- bird's eye
        # Fit the view to what actually has to be visible. A clip crawling in
        # traffic covers 7 m in the 4 s horizon; a fixed 70 m frame renders it
        # as a dot.
        extent = float(
            np.abs(np.vstack([model_xy, human_xy]) - model_xy[0]).max(initial=0.0)
        )
        span = float(np.clip(2.6 * extent + 12.0, 26.0, args.span))
        bev = Bev(args.cam_width, span, model_xy[0], ego.rear_axle.heading)
        centre = Point(*model_xy[0])
        if lanes:
            for lane in lanes.values():
                left = np.asarray(lane["left"], float)[:, :2]
                right = np.asarray(lane["right"], float)[:, :2]
                if len(left) < 2 or len(right) < 2:
                    continue
                poly = Polygon(np.vstack([left, right[::-1]])).buffer(0)
                if poly.is_empty or poly.distance(centre) > span:
                    continue
                bev.poly(np.asarray(poly.exterior.coords)[:, :2], fill=LANE, edge=LANE_EDGE)
        for edge in getattr(cache, "road_edges", None) or []:
            geom = getattr(edge, "geometry", edge)
            coords_xy = np.asarray(getattr(geom, "coords", []), float)
            if len(coords_xy) >= 2 and LineString(coords_xy[:, :2]).distance(centre) < span:
                bev.line(coords_xy[:, :2], ROAD_EDGE, 2)

        band = _offset_band(human_xy, GT_LATERAL_LIMIT_M)
        if band is not None:
            bev.poly(band, fill=None, edge=BAND, width=2)

        observation = cache.observation
        for other in observation[0].tokens:
            if observation.red_light_token in other:
                continue
            obj = observation.unique_objects.get(other)
            if obj is None:
                continue
            body = _round_corners(
                observation[0][other],
                VEHICLE_CORNER_ROUNDNESS * min(obj.box.length, obj.box.width) / 2.0,
            )
            if body.is_empty or body.distance(centre) > span:
                continue
            bev.poly(np.asarray(body.exterior.coords)[:, :2], fill=ACTOR)

        # Why the frame scored what it did, drawn rather than described.
        cause = ""
        if float(row["no_at_fault_collisions"]) < 0.5:
            step, other = _first_actor_contact(coords, observation, radius)
            if other is not None:
                bev.poly(np.asarray(other.exterior.coords)[:, :2], fill=CULPRIT)
                bev.line(coords[step, :4], CULPRIT, 2, closed=True)
                cause = f"NC 0  --  hit at t = {step * sampling.interval_length:.1f} s"
        elif float(row["drivable_area_compliance"]) < 0.5:
            step, edge = _first_edge_contact(coords, getattr(cache, "road_edges", None), radius)
            if edge is not None:
                bev.line(edge, CULPRIT, 3)
                bev.line(coords[step, :4], CULPRIT, 2, closed=True)
                cause = f"DAC 0  --  road edge at t = {step * sampling.interval_length:.1f} s"
        elif float(row["gt_compliance"]) < 0.5:
            cause = f"GT 0  --  more than {GT_LATERAL_LIMIT_M:.0f} m off the recorded run"
        if cause:
            cv2.putText(bev.canvas, cause, (12, 64), F, 0.5, CULPRIT, 1, cv2.LINE_AA)

        bev.line(human_xy, HUMAN, 2, dashed=True)
        bev.line(model_xy, MODEL, 2)
        bev.line(coords[0, :4], INK, 2, closed=True)              # where it starts
        bev.line(coords[-1, :4], MODEL, 2, closed=True)           # where it ends, 4 s on

        cv2.putText(bev.canvas, "bird's eye  --  controller rollout of the scored trajectory",
                    (12, 22), F, 0.48, DIM, 1, cv2.LINE_AA)
        cv2.putText(bev.canvas, f"{span:.0f} m across", (12, 42), F, 0.44, DIM, 1, cv2.LINE_AA)
        for i, (text, colour) in enumerate([
            ("other traffic", ACTOR),
            (f"+-{GT_LATERAL_LIMIT_M:.0f} m drift limit (GT)", LEGEND_BAND),
            ("model, 4 s rollout", MODEL),
            ("recorded human", HUMAN),
            ("ego now", INK),
        ]):
            cv2.putText(bev.canvas, text, (12, bev.size - 16 - 20 * i), F, 0.46, colour, 1, cv2.LINE_AA)

        # ------------------------------------------------------------- cameras
        model_ego = np.vstack([[0.0, 0.0], np.asarray(predictions[token]["trajectory"].poses)[:, :2]])
        human_ego = np.vstack([[0.0, 0.0], np.asarray(cache.human_trajectory.poses)[:, :2]])
        tiles = []
        for name in CAMS:
            cam = frame["cams"][name]
            image = cv2.imread(str(image_root / cam["data_path"]), cv2.IMREAD_COLOR)
            if image is None:
                tiles = []
                break
            height = int(round(args.cam_width * image.shape[0] / image.shape[1]))
            canvas = cv2.resize(image, (args.cam_width, height), interpolation=cv2.INTER_AREA)
            # Per axis: the stored frames are the rectified image squashed to
            # 512x256, so a single scalar scale puts the path off-frame.
            projection = np.diag([
                args.cam_width / rect_w, height / rect_h, 1.0, 1.0,
            ]) @ lidar2img(cam)
            _polyline(canvas, human_ego, projection, HUMAN, "human", width=4)
            _polyline(canvas, model_ego, projection, MODEL, "model", width=4)
            cv2.putText(canvas, name, (10, 24), F, 0.55, INK, 1, cv2.LINE_AA)
            tiles.append(canvas)
        if not tiles:
            continue
        cam_row = np.hstack(tiles)

        panel = _panel(cam_row.shape[1] - bev.size, bev.size, row, series, index, token)
        body = np.hstack([bev.canvas, panel])
        header = np.full((34, cam_row.shape[1], 3), PANEL, np.uint8)
        cv2.putText(header, f"{args.log}    frame {index + 1}/{len(tokens)}    "
                            f"score {float(row['score']):.3f}", (14, 23), F, 0.56, INK, 1, cv2.LINE_AA)
        canvas = np.vstack([header, cam_row, body])

        if writer is None:
            writer = VideoWriter(args.out, args.fps, (canvas.shape[1], canvas.shape[0]))
        writer.write(canvas)

    if writer is None:
        print("no frames rendered", flush=True)
        return
    writer.release()
    print(f"wrote {args.out}  codec={writer.codec}  frames={len(tokens)}  "
          f"clip score={np.mean(series):.4f}", flush=True)


def _states_from(trajectory, ego, sampling) -> np.ndarray:
    """Ego-frame Trajectory -> the state array the simulator rolls out."""
    return get_trajectory_as_array(transform_trajectory(trajectory, ego), sampling, ego.time_point)


if __name__ == "__main__":
    main()
