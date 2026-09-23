"""Play one clip's rectified views back as video, with the geometry drawn on.

A filmstrip shows that each frame is right; a video shows that the frames are
right *together* -- the ego path sliding along the road as the car drives it,
the boxes sticking to the cars they belong to, the horizon staying put. A
per-frame error that a still hides tends to jump out as flicker.

    source setup_nurec.sh
    python scripts/nurec/preview_video.py --log nurec-<clip> --out preview.mp4
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.nurec.verify_rectified_projection import (
    _EDGES, box_corners, future_path_in_ego_frame, lidar2img, project)

CAMS = ("CAM_L0", "CAM_F0", "CAM_R0")


def render_frame(frames, index, image_root: Path, scale: float, path_len: int):
    frame = frames[index]
    path = future_path_in_ego_frame(frames, index, path_len)
    tiles = []
    for name in CAMS:
        camera = frame["cams"][name]
        image = cv2.imread(str(image_root / camera["data_path"]), cv2.IMREAD_COLOR)
        if image is None:
            return None
        canvas = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        projection = np.diag([scale, scale, 1.0, 1.0]) @ lidar2img(camera)

        boxes = np.asarray(frame["anns"]["gt_boxes"], dtype=np.float64).reshape(-1, 7)
        for box in boxes:
            pixels, in_front = project(box_corners(box), projection)
            if not in_front.all():
                continue
            points = pixels.astype(np.int32)
            w = canvas.shape[1]
            if not ((points[:, 0] > -w) & (points[:, 0] < 2 * w)).all():
                continue
            for a, b in _EDGES:
                cv2.line(canvas, tuple(points[a]), tuple(points[b]), (0, 220, 255), 1, cv2.LINE_AA)

        if len(path) > 1:
            dense = np.concatenate(
                [np.linspace(path[i], path[i + 1], 12) for i in range(len(path) - 1)])
            pixels, in_front = project(dense, projection)
            drawn = [tuple(p.astype(int)) for p, ok in zip(pixels, in_front) if ok]
            for a, b in zip(drawn, drawn[1:]):
                cv2.line(canvas, a, b, (0, 0, 255), 3, cv2.LINE_AA)

        cv2.putText(canvas, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5 * scale, (255, 255, 255), 1, cv2.LINE_AA)
        tiles += [canvas, np.full((canvas.shape[0], 3, 3), 35, np.uint8)]

    row = np.hstack(tiles[:-1])
    bar = np.full((26, row.shape[1], 3), 25, np.uint8)
    ego = frame["ego2global_translation"]
    cv2.putText(bar, f"t={index * 0.5:5.1f}s   frame {index:02d}/{len(frames) - 1}   "
                     f"ego ({ego[0]:7.1f}, {ego[1]:7.1f}) m   "
                     f"512x256 K377   red = ego future path   yellow = gt boxes",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (215, 215, 215), 1, cv2.LINE_AA)
    return np.vstack([bar, row])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="log name, e.g. nurec-<clip-id>")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=4.0,
                        help="frames are 0.5 s apart, so 2.0 is real time")
    parser.add_argument("--scale", type=float, default=2.0)
    parser.add_argument("--path-len", type=int, default=16)
    args = parser.parse_args()

    prepared = Path(os.environ["NUREC_PREPARED_ROOT"])
    image_root = Path(os.environ["NUREC_SENSOR_ROOT"])
    with (prepared / "navsim_logs" / "trainval" / f"{args.log}.pkl").open("rb") as stream:
        frames = pickle.load(stream)

    writer = None
    written = 0
    for index in range(len(frames)):
        canvas = render_frame(frames, index, image_root, args.scale, args.path_len)
        if canvas is None:
            continue          # the tail frames of a log carry no image by design
        if writer is None:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"),
                                     args.fps, (canvas.shape[1], canvas.shape[0]))
        writer.write(canvas)
        written += 1
    if writer is None:
        raise SystemExit(f"no readable frames for {args.log}")
    writer.release()
    print(f"{written} frames at {args.fps} fps -> {args.out}")


if __name__ == "__main__":
    main()
