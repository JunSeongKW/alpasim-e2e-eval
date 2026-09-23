"""Run a checkpoint on a few scenes and draw what it predicts.

Two views per scene, because a trajectory that looks fine in one can be wrong in
the other: the front camera (does the path sit on the road the model can see?)
and a top-down BEV against the ground truth (is it the right shape?).

The prediction is a pick from the 4,096-entry trajectory vocabulary -- the model
scores every candidate and takes the argmax -- so early in training it will
choose something close to the vocabulary's mean and the point is the plumbing,
not the driving.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from navsim.agents.drivesuprim.drivesuprim_agent import DriveSuprimAgent
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
from torch.utils.data import DataLoader, Subset
from navsim.planning.training.dataset_ssl import DatasetSSL
from scripts.nurec.verify_rectified_projection import lidar2img, project

CFG_DIR = str(Path(__file__).resolve().parents[2] / "navsim/planning/script/config/common/agent")


def _to_xy(traj: np.ndarray) -> np.ndarray:
    """Vocabulary rows are (T, 3) = x, y, heading in the ego frame."""
    return np.asarray(traj, dtype=np.float64)[:, :2]


def _ground(xy: np.ndarray) -> np.ndarray:
    return np.concatenate([xy, np.zeros((len(xy), 1))], axis=1)


def draw_camera(camera, image_root: Path, pred_xy, gt_xy):
    image = cv2.imread(str(image_root / camera["data_path"]), cv2.IMREAD_COLOR)
    if image is None:
        return None
    canvas = cv2.resize(image, (image.shape[1] * 2, image.shape[0] * 2))
    projection = lidar2img(camera)
    scale = np.diag([2.0, 2.0, 1.0, 1.0])
    for xy, colour, label in ((gt_xy, (90, 220, 90), "GT"), (pred_xy, (60, 120, 255), "pred")):
        dense = np.concatenate([np.linspace(xy[i], xy[i + 1], 10) for i in range(len(xy) - 1)])
        pixels, in_front = project(_ground(dense), scale @ projection)
        points = [tuple(p.astype(int)) for p, ok in zip(pixels, in_front) if ok]
        for a, b in zip(points, points[1:]):
            cv2.line(canvas, a, b, colour, 3, cv2.LINE_AA)
        if points:
            cv2.putText(canvas, label, (points[-1][0] + 6, points[-1][1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2, cv2.LINE_AA)
    return canvas


def draw_bev(pred_xy, gt_xy, size=512, metres=40.0):
    canvas = np.full((size, size, 3), 24, np.uint8)
    ppm = size / (2 * metres)

    def to_px(xy):
        # ego x forward -> up, ego y left -> left
        return np.stack([size / 2 - xy[:, 1] * ppm, size / 2 - xy[:, 0] * ppm], axis=1).astype(int)

    for r in range(10, int(metres) + 1, 10):
        cv2.circle(canvas, (size // 2, size // 2), int(r * ppm), (55, 55, 55), 1)
        cv2.putText(canvas, f"{r}m", (size // 2 + 4, size // 2 - int(r * ppm) + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (110, 110, 110), 1, cv2.LINE_AA)
    cv2.drawMarker(canvas, (size // 2, size // 2), (200, 200, 200), cv2.MARKER_TRIANGLE_UP, 14, 2)

    for xy, colour, label in ((gt_xy, (90, 220, 90), "GT"), (pred_xy, (60, 120, 255), "pred")):
        pts = to_px(xy)
        for a, b in zip(pts, pts[1:]):
            cv2.line(canvas, tuple(a), tuple(b), colour, 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(canvas, tuple(p), 3, colour, -1)
        cv2.putText(canvas, label, (10, 20 if label == "GT" else 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1, cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--agent", default="drivesuprim_agent_bevformer_vov_v2_vits_planonly_scratch")
    parser.add_argument("--scenes", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    parser.add_argument("--tokens", default=None,
                        help="comma-separated tokens to draw instead of an even stride")
    parser.add_argument("--logs", type=int, default=None,
                        help="how many val logs to load (default: --scenes)")
    args = parser.parse_args()

    with initialize_config_dir(version_base=None, config_dir=CFG_DIR):
        cfg = instantiate(compose(config_name=args.agent)["config"])
    cfg.vocab_path = os.environ["NUREC_VOCAB_PATH"]
    cfg.ori_vocab_pdm_score_dir = os.environ["NUREC_ORI_PDM_SCORE_DIR"]
    cfg.only_ori_input = True
    cfg.use_traffic_light_compliance = False
    # Must match what the checkpoint was trained with; see setup_nurec.sh.
    cfg.use_route = os.environ.get("NUREC_USE_ROUTE", "true").lower() == "true"
    cfg.training = False

    prepared = Path(os.environ["NUREC_PREPARED_ROOT"])
    image_root = Path(os.environ["NUREC_SENSOR_ROOT"])
    _want = [t for t in (args.tokens or "").split(",") if t]
    _nlogs = args.logs or (120 if _want else args.scenes)
    logs = json.load(open(prepared / "index.json"))["logs"]["val"][: _nlogs]

    scene_filter = SceneFilter(num_history_frames=4, num_future_frames=8, frame_interval=1,
                               has_route=True, max_scenes=None, log_names=logs, tokens=None,
                               include_synthetic_scenes=False)
    sensor_config = SensorConfig(cam_f0=[0, 1, 2, 3], cam_l0=[0, 1, 2, 3], cam_r0=[0, 1, 2, 3],
                                 cam_l1=False, cam_l2=False, cam_r1=False, cam_r2=False,
                                 cam_b0=False, lidar_pc=False)
    loader = SceneLoader(data_path=prepared / "navsim_logs" / "trainval",
                         original_sensor_path=image_root,
                         synthetic_sensor_path=None, synthetic_scenes_path=None,
                         scene_filter=scene_filter, sensor_config=sensor_config)

    agent = DriveSuprimAgent(cfg, lr=1e-4, checkpoint_path=str(args.checkpoint))
    agent.initialize()
    agent.eval()
    # SSLMetaArch holds a teacher and a student; .cuda() on the agent does not
    # reach both, so a linear layer ends up with cpu weights against cuda inputs.
    # Six scenes on the CPU takes a couple of minutes and needs no free GPU.
    device = torch.device(args.device)
    agent.to(device)

    dataset = DatasetSSL(scene_loader=loader, feature_builders=agent.get_feature_builders(),
                         target_builders=agent.get_target_builders(), cfg=cfg, cache_path=None)

    rows = []
    if _want:
        _index = {tok: i for i, tok in enumerate(loader.tokens)}
        missing = [t for t in _want if t not in _index]
        if missing:
            raise SystemExit(f"tokens not in the loaded logs: {missing}")
        picks = [_index[t] for t in _want]
    else:
        step = max(1, len(dataset) // args.scenes)
        picks = list(range(0, min(len(dataset), step * args.scenes), step))
    # Collate through a DataLoader rather than by hand: the model's status head
    # is shape-sensitive and hand-batching silently produces the wrong rank.
    loader_iter = DataLoader(Subset(dataset, picks), batch_size=1, num_workers=0)
    with torch.no_grad():
        for features, targets, tokens in loader_iter:
            token = tokens[0]
            # Features hold lists of tensors as well as bare tensors
            # (status_feature is one per history frame), so move recursively --
            # a shallow move leaves those on the CPU and the first Linear then
            # raises "Expected all tensors to be on the same device".
            def _move(v):
                if torch.is_tensor(v):
                    return v.to(device)
                if isinstance(v, list):
                    return [_move(x) for x in v]
                if isinstance(v, dict):
                    return {k: _move(x) for k, x in v.items()}
                return v
            to_dev = lambda d: {k: _move(v) for k, v in d.items()}
            batch = (to_dev(features), to_dev(targets), list(tokens))
            out = agent.forward(batch)
            # With cfg.training False, SSLMetaArch returns (teacher_pred, []) --
            # the student branch is not run at all.
            if isinstance(out, tuple):
                out = out[0]
            pred = _to_xy(out["trajectory"][0].float().cpu().numpy())
            # The dataset only builds targets when cfg.training is on, and this
            # runs with it off so SSLMetaArch takes the inference branch.  Take
            # the ground truth from the scene instead -- the same call the target
            # builder makes.
            scene = loader.get_scene_from_token(token)
            gt = _to_xy(np.asarray(scene.get_future_trajectory(
                num_trajectory_frames=int(4 / 0.5)).poses, dtype=np.float64))
            pred = np.vstack([[0.0, 0.0], pred])
            gt = np.vstack([[0.0, 0.0], gt])

            cam_obj = scene.frames[3].cameras.cam_f0
            camera = {"data_path": cam_obj.camera_path,
                      "cam_intrinsic": cam_obj.intrinsics,
                      "sensor2lidar_rotation": cam_obj.sensor2lidar_rotation,
                      "sensor2lidar_translation": cam_obj.sensor2lidar_translation}
            cam = draw_camera(camera, image_root, pred, gt)
            bev = draw_bev(pred, gt)
            if cam is None:
                continue
            bev = cv2.resize(bev, (cam.shape[0], cam.shape[0]))
            err = float(np.linalg.norm(pred[-1] - gt[-1]))
            row = np.hstack([cam, np.zeros((cam.shape[0], 4, 3), np.uint8), bev])
            cv2.putText(row, f"{token[:12]}  final-point error {err:.2f} m", (8, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 2, cv2.LINE_AA)
            rows.append(row)
            print(f"  {token[:12]}  pred_end=({pred[-1][0]:6.2f},{pred[-1][1]:6.2f})  "
                  f"gt_end=({gt[-1][0]:6.2f},{gt[-1][1]:6.2f})  err={err:.2f} m")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    width = max(r.shape[1] for r in rows)
    cv2.imwrite(str(args.out), np.vstack(
        [np.pad(r, ((0, 0), (0, width - r.shape[1]), (0, 0))) for r in rows]))
    print(args.out)


if __name__ == "__main__":
    main()
