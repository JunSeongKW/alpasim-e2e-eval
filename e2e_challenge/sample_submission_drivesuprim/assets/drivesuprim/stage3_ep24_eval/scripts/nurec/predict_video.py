#!/usr/bin/env python3
"""Play a clip back as video with the model's predicted trajectory drawn on.

preview_video.py draws the recorded geometry; this runs a checkpoint on every
frame of the clip and draws what the model would do next, beside the human's
path. A still shows one decision; a video shows whether consecutive decisions
agree -- the failure two_frame_extended_comfort measures.

    source setup_nurec.sh
    python scripts/nurec/predict_video.py --checkpoint <ckpt> --log nurec-<clip> --out out.mp4
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
import cv2, numpy as np, torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from torch.utils.data import DataLoader, Subset

from navsim.agents.drivesuprim.drivesuprim_agent import DriveSuprimAgent
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset_ssl import DatasetSSL
from scripts.nurec.verify_rectified_projection import lidar2img, project

CFG_DIR = str(Path(__file__).resolve().parents[2] / "navsim/planning/script/config/common/agent")
CAMS = ("CAM_L0", "CAM_F0", "CAM_R0")


def _ground(xy):
    return np.concatenate([np.asarray(xy, np.float64), np.zeros((len(xy), 1))], axis=1)


def _polyline(canvas, xy, projection, colour, label, width=3):
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
        cv2.putText(canvas, label, (pts[-1][0] + 6, pts[-1][1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--agent", default="drivesuprim_agent_bevformer_vov_v2_vits_planonly_scratch")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--scale", type=float, default=2.0)
    args = ap.parse_args()

    with initialize_config_dir(version_base=None, config_dir=CFG_DIR):
        cfg = instantiate(compose(config_name=args.agent)["config"])
    cfg.vocab_path = os.environ["NUREC_VOCAB_PATH"]
    cfg.ori_vocab_pdm_score_dir = os.environ["NUREC_ORI_PDM_SCORE_DIR"]
    cfg.only_ori_input = True
    cfg.use_traffic_light_compliance = False
    cfg.use_route = os.environ.get("NUREC_USE_ROUTE", "true").lower() == "true"
    cfg.training = False

    prepared = Path(os.environ["NUREC_PREPARED_ROOT"])
    image_root = Path(os.environ["NUREC_SENSOR_ROOT"])
    A = [0, 1, 2, 3]
    loader = SceneLoader(
        data_path=prepared / "navsim_logs" / "trainval",
        original_sensor_path=image_root, synthetic_sensor_path=None, synthetic_scenes_path=None,
        scene_filter=SceneFilter(num_history_frames=4, num_future_frames=8, frame_interval=1,
                                 has_route=True, max_scenes=None, log_names=[args.log],
                                 tokens=None, include_synthetic_scenes=False),
        sensor_config=SensorConfig(cam_f0=A, cam_l0=A, cam_r0=A, cam_l1=False, cam_l2=False,
                                   cam_r1=False, cam_r2=False, cam_b0=False, lidar_pc=False))
    print(f"log {args.log}: {len(loader.tokens)} tokens", flush=True)

    agent = DriveSuprimAgent(cfg, lr=1e-4, checkpoint_path=str(args.checkpoint))
    agent.initialize(); agent.eval()
    device = torch.device(args.device); agent.to(device)
    dataset = DatasetSSL(scene_loader=loader, feature_builders=agent.get_feature_builders(),
                         target_builders=agent.get_target_builders(), cfg=cfg, cache_path=None)

    def move(v):
        if torch.is_tensor(v): return v.to(device)
        if isinstance(v, list): return [move(x) for x in v]
        if isinstance(v, dict): return {k: move(x) for k, x in v.items()}
        return v

    # Images and projection come from the raw log pickle, the same path
    # preview_video.py uses -- the Scene dataclass does not carry cam_intrinsic
    # in the shape lidar2img() expects.
    import pickle as _pk
    with (prepared / "navsim_logs" / "trainval" / f"{args.log}.pkl").open("rb") as fh:
        raw = _pk.load(fh)
    by_token = {f["token"]: f for f in raw}

    errs, writer = [], None
    with torch.no_grad():
        for features, targets, tokens in DataLoader(dataset, batch_size=1, num_workers=2):
            token = tokens[0]
            frame = by_token.get(token)
            if frame is None:
                continue
            out = agent.forward((move(features), move(targets), list(tokens)))
            if isinstance(out, tuple):
                out = out[0]
            pred = np.asarray(out["trajectory"][0].float().cpu().numpy(), np.float64)[:, :2]
            gt = np.asarray(loader.get_scene_from_token(token).get_future_trajectory(
                num_trajectory_frames=8).poses, np.float64)[:, :2]
            pred = np.vstack([[0., 0.], pred])
            gt = np.vstack([[0., 0.], gt])
            err = float(np.linalg.norm(pred[-1] - gt[-1]))

            tiles, ok = [], True
            for name in CAMS:
                cam = frame["cams"][name]
                img = cv2.imread(str(image_root / cam["data_path"]), cv2.IMREAD_COLOR)
                if img is None:
                    ok = False
                    break
                canvas = cv2.resize(img, None, fx=args.scale, fy=args.scale,
                                    interpolation=cv2.INTER_CUBIC)
                proj = np.diag([args.scale, args.scale, 1., 1.]) @ lidar2img(cam)
                _polyline(canvas, gt, proj, (90, 220, 90), "GT")
                _polyline(canvas, pred, proj, (60, 120, 255), "pred")
                cv2.putText(canvas, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5 * args.scale, (255, 255, 255), 1, cv2.LINE_AA)
                tiles += [canvas, np.full((canvas.shape[0], 3, 3), 35, np.uint8)]
            if not ok:
                continue
            row = np.hstack(tiles[:-1])
            bar = np.full((28, row.shape[1], 3), 25, np.uint8)
            cv2.putText(bar, f"{token[:12]}   final-point error {err:5.2f} m   "
                             f"green = human GT   orange = prediction",
                        (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (215, 215, 215), 1, cv2.LINE_AA)
            canvas = np.vstack([bar, row])
            if writer is None:
                args.out.parent.mkdir(parents=True, exist_ok=True)
                writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"),
                                         args.fps, (canvas.shape[1], canvas.shape[0]))
            writer.write(canvas)
            errs.append(err)
    if writer is not None:
        writer.release()
        print(f"wrote {args.out}  frames={len(errs)}  median err={np.median(errs):.2f} m",
              flush=True)
    else:
        print("no frames rendered", flush=True)


if __name__ == "__main__":
    main()
