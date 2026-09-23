"""Optional, inference-only BEV diagnostics for the challenge driver.

The plan-only checkpoint has no semantic BEV head.  Consequently PCA colors
below are latent features, not road/object classes.  The blank-image ablation
measures whether camera pixels influence each BEV cell.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Mapping

import cv2
import numpy as np
import torch

LOGGER = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name, "1" if default else "0")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _tensor_from_hook(output) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, Mapping):
        for key in ("bev", "bev_feature", "bev_features"):
            value = output.get(key)
            if torch.is_tensor(value):
                return value
    if isinstance(output, (tuple, list)):
        for value in output:
            if torch.is_tensor(value) and value.ndim == 4:
                return value
    raise TypeError(f"no [B,C,H,W] tensor in BEV output {type(output)!r}")


def _pca_rgb(bev: np.ndarray) -> np.ndarray:
    channels, height, width = bev.shape
    values = np.nan_to_num(
        bev.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0
    )
    flat = values.reshape(channels, height * width).T
    flat -= flat.mean(axis=0, keepdims=True)
    covariance = (flat.T @ flat) / max(flat.shape[0] - 1, 1)
    _, eigenvectors = np.linalg.eigh(covariance)
    rgb = (flat @ eigenvectors[:, ::-1][:, :3]).reshape(height, width, 3)
    lower = np.percentile(rgb, 2, axis=(0, 1))
    upper = np.percentile(rgb, 98, axis=(0, 1))
    rgb = np.clip((rgb - lower) / np.maximum(upper - lower, 1e-8), 0, 1)
    # Same orientation as the training-package visualizer.
    return np.flipud(np.rot90(rgb, k=-1))


def _sensitivity(real: np.ndarray, blind: np.ndarray) -> np.ndarray:
    real = np.nan_to_num(real, nan=0.0, posinf=0.0, neginf=0.0)
    blind = np.nan_to_num(blind, nan=0.0, posinf=0.0, neginf=0.0)
    return np.linalg.norm(real - blind, axis=0) / (
        np.linalg.norm(real, axis=0) + 1e-8
    )


class BevDebugger:
    """Capture selected teacher-BEV forwards without changing normal runs."""

    def __init__(self, agent, config) -> None:
        value = os.environ.get("DRIVESUPRIM_BEV_DEBUG_DIR", "").strip()
        self.output_dir = Path(value) if value else None
        self.every = max(int(os.environ.get("DRIVESUPRIM_BEV_DEBUG_EVERY", "20")), 1)
        self.maximum = max(int(os.environ.get("DRIVESUPRIM_BEV_DEBUG_MAX", "10")), 0)
        self.ablation = _env_flag("DRIVESUPRIM_BEV_DEBUG_ABLATION", True)
        self.replica = (
            os.environ.get("DRIVESUPRIM_BEV_DEBUG_REPLICA", "driver").strip()
            or "driver"
        )
        self.calls = 0
        self.saved = 0
        self.agent = agent
        self.config = config
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            LOGGER.warning(
                "BEV debug enabled: dir=%s every=%d max=%d ablation=%s; "
                "diagnostic runs must not be reported as challenge scores",
                self.output_dir, self.every, self.maximum, self.ablation,
            )

    def should_capture(self) -> bool:
        self.calls += 1
        return (
            self.output_dir is not None
            and self.saved < self.maximum
            and (self.calls - 1) % self.every == 0
            and self.config.backbone_type == "bevformer_m"
        )

    def install_hook(self, captured: list[torch.Tensor]):
        branch_name = str(getattr(self.config.inference, "model", "teacher"))
        branch = getattr(self.agent.model, branch_name)

        def hook(_module, _inputs, output):
            captured.append(_tensor_from_hook(output).detach().float().cpu())

        return branch.model._backbone.register_forward_hook(hook)

    @staticmethod
    def blind_features(features):
        result = dict(features)
        for key in ("bev_imgs", "bev_imgs_teacher"):
            if key in result:
                result[key] = torch.zeros_like(result[key])
        return result

    def save(
        self, features, prediction, captured: list[torch.Tensor], context: str = ""
    ) -> None:
        if not captured:
            raise RuntimeError("BEV hook captured no tensor")
        assert self.output_dir is not None
        real = captured[0][0].numpy()
        blind = captured[1][0].numpy() if len(captured) > 1 else None
        sensitivity = _sensitivity(real, blind) if blind is not None else None
        safe_context = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in context
        )[:80]
        context_part = f"_{safe_context}" if safe_context else ""
        stem = f"{self.replica}{context_part}_call{self.calls:05d}_capture{self.saved:03d}"

        imgs = features["bev_imgs_teacher"].detach().float().cpu().numpy()[0]
        lidar2img = features["lidar2img"].detach().float().cpu().numpy()[0]
        ego_pose = features["bev_ego_pose"].detach().float().cpu().numpy()[0]
        route_feature = (
            features["route_feature"].detach().float().cpu().numpy()[0]
            if "route_feature" in features else np.zeros((0, 2), np.float32)
        )
        route_mask = (
            features["route_mask"].detach().float().cpu().numpy()[0]
            if "route_mask" in features else np.zeros((0,), np.float32)
        )
        trajectory_key = "final_traj" if "final_traj" in prediction else "trajectory"
        trajectory = prediction[trajectory_key].detach().float().cpu().numpy()[0]

        archive = {
            "bev": real.astype(np.float16),
            "images_rgb_0_1": imgs.astype(np.float16),
            "lidar2img": lidar2img,
            "bev_ego_pose": ego_pose,
            "route_feature_normalized": route_feature,
            "route_mask": route_mask,
            "trajectory": trajectory,
        }
        if blind is not None:
            archive["bev_blank_images"] = blind.astype(np.float16)
            archive["camera_sensitivity"] = sensitivity.astype(np.float32)
        np.savez_compressed(self.output_dir / f"{stem}.npz", **archive)

        pca = (_pca_rgb(real) * 255).astype(np.uint8)
        pca = cv2.cvtColor(pca, cv2.COLOR_RGB2BGR)
        pca = cv2.resize(pca, (560, 560), interpolation=cv2.INTER_NEAREST)
        cv2.putText(
            pca, "latent BEV PCA (forward up, ego bottom)", (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA,
        )
        cv2.imwrite(str(self.output_dir / f"{stem}_bev_pca.png"), pca)

        camera_views = []
        for name, image in zip(("CAM_L0", "CAM_F0", "CAM_R0"), imgs[-1]):
            rgb = np.clip(image.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.putText(
                bgr, name, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (0, 255, 0), 1, cv2.LINE_AA,
            )
            camera_views.append(bgr)
        cv2.imwrite(
            str(self.output_dir / f"{stem}_inputs.png"),
            np.concatenate(camera_views, axis=1),
        )

        metadata = {
            "replica": self.replica,
            "session_uuid": context,
            "inference_call": self.calls,
            "bev_shape": list(real.shape),
            "bev_finite_fraction": float(np.isfinite(real).mean()),
            "bev_std": float(np.nanstd(real)),
            "route_valid": int(route_mask.sum()),
            "trajectory_endpoint_m": float(np.linalg.norm(trajectory[-1, :2])),
            "camera_order": ["CAM_L0", "CAM_F0", "CAM_R0"],
            "temporal_order": "oldest_to_current: t-1.0,t-0.5,t",
            "pca_note": "latent 256-channel projection; colors are not semantic labels",
        }
        if sensitivity is not None:
            oriented = np.flipud(np.rot90(sensitivity, k=-1))
            upper = max(float(np.percentile(oriented, 99)), 1e-8)
            heat = cv2.applyColorMap(
                (np.clip(oriented / upper, 0, 1) * 255).astype(np.uint8),
                cv2.COLORMAP_MAGMA,
            )
            heat = cv2.resize(heat, (560, 560), interpolation=cv2.INTER_NEAREST)
            cv2.putText(
                heat, "camera sensitivity (forward up)", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA,
            )
            cv2.imwrite(
                str(self.output_dir / f"{stem}_camera_sensitivity.png"), heat
            )
            metadata.update({
                "camera_sensitivity_mean": float(sensitivity.mean()),
                "camera_sensitivity_p90": float(np.percentile(sensitivity, 90)),
                "camera_sensitivity_cells_above_0_1": float((sensitivity > 0.1).mean()),
                "camera_sensitivity_definition": (
                    "||BEV(real)-BEV(blank images)|| / ||BEV(real)||; "
                    "diagnostic and not bounded by 1"
                ),
            })

        (self.output_dir / f"{stem}.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        self.saved += 1
        print(f"[DriveSuprim] BEV DEBUG: saved {stem} to {self.output_dir}", flush=True)
