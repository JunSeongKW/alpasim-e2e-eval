"""Small inference-only adapter around the original DriveSuprim package."""

from __future__ import annotations

import json
import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
import torch

from .bev_debug import BevDebugger

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


@dataclass(frozen=True)
class DriveSuprimPrediction:
    """A future ego-relative trajectory, without the present pose."""

    poses: np.ndarray  # [T, 3]: forward x, left y, heading (radians)
    candidate_vocab: np.ndarray | None = None
    feasibility_mask: np.ndarray | None = None
    # Which rule rejected a candidate, so the overlay can tell the two apart
    # instead of painting every rejection the same colour. True = passed.
    feasibility_drivable_mask: np.ndarray | None = None
    feasibility_collision_mask: np.ndarray | None = None
    selected_index: int | None = None
    coarse_path: np.ndarray | None = None
    coarse_candidates: np.ndarray | None = None
    point_cloud_range: tuple[float, float, float, float, float, float] | None = None
    drivable_probability: np.ndarray | None = None
    detected_agents: np.ndarray | None = None
    detection_classes: tuple[str, ...] = ()
    # Ranking-score decomposition over the refinement candidates. `rank_terms`
    # maps each term of the ranking score to its per-candidate value [K], so a
    # chosen trajectory can be traced back to the term that selected it.
    rank_terms: dict[str, np.ndarray] | None = None
    rank_total: np.ndarray | None = None
    rank_selected: int | None = None
    rank_stats: dict[str, dict[str, float]] | None = None


def _apply_config_overrides(
    obj,
    values: dict,
    prefix: str = "config",
) -> None:
    """Recursively apply a resolved DriveSuprim YAML config to its dataclass.

    The generated cnx_stage3_config.json was written with resolve=False, so
    unresolved Hydra/OmegaConf interpolation strings are intentionally skipped.
    Runtime-critical paths such as vocab/checkpoint are set explicitly later.
    """

    for key, value in values.items():
        name = f"{prefix}.{key}"

        if not hasattr(obj, key):
            LOGGER.debug(
                "ignoring unknown DriveSuprim config key: %s",
                name,
            )
            continue

        current = getattr(obj, key)

        if isinstance(value, str) and "${" in value:
            LOGGER.debug(
                "skipping unresolved config interpolation: %s=%s",
                name,
                value,
            )
            continue

        if isinstance(value, dict):
            if (
                current is not None
                and (
                    is_dataclass(current)
                    or hasattr(current, "__dict__")
                )
            ):
                _apply_config_overrides(
                    current,
                    value,
                    prefix=name,
                )
            else:
                setattr(obj, key, value)
            continue

        setattr(obj, key, value)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(
        name,
        "1" if default else "0",
    )
    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


class DriveSuprimPolicy:
    """Load DriveSuprim and expose an AlpaSim-friendly prediction method."""

    output_frequency_hz = 10.0
    required_cameras = ("CAM_L0", "CAM_F0", "CAM_R0")

    def __init__(
        self,
        *,
        checkpoint_path: str,
        vocab_path: str,
        backbone_type: str = "auto",
        device: str = "cuda",
        vit_checkpoint_path: str = "",
        vov_checkpoint_path: str = "",
        config_path: str = "",
    ) -> None:
        from navsim.agents.drivesuprim.drivesuprim_agent import (
            DriveSuprimAgent,
        )
        from navsim.agents.drivesuprim.drivesuprim_config import (
            DriveSuprimConfig,
        )

        # Ranking decomposition is logged every Nth inference. 1 logs every
        # decision (~180 lines per clip); raise it to thin out long runs.
        self._rank_log_counter = 0
        # A candidate whose 4 s endpoint is within this radius is "staying put".
        self._stop_endpoint_m = float(
            os.environ.get("DRIVESUPRIM_STOP_ENDPOINT_M", "1.0")
        )
        self._rank_log_every = max(
            1,
            int(os.environ.get("DRIVESUPRIM_RANK_LOG_EVERY", "1")),
        )

        resolved = torch.device(
            device
            if torch.cuda.is_available()
            else "cpu"
        )

        if resolved.type != "cuda":
            LOGGER.warning(
                "CUDA is unavailable; DriveSuprim will run on CPU"
            )

        for path, label in (
            (checkpoint_path, "checkpoint"),
            (vocab_path, "vocabulary"),
        ):
            if not Path(path).is_file():
                raise FileNotFoundError(
                    f"DriveSuprim {label} not found: {path}"
                )

        if config_path and not Path(config_path).is_file():
            raise FileNotFoundError(
                f"DriveSuprim config not found: {config_path}"
            )

        detected_backbone = _detect_backbone(
            checkpoint_path
        )

        requested_backbone = backbone_type
        expected_checkpoint_backbone = (
            "resnet50"
            if requested_backbone == "bevformer_r50"
            else requested_backbone
        )

        if backbone_type == "auto":
            backbone_type = detected_backbone
            LOGGER.info(
                "detected DriveSuprim backbone=%s "
                "from checkpoint",
                backbone_type,
            )
        elif expected_checkpoint_backbone != detected_backbone:
            raise ValueError(
                "DriveSuprim checkpoint/backbone mismatch: "
                f"checkpoint uses {detected_backbone!r}, "
                f"but requested backbone is "
                f"{backbone_type!r}"
            )

        config = DriveSuprimConfig()

        # ----------------------------------------------------------
        # New CNX stage3 path:
        # apply the complete composed stage3 configuration.
        # ----------------------------------------------------------
        if config_path:
            LOGGER.info(
                "loading DriveSuprim config overrides from %s",
                config_path,
            )

            with open(
                config_path,
                "r",
                encoding="utf-8",
            ) as f:
                overrides = json.load(f)

            _apply_config_overrides(
                config,
                overrides,
            )

        else:
            # ------------------------------------------------------
            # Legacy fallback for the previous ViT/VoV/R34 adapter.
            # ------------------------------------------------------
            config.normalize_vocab_pos = True

            config.refinement.use_multi_stage = True
            config.refinement.num_refinement_stage = 1
            config.refinement.stage_layers = "3"
            config.refinement.topks = "256"
            config.refinement.use_mid_output = True
            config.refinement.use_separate_stage_heads = True
            config.refinement.use_imi_learning_in_refinement = True

            config.inference.model = "teacher"
            config.inference.use_first_stage_traj_in_infer = False

            # The released ViT checkpoint was trained with an 8x32 image
            # token grid (256 tokens), unlike the 16x64 legacy default.
            if backbone_type == "vit":
                config.camera_width = 1024
                config.camera_height = 256
                config.img_horz_anchors = 32
                config.img_vert_anchors = 8

        # ----------------------------------------------------------
        # Runtime inference overrides.
        # These values must not depend on training-time YAML paths.
        # ----------------------------------------------------------
        config.training = False
        config.only_ori_input = True

        # R50 stage3 is a BEVFormer model whose image encoder is ResNet-50.
        # Keep a distinct external alias for checkpoint validation, then use
        # the BEVFormer runtime path after the resolved config is applied.
        config.backbone_type = (
            "bevformer_m"
            if backbone_type == "bevformer_r50"
            else backbone_type
        )

        config.n_camera = 3

        if hasattr(config, "bev_num_cameras"):
            config.bev_num_cameras = 3

        config.vocab_path = vocab_path
        config.vocab_size = int(
            np.load(
                vocab_path,
                mmap_mode="r",
            ).shape[0]
        )

        config.inference.model = "teacher"
        config.inference.use_first_stage_traj_in_infer = False

        if config.backbone_type == "bevformer_m":
            # The final stage3 checkpoint already contains the image-backbone
            # weights. Do not let timm download pretrained
            # weights at container startup.
            config.bevformer_cnn_pretrained = False

            # Full DriveSuprim checkpoints include the ViT image backbone.
            # Avoid a redundant HuggingFace download during offline startup.
            if hasattr(config, "bevformer_vit_pretrained"):
                config.bevformer_vit_pretrained = False

            if config.vocab_size != 4096:
                raise ValueError(
                    "BEVFormer stage3 expects a 4096-trajectory "
                    f"vocabulary, got {config.vocab_size}"
                )

        if backbone_type == "vit":
            config.vit_ckpt = vit_checkpoint_path

        if backbone_type == "vov":
            config.vov_ckpt = vov_checkpoint_path

        print(
            "[DriveSuprim] RESOLVED CONFIG:"
            f" backbone={config.backbone_type}"
            f" image_backbone={getattr(config, 'bevformer_img_backbone_type', None)}"
            f" seq_len={getattr(config, 'seq_len', None)}"
            f" bev_seq_len={getattr(config, 'bev_seq_len', None)}"
            f" n_camera={getattr(config, 'n_camera', None)}"
            f" bev_num_cameras={getattr(config, 'bev_num_cameras', None)}"
            f" bev_img={getattr(config, 'bev_img_width', None)}x"
            f"{getattr(config, 'bev_img_height', None)}"
            f" bev_grid={getattr(config, 'bev_w', None)}x"
            f"{getattr(config, 'bev_h', None)}"
            f" vocab={config.vocab_size}"
            f" feasibility={getattr(config, 'feasibility_enabled', None)}"
            f" inference_model={getattr(config.inference, 'model', None)}"
            f" topks={getattr(config.refinement, 'topks', None)}",
            flush=True,
        )

        # Printed separately so a ranking-weight sweep can be told apart in the
        # logs: the run is only interpretable if the weights it actually used
        # are on the record, not just the file it was pointed at.
        print(
            "[DriveSuprim] RANK WEIGHTS:"
            f" product={getattr(config, 'pdm_rank_product', None)}"
            f" terms={getattr(config, 'pdm_rank_product_terms', None)}"
            f" exponents={getattr(config, 'pdm_rank_product_exponents', None)}"
            f" imi_weight={getattr(config, 'pdm_imi_rank_weight', None)}"
            f" imi_normalize={getattr(config, 'pdm_rank_imi_normalize', None)}"
            f" aggregate_weight={getattr(config, 'pdm_aggregate_rank_weight', None)}",
            flush=True,
        )

        LOGGER.info(
            "DriveSuprim resolved config: "
            "backbone=%s "
            "image_backbone=%s "
            "seq_len=%s "
            "bev_seq_len=%s "
            "n_camera=%s "
            "bev_num_cameras=%s "
            "bev_img=%sx%s "
            "bev_grid=%sx%s "
            "vocab=%s "
            "feasibility=%s "
            "inference_model=%s "
            "refine_topks=%s",
            config.backbone_type,
            getattr(config, "bevformer_img_backbone_type", None),
            getattr(config, "seq_len", None),
            getattr(config, "bev_seq_len", None),
            getattr(config, "n_camera", None),
            getattr(config, "bev_num_cameras", None),
            getattr(config, "bev_img_width", None),
            getattr(config, "bev_img_height", None),
            getattr(config, "bev_w", None),
            getattr(config, "bev_h", None),
            config.vocab_size,
            getattr(config, "feasibility_enabled", None),
            getattr(config.inference, "model", None),
            getattr(config.refinement, "topks", None),
        )

        agent = DriveSuprimAgent(
            config=config,
            lr=0.0,
            checkpoint_path=checkpoint_path,
            pdm_split=None,
            metrics=[],
        )

        if _env_flag(
            "DRIVESUPRIM_REQUIRE_EXACT_CHECKPOINT",
            default=False,
        ):
            checkpoint = torch.load(
                checkpoint_path,
                map_location=torch.device("cpu"),
            )
            state_dict = checkpoint.get("state_dict", checkpoint)
            incompatible = agent.load_state_dict(
                {
                    key.replace("agent.", "", 1): value
                    for key, value in state_dict.items()
                },
                strict=False,
            )
            if incompatible.missing_keys or incompatible.unexpected_keys:
                raise RuntimeError(
                    "DriveSuprim checkpoint/config mismatch: "
                    f"missing={incompatible.missing_keys[:20]} "
                    f"unexpected={incompatible.unexpected_keys[:20]}"
                )
            print(
                "[DriveSuprim] exact checkpoint load: missing=0 unexpected=0",
                flush=True,
            )
            del checkpoint, state_dict
        else:
            agent.initialize()

        print(
            "[DriveSuprim] moving model to device:",
            resolved,
            flush=True,
        )

        self._agent = (
            agent
            .eval()
            .to(resolved)
        )

        print(
            "[DriveSuprim] model ready on device:",
            resolved,
            flush=True,
        )

        self._config = config
        self._device = resolved

        use_fp16 = _env_flag(
            "DRIVESUPRIM_USE_FP16",
            default=False,
        )

        self._use_autocast = (
            resolved.type == "cuda"
            and use_fp16
        )

        LOGGER.info(
            "DriveSuprim precision: device=%s autocast_fp16=%s",
            resolved,
            self._use_autocast,
        )

        self._bev_debugger = BevDebugger(self._agent, self._config)

    def predict(
        self,
        camera_history: Sequence[
            Mapping[str, np.ndarray]
        ],
        status_history: Sequence[
            np.ndarray
        ],
        *,
        route_waypoints: np.ndarray | None = None,
        lidar2img_by_camera: (
            Mapping[str, np.ndarray]
            | None
        ) = None,
        bev_ego_pose: (
            np.ndarray
            | None
        ) = None,
        debug_context: str = "",
    ) -> DriveSuprimPrediction:
        if (
            not camera_history
            or not status_history
        ):
            raise ValueError(
                "camera and ego-status "
                "histories must not be empty"
            )

        if (
            self._config.backbone_type
            == "bevformer_m"
        ):
            return self._predict_bevformer(
                camera_history,
                status_history,
                route_waypoints=route_waypoints,
                lidar2img_by_camera=(
                    lidar2img_by_camera
                ),
                bev_ego_pose=(
                    bev_ego_pose
                ),
                debug_context=debug_context,
            )

        # ----------------------------------------------------------
        # Legacy ViT / VoV / ResNet path
        # ----------------------------------------------------------

        cameras = _pad_history(
            camera_history,
            self._config.seq_len,
        )

        statuses = list(
            reversed(
                _pad_history(
                    status_history,
                    self._config.seq_len,
                )
            )
        )

        if getattr(self._config, "use_route", False):
            # Training zeroes the four command slots whenever the route is the
            # intent signal (`_get_status_feature`), so a live command here
            # would feed `_status_encoding` a pattern it never saw.
            statuses = [
                np.asarray(item, dtype=np.float32).copy() for item in statuses
            ]
            for item in statuses:
                item[:4] = 0.0

        images = [
            self._preprocess_camera_set(
                item
            )
            for item in cameras
        ]

        status_tensors = [
            torch.as_tensor(
                item,
                dtype=torch.float32,
                device=self._device,
            ).reshape(
                1,
                8,
            )
            for item in statuses
        ]

        features = {
            "ori_teacher": [
                image
                .unsqueeze(0)
                .to(self._device)
                for image in images
            ],
            "ori": [
                image
                .unsqueeze(0)
                .to(self._device)
                for image in images
            ],
            "rotated": [],
            "status_feature": (
                status_tensors
            ),
        }

        if getattr(self._config, "use_route", False):
            # The flat front-end reads the route through the same tokens the
            # BEV one does; without these the model raises rather than quietly
            # driving blind (`_route_inputs`).
            route_feature, route_mask = _prepare_route_features(
                route_waypoints,
                slots=20,
                norm_m=float(getattr(self._config, "route_norm_m", 80.0)),
            )
            features["route_feature"] = torch.as_tensor(
                route_feature,
                dtype=torch.float32,
                device=self._device,
            ).unsqueeze(0)
            features["route_mask"] = torch.as_tensor(
                route_mask,
                dtype=torch.float32,
                device=self._device,
            ).unsqueeze(0)

        print(
            "[DriveSuprim] FLAT INPUT:"
            f" imgs={tuple(images[0].shape)} x{len(images)}"
            f" status={len(status_tensors)}"
            + (
                f" route_valid={int(features['route_mask'].sum().item())}"
                if "route_mask" in features
                else " route=off"
            ),
            flush=True,
        )

        prediction = (
            self._run_agent(
                features
            )
        )

        return self._prediction_to_result(
            prediction,
            statuses,
        )

    def _predict_bevformer(
        self,
        camera_history,
        status_history,
        *,
        route_waypoints,
        lidar2img_by_camera,
        bev_ego_pose,
        debug_context,
    ) -> DriveSuprimPrediction:
        """Build exact CNX stage3 BEVFormer inference tensors."""

        if lidar2img_by_camera is None:
            raise ValueError(
                "BEVFormer requires "
                "lidar2img_by_camera"
            )

        if bev_ego_pose is None:
            raise ValueError(
                "BEVFormer requires "
                "bev_ego_pose"
            )

        bev_seq_len = int(
            self._config.bev_seq_len
        )

        frames = _pad_history(
            camera_history,
            bev_seq_len,
        )

        camera_order = (
            "CAM_L0",
            "CAM_F0",
            "CAM_R0",
        )

        frame_tensors = []

        for frame in frames:
            per_camera = []

            for camera_name in (
                camera_order
            ):
                if camera_name not in frame:
                    raise ValueError(
                        "missing BEV camera "
                        f"{camera_name}"
                    )

                image = np.asarray(
                    frame[camera_name]
                )

                expected_hw = (
                    int(
                        self
                        ._config
                        .bev_img_height
                    ),
                    int(
                        self
                        ._config
                        .bev_img_width
                    ),
                )

                if (
                    image.shape[:2]
                    != expected_hw
                ):
                    image = cv2.resize(
                        image,
                        (
                            expected_hw[1],
                            expected_hw[0],
                        ),
                        interpolation=(
                            cv2.INTER_LINEAR
                        ),
                    )

                if (
                    image.ndim != 3
                    or image.shape[2] != 3
                ):
                    raise ValueError(
                        f"{camera_name}: "
                        "expected RGB image, "
                        f"got {image.shape}"
                    )

                tensor = (
                    torch
                    .from_numpy(
                        np.ascontiguousarray(
                            image
                        )
                    )
                    .permute(
                        2,
                        0,
                        1,
                    )
                    .float()
                    / 255.0
                )

                per_camera.append(
                    tensor
                )

            frame_tensors.append(
                torch.stack(
                    per_camera,
                    dim=0,
                )
            )

        # [1, T, Ncam, 3, H, W]
        bev_imgs = (
            torch.stack(
                frame_tensors,
                dim=0,
            )
            .unsqueeze(0)
            .to(
                self._device,
                non_blocking=True,
            )
        )

        base_lidar2img = np.stack(
            [
                np.asarray(
                    lidar2img_by_camera[
                        name
                    ],
                    dtype=np.float32,
                )
                for name in camera_order
            ],
            axis=0,
        )

        if (
            base_lidar2img.shape
            != (3, 4, 4)
        ):
            raise ValueError(
                "unexpected lidar2img "
                f"shape={base_lidar2img.shape}"
            )

        # Static rig calibration is shared by each temporal frame.
        lidar2img = np.repeat(
            base_lidar2img[
                None,
                ...,
            ],
            bev_seq_len,
            axis=0,
        )

        # [1, T, Ncam, 4, 4]
        lidar2img_tensor = (
            torch.as_tensor(
                lidar2img,
                dtype=torch.float32,
                device=self._device,
            )
            .unsqueeze(0)
        )

        bev_pose = np.asarray(
            bev_ego_pose,
            dtype=np.float32,
        )

        if (
            bev_pose.shape
            != (bev_seq_len, 3)
        ):
            raise ValueError(
                "unexpected bev_ego_pose "
                f"shape={bev_pose.shape}, "
                f"expected="
                f"({bev_seq_len}, 3)"
            )

        bev_pose_tensor = (
            torch.as_tensor(
                bev_pose,
                dtype=torch.float32,
                device=self._device,
            )
            .unsqueeze(0)
        )

        # Ordinary DriveSuprim status path remains seq_len=2 and
        # uses [present, older] order.
        statuses = list(
            reversed(
                _pad_history(
                    status_history,
                    self._config.seq_len,
                )
            )
        )

        if getattr(self._config, "use_route", False):
            # The route replaces the four-way command during training, while
            # retaining the checkpoint's eight-dimensional status projection.
            statuses = [np.asarray(item, dtype=np.float32).copy() for item in statuses]
            for item in statuses:
                item[:4] = 0.0

        status_tensors = [
            torch.as_tensor(
                item,
                dtype=torch.float32,
                device=self._device,
            ).reshape(
                1,
                8,
            )
            for item in statuses
        ]

        # inference.model=teacher, therefore use the clean rectified
        # tensors for both branches during closed-loop deployment.
        features = {
            "bev_imgs_teacher": (
                bev_imgs
            ),
            "bev_imgs": (
                bev_imgs
            ),
            "lidar2img": (
                lidar2img_tensor
            ),
            "bev_ego_pose": (
                bev_pose_tensor
            ),
            "status_feature": (
                status_tensors
            ),
        }

        if getattr(self._config, "use_route", False):
            route_feature, route_mask = _prepare_route_features(
                route_waypoints,
                slots=20,
                norm_m=float(getattr(self._config, "route_norm_m", 80.0)),
            )
            features["route_feature"] = torch.as_tensor(
                route_feature,
                dtype=torch.float32,
                device=self._device,
            ).unsqueeze(0)
            features["route_mask"] = torch.as_tensor(
                route_mask,
                dtype=torch.float32,
                device=self._device,
            ).unsqueeze(0)

        print(
            "[DriveSuprim] BEV INPUT:"
            f" imgs={tuple(bev_imgs.shape)}"
            f" lidar2img="
            f"{tuple(lidar2img_tensor.shape)}"
            f" ego_pose="
            f"{tuple(bev_pose_tensor.shape)}"
            f" status={len(status_tensors)}",
            f" route_valid={int(features['route_mask'].sum().item())}"
            if "route_mask" in features
            else " route=off",
            flush=True,
        )

        prediction = (
            self._run_agent(
                features,
                debug_context=debug_context,
            )
        )

        return self._prediction_to_result(
            prediction,
            statuses,
        )

    def _run_agent(
        self,
        features,
        *,
        debug_context: str = "",
    ):
        autocast = (
            torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
            )
            if self._use_autocast
            else nullcontext()
        )

        capture_debug = self._bev_debugger.should_capture()
        captured = []
        hook = (
            self._bev_debugger.install_hook(captured)
            if capture_debug else None
        )

        try:
            with (
                torch.inference_mode(),
                autocast,
            ):
                prediction, _ = self._agent(
                    (
                        features,
                        None,
                        None,
                    )
                )

                if capture_debug and self._bev_debugger.ablation:
                    blind_features = self._bev_debugger.blind_features(features)
                    self._agent(
                        (
                            blind_features,
                            None,
                            None,
                        )
                    )
        finally:
            if hook is not None:
                hook.remove()

        if capture_debug:
            try:
                self._bev_debugger.save(
                    features, prediction, captured, context=debug_context
                )
            except Exception:
                LOGGER.exception("failed to write BEV debug capture")

        return prediction

    def _prediction_to_result(
        self,
        prediction,
        statuses,
    ) -> DriveSuprimPrediction:
        key = (
            "trajectory"
            if (
                self
                ._config
                .inference
                .use_first_stage_traj_in_infer
            )
            else "final_traj"
        )

        if key not in prediction:
            key = "trajectory"

        poses = (
            prediction[key]
            .detach()
            .float()
            .cpu()
            .numpy()
        )

        poses = np.asarray(
            poses
        ).reshape(
            -1,
            poses.shape[-1],
        )

        if poses.shape[1] < 3:
            raise ValueError(
                "unexpected DriveSuprim "
                "trajectory shape: "
                f"{poses.shape}"
            )

        poses = poses[:, :3]

        LOGGER.info(
            "DriveSuprim %s: "
            "velocity=(%.2f, %.2f)m/s "
            "acceleration=(%.2f, %.2f)m/s2 "
            "endpoint=%.2fm",
            key,
            statuses[0][4],
            statuses[0][5],
            statuses[0][6],
            statuses[0][7],
            float(
                np.linalg.norm(
                    poses[-1, :2]
                )
            ),
        )

        print(
            "[DriveSuprim] prediction:"
            f" key={key}"
            f" shape={poses.shape}"
            f" endpoint="
            f"{float(np.linalg.norm(poses[-1, :2])):.3f}m",
            flush=True,
        )

        candidate_vocab = _tensor_numpy(
            prediction.get("trajectory_vocab")
        )
        if candidate_vocab is not None:
            candidate_vocab = np.asarray(candidate_vocab)
            if candidate_vocab.ndim == 4 and candidate_vocab.shape[0] == 1:
                candidate_vocab = candidate_vocab[0]
            if candidate_vocab.ndim != 3 or candidate_vocab.shape[-1] < 2:
                LOGGER.warning(
                    "ignoring trajectory_vocab shape %s",
                    candidate_vocab.shape,
                )
                candidate_vocab = None

        def _bool_mask(key):
            v = _tensor_numpy(prediction.get(key))
            return None if v is None else np.asarray(v, dtype=np.bool_).reshape(-1)

        feas_drivable = _bool_mask("feasibility_drivable_mask")
        feas_collision = _bool_mask("feasibility_collision_mask")
        feasibility_mask = _tensor_numpy(
            prediction.get("feasibility_mask")
        )
        if feasibility_mask is not None:
            feasibility_mask = np.asarray(
                feasibility_mask,
                dtype=np.bool_,
            ).reshape(-1)
        elif candidate_vocab is not None:
            feasibility_mask = np.ones(
                candidate_vocab.shape[0],
                dtype=np.bool_,
            )

        # Gate accounting. A mask that survives everything is indistinguishable
        # from no mask at all in the overlay, so record which of the two it is:
        # `source=model` means the network returned one, `source=fallback` means
        # it did not and every candidate is being reported as feasible.
        if feasibility_mask is not None:
            n_total = int(feasibility_mask.size)
            n_keep = int(feasibility_mask.sum())
            src = "model" if prediction.get("feasibility_mask") is not None else "fallback"
            self._feas_log_counter = getattr(self, "_feas_log_counter", 0) + 1
            if self._feas_log_counter % self._rank_log_every == 0:
                dr = ("n/a" if feas_drivable is None
                      else f"{int((~feas_drivable).sum())}")
                co = ("n/a" if feas_collision is None
                      else f"{int((~feas_collision).sum())}")
                print(
                    f"[DriveSuprim] FEASGATE: source={src} kept={n_keep}/{n_total}"
                    f" ({100.0 * n_keep / max(n_total, 1):.1f}%)"
                    f" dropped={n_total - n_keep}"
                    f" by_drivable={dr} by_collision={co}",
                    flush=True,
                )

        selected = _tensor_numpy(
            prediction.get("selected_indices")
        )
        selected_index = (
            int(np.asarray(selected).reshape(-1)[0])
            if selected is not None
            else None
        )
        coarse_path = _tensor_numpy(
            prediction.get("trajectory")
        )
        if coarse_path is not None:
            coarse_path = np.asarray(coarse_path).reshape(
                -1,
                coarse_path.shape[-1],
            )[:, :3]

        coarse_candidates = None
        refinement = prediction.get("refinement")
        if isinstance(refinement, (list, tuple)) and refinement:
            coarse_candidates = _tensor_numpy(refinement[0].get("trajs"))
            if coarse_candidates is not None:
                coarse_candidates = np.asarray(coarse_candidates)
                if coarse_candidates.ndim == 4 and coarse_candidates.shape[0] == 1:
                    coarse_candidates = coarse_candidates[0]
                if (
                    coarse_candidates.ndim != 3
                    or coarse_candidates.shape[-1] < 2
                ):
                    LOGGER.warning(
                        "ignoring refinement coarse candidate shape %s",
                        coarse_candidates.shape,
                    )
                    coarse_candidates = None
                else:
                    coarse_candidates = coarse_candidates[:, :, :3]

        pc_range = getattr(
            self._config,
            "point_cloud_range",
            None,
        )
        point_cloud_range = (
            tuple(float(value) for value in pc_range)
            if pc_range
            else None
        )

        (
            rank_terms,
            rank_total,
            rank_selected,
            rank_stats,
        ) = self._rank_decomposition(prediction, poses)

        return DriveSuprimPrediction(
            poses=poses,
            candidate_vocab=candidate_vocab,
            feasibility_mask=feasibility_mask,
            feasibility_drivable_mask=feas_drivable,
            feasibility_collision_mask=feas_collision,
            selected_index=selected_index,
            coarse_path=coarse_path,
            coarse_candidates=coarse_candidates,
            point_cloud_range=point_cloud_range,
            drivable_probability=(
                _extract_drivable_probability(prediction)
            ),
            detected_agents=(
                _extract_detected_agents(
                    prediction,
                    self._config,
                )
            ),
            detection_classes=tuple(
                str(name)
                for name in getattr(
                    self._config,
                    "aux_agent_classes",
                    (),
                )
            ),
            rank_terms=rank_terms,
            rank_total=rank_total,
            rank_selected=rank_selected,
            rank_stats=rank_stats,
        )

    def _rank_decomposition(self, prediction, poses):
        """Break the ranking score into its terms over the refinement candidates.

        Returns (terms, total, selected, stats), all None when the model did not
        expose per-candidate head outputs (non-refinement backbones).
        """

        refinement = prediction.get("refinement")
        if not isinstance(refinement, (list, tuple)) or not refinement:
            return None, None, None, None

        stage = refinement[-1]
        layer_results = stage.get("layer_results")
        heads_src = None
        if isinstance(layer_results, (list, tuple)) and layer_results:
            heads_src = layer_results[-1]
        elif isinstance(refinement[0].get("coarse_score"), dict):
            heads_src = refinement[0]["coarse_score"]
        if not isinstance(heads_src, dict) or not heads_src:
            return None, None, None, None

        heads = {}
        for name, value in heads_src.items():
            array = _tensor_numpy(value)
            if array is None:
                continue
            array = np.asarray(array, dtype=np.float64).reshape(-1)
            if array.size:
                heads[name] = array
        sizes = {value.size for value in heads.values()}
        if not heads or len(sizes) != 1:
            return None, None, None, None

        terms = _rank_term_breakdown(self._config, heads)
        if not terms:
            return None, None, None, None

        if getattr(self._config, "pdm_rank_product", False):
            # Factors multiply; imi and the aggregate head are added on top,
            # exactly as `_rank_product` and its caller assemble them.
            additive = {"imi", "pdm_score"}
            total = None
            for name, value in terms.items():
                if name in additive:
                    continue
                total = value if total is None else total * value
            if total is None:
                total = np.zeros_like(next(iter(terms.values())))
            for name in additive:
                if name in terms:
                    total = total + terms[name]
        else:
            total = None
            for value in terms.values():
                total = value if total is None else total + value
        selected = int(np.argmax(total))

        # The refinement head re-ranks the surviving vocabulary entries without
        # moving them, so the driven trajectory must BE the argmax candidate.
        # Verify rather than assume: a mismatch means this decomposition is
        # describing a different decision than the one that was driven.
        candidates = _tensor_numpy(refinement[-1].get("trajs"))
        if candidates is not None:
            candidates = np.asarray(candidates)
            if candidates.ndim == 4 and candidates.shape[0] == 1:
                candidates = candidates[0]
            if (
                candidates.ndim == 3
                and candidates.shape[0] == total.size
                and poses is not None
            ):
                gap = float(
                    np.abs(
                        candidates[selected, :, : poses.shape[-1]] - poses
                    ).max()
                )
                if gap > 1e-3:
                    LOGGER.warning(
                        "rank argmax candidate differs from the driven "
                        "trajectory by %.4f m; decomposition may not describe "
                        "the executed choice",
                        gap,
                    )

        stats = _rank_term_stats(
            terms,
            total,
            selected,
            product_mode=bool(getattr(self._config, "pdm_rank_product", False)),
        )

        self._rank_log_counter = getattr(self, "_rank_log_counter", 0) + 1
        if self._rank_log_counter % self._rank_log_every == 0:
            print(
                _rank_debug_line(stats, selected, total, int(total.size)),
                flush=True,
            )
            if (
                candidates is not None
                and candidates.ndim == 3
                and candidates.shape[0] == total.size
            ):
                line = _stop_margin_line(
                    terms,
                    total,
                    selected,
                    candidates,
                    self._stop_endpoint_m,
                    product_mode=bool(
                        getattr(self._config, "pdm_rank_product", False)
                    ),
                )
                if line:
                    print(line, flush=True)

        return terms, total, selected, stats

    def _preprocess_camera_set(
        self,
        images: Mapping[str, np.ndarray],
    ) -> torch.Tensor:
        """Legacy stitched-camera preprocessing.

        This is retained only for the existing non-BEV models.
        """

        missing = [
            camera
            for camera in self.required_cameras
            if camera not in images
        ]

        if missing:
            raise ValueError(
                f"missing DriveSuprim cameras: {missing}"
            )

        stitched = _stitch_three_cameras(
            images["CAM_L0"],
            images["CAM_F0"],
            images["CAM_R0"],
        )

        resized = cv2.resize(
            stitched,
            (
                self._config.camera_width,
                self._config.camera_height,
            ),
            interpolation=cv2.INTER_LINEAR,
        )

        return (
            torch
            .from_numpy(
                np.ascontiguousarray(resized)
            )
            .permute(2, 0, 1)
            .float()
            / 255.0
        )


def _tensor_numpy(value) -> np.ndarray | None:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value)


def _extract_drivable_probability(
    prediction,
) -> np.ndarray | None:
    logits = prediction.get(
        "bev_drivable_map",
        prediction.get("bev_seg_map"),
    )
    if logits is None or not torch.is_tensor(logits):
        return None
    logits = logits.detach().float()
    if (
        logits.ndim != 4
        or logits.shape[0] < 1
        or logits.shape[1] < 2
    ):
        LOGGER.warning(
            "ignoring BEV segmentation shape %s",
            tuple(logits.shape),
        )
        return None
    probability = (
        logits[0]
        .softmax(dim=0)[1]
        .cpu()
        .numpy()
    )
    return np.rint(
        np.clip(probability, 0.0, 1.0) * 255.0
    ).astype(np.uint8)


def _extract_detected_agents(
    prediction,
    config,
) -> np.ndarray | None:
    states = _tensor_numpy(
        prediction.get("agent_states")
    )
    labels = _tensor_numpy(
        prediction.get("agent_labels")
    )
    if states is None or labels is None:
        return None
    states = np.asarray(states, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    if states.ndim == 3:
        states = states[0]
    if labels.ndim == 3:
        labels = labels[0]
    if (
        states.ndim != 2
        or labels.ndim not in (1, 2)
        or len(states) != len(labels)
    ):
        LOGGER.warning(
            "ignoring detection shapes states=%s labels=%s",
            states.shape,
            labels.shape,
        )
        return None

    probabilities = 1.0 / (
        1.0 + np.exp(-np.clip(labels, -30.0, 30.0))
    )
    if probabilities.ndim == 2:
        class_indices = probabilities.argmax(axis=1)
        confidences = probabilities.max(axis=1)
    else:
        class_indices = np.zeros(
            len(probabilities),
            dtype=np.int64,
        )
        confidences = probabilities

    threshold = float(
        getattr(
            config,
            "feasibility_agent_conf_thresh",
            0.3,
        )
    )
    keep = confidences >= threshold
    states = states[keep]
    class_indices = class_indices[keep]
    confidences = confidences[keep]
    if not len(states):
        return np.empty((0, 7), dtype=np.float32)

    if (
        states.shape[1] >= 10
        and getattr(config, "aux_agent_box_3d", False)
    ):
        yaw = np.arctan2(states[:, 6], states[:, 7])
        length = np.exp(np.clip(states[:, 2], -5.0, 5.0))
        width = np.exp(np.clip(states[:, 3], -5.0, 5.0))
        boxes = np.column_stack(
            (states[:, 0], states[:, 1], yaw, length, width)
        )
    elif states.shape[1] >= 5:
        boxes = states[:, :5]
    else:
        LOGGER.warning(
            "ignoring agent state width %d",
            states.shape[1],
        )
        return None

    result = np.column_stack(
        (boxes, class_indices, confidences)
    ).astype(np.float32)
    valid = (
        np.all(np.isfinite(result), axis=1)
        & (result[:, 3] > 0.05)
        & (result[:, 4] > 0.05)
    )
    return result[valid]


def _pad_history(
    items: Sequence,
    size: int,
) -> list:
    values = list(items)[-size:]

    if not values:
        raise ValueError(
            "cannot pad an empty history"
        )

    return (
        [values[0]] * (size - len(values))
        + values
    )


def _detect_backbone(
    checkpoint_path: str,
) -> str:
    """Infer the architecture from checkpoint parameter names."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    state_dict = checkpoint.get(
        "state_dict",
        checkpoint,
    )

    keys = tuple(state_dict)

    encoder_marker = "._backbone.image_encoder."

    encoder_keys = [
        key
        for key in keys
        if encoder_marker in key
    ]

    # DINOv3 ViT-S / BEVFormer-M. This is distinct from the legacy stitched
    # ViT path below: it lives under image_backbone.vit and consumes three
    # calibrated camera tensors.
    if any(
        ".image_backbone.vit.patch_embed.proj.weight" in key
        for key in encoder_keys
    ):
        return "bevformer_m"

    # ConvNeXt V2-Tiny / BEVFormer-M.
    if any(
        ".image_backbone.stages_0.blocks.0.conv_dw.weight"
        in key
        for key in encoder_keys
    ):
        return "bevformer_m"

    # DINOv3 ViT-S, flat. DriveSuprim's original front-end: three rectified
    # cameras stitched into one panorama and read directly, with no BEV
    # projection, so the encoder holds the ViT at its top level instead of
    # under `image_backbone`. That one level is the whole difference from the
    # bevformer_m case above, and the two substrings cannot both match.
    if any(
        ".image_encoder.vit.patch_embed.proj.weight"
        in key
        for key in encoder_keys
    ):
        return "vits_flat"

    # ViT.
    if any(
        ".pretrained.patch_embed.proj.weight"
        in key
        for key in encoder_keys
    ):
        return "vit"

    # VoVNet.
    if any(
        ".stem.stem_1/conv.weight"
        in key
        for key in encoder_keys
    ):
        return "vov"

    # ResNet-50.
    if any(
        ".layer1.0.conv3.weight"
        in key
        for key in encoder_keys
    ):
        return "resnet50"

    # ResNet-34.
    if any(
        ".layer1.0.conv1.weight"
        in key
        for key in encoder_keys
    ):
        return "resnet34"

    raise ValueError(
        "cannot identify the DriveSuprim backbone "
        "from checkpoint parameter names: "
        f"{checkpoint_path}"
    )


# Ranking-score decomposition -------------------------------------------------
#
# The model does not pick the trajectory it drives by any single head. It ranks
# candidates by a weighted sum whose terms live on different scales: three
# multiplicative gates enter as w*log(sigmoid(x)), progress as a log of a
# weighted sigmoid, imitation as a log-softmax, and the aggregate head as a bare
# sigmoid bonus. Reading the head outputs alone therefore says nothing about
# which one decided. What decides is a term's *spread across candidates*: a term
# that is nearly constant over the 256 survivors cannot reorder them however
# large its value, while a term with a wide spread moves the argmax.
#
# These reproduce `navsim.agents.drivesuprim.drivesuprim_model._rank_score`
# term by term. Keep them in step with it.

_RANK_TERM_LABELS = {
    "imi": "il",
    "no_at_fault_collisions": "nc",
    "drivable_area_compliance": "dac",
    "gt_compliance": "gt",
    "ego_progress": "ep",
    "pdm_score": "agg",
}


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return -np.logaddexp(0.0, -x)


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    shifted = x - x.max()
    return shifted - np.log(np.exp(shifted).sum())


def _rank_product_breakdown(config, heads: dict) -> dict:
    """Factors of the product ranking, one array per head."""

    terms: dict = {}
    exponents = getattr(config, "pdm_rank_product_exponents", None) or {}
    for name in getattr(config, "pdm_rank_product_terms", ()):
        if name not in heads:
            continue
        if name == "traffic_light_compliance" and not getattr(
            config, "use_traffic_light_compliance", False
        ):
            continue
        if name == "imi":
            value = _softmax(heads[name])
        else:
            value = _sigmoid(heads[name])
        exponent = float(exponents.get(name, 1.0))
        if exponent != 1.0:
            value = np.clip(value, 1e-6, None) ** exponent
        terms[name] = value

    if "imi" in heads and "imi" not in terms:
        weight = float(getattr(config, "pdm_imi_rank_weight", 0.02))
        if weight:
            terms["imi"] = weight * _softmax(heads["imi"])

    if (
        getattr(config, "pdm_aggregate_head", False)
        and "pdm_score" in heads
        and getattr(config, "pdm_aggregate_rank_weight", 0.0)
    ):
        weight = float(config.pdm_aggregate_rank_weight)
        terms["pdm_score"] = weight * _sigmoid(heads["pdm_score"])

    return terms


def _softmax(x: np.ndarray) -> np.ndarray:
    return np.exp(_log_softmax(x))


def _rank_term_breakdown(config, heads: dict) -> dict:
    """Per-candidate value of every term in the ranking score.

    Mirrors `_rank_score`, including the `pdm_rank_product` branch. In product
    mode the terms are FACTORS, not addends: the reported value is the factor
    itself and the total is their product plus the imitation bonus. The
    breakdown stays meaningful because influence is measured as spread, and a
    factor that is near-constant across candidates cannot reorder them whether
    it multiplies or adds.
    """

    if getattr(config, "pdm_rank_product", False):
        return _rank_product_breakdown(config, heads)

    terms: dict = {}

    if "imi" in heads:
        weight = float(getattr(config, "pdm_imi_rank_weight", 0.02))
        if weight:
            terms["imi"] = weight * _log_softmax(heads["imi"])

    for name, weight in (getattr(config, "pdm_score_log", {}) or {}).items():
        if name in heads and weight:
            if name == "traffic_light_compliance" and not getattr(
                config, "use_traffic_light_compliance", False
            ):
                continue
            terms[name] = float(weight) * _log_sigmoid(heads[name])

    inner = None
    for name, weight in (getattr(config, "pdm_score_sum", {}) or {}).items():
        if name in heads and weight:
            value = float(weight) * _sigmoid(heads[name])
            inner = value if inner is None else inner + value
    if inner is not None:
        scale = float(getattr(config, "pdm_score_sum_scale", 1.0))
        sum_name = next(
            iter(getattr(config, "pdm_score_sum", {}) or {"ego_progress": 1.0})
        )
        terms[sum_name] = scale * np.log(np.clip(inner, 1e-12, None))

    if getattr(config, "pdm_aggregate_head", False) and "pdm_score" in heads:
        weight = float(getattr(config, "pdm_aggregate_rank_weight", 1.0))
        terms["pdm_score"] = weight * _sigmoid(heads["pdm_score"])

    return terms


def _rank_term_stats(terms: dict, total: np.ndarray, selected: int,
                    product_mode: bool = False) -> dict:
    """Quantify how much each term drove this frame's choice.

    value      the term's value for the trajectory that was driven
    spread     std of the term over the candidates -- its capacity to reorder
    influence  that spread as a share of all terms' spread (percent)
    loo_pick   the candidate the model would drive with this term removed
    loo_flip   whether removing the term changes the choice
    solo_rank  where the driven candidate ranks under this term ALONE (1 = best)
    """

    spreads = {name: float(np.std(value)) for name, value in terms.items()}
    spread_sum = sum(spreads.values()) or 1.0

    stats: dict = {}
    for name, value in terms.items():
        if product_mode and name not in ("imi", "pdm_score"):
            # Removing a factor means dividing it out, not subtracting it.
            without = total / np.clip(value, 1e-9, None)
        else:
            without = total - value
        loo_pick = int(np.argmax(without))
        order = np.argsort(-value)
        solo_rank = int(np.where(order == selected)[0][0]) + 1
        stats[name] = {
            "value": float(value[selected]),
            "spread": spreads[name],
            "influence": 100.0 * spreads[name] / spread_sum,
            "loo_pick": loo_pick,
            "loo_flip": bool(loo_pick != selected),
            "solo_rank": solo_rank,
        }
    return stats


def _stop_margin_line(
    terms: dict,
    total: np.ndarray,
    selected: int,
    candidates: np.ndarray,
    stop_endpoint_m: float,
    product_mode: bool = False,
) -> str | None:
    """Why the model did not stop, decomposed term by term.

    Finds the most stop-like candidate still on the shortlist and splits the
    score gap between it and the trajectory actually driven. Each term's share
    of that gap is `term(driven) - term(stop)`: positive means the term pushed
    the driven candidate ahead of stopping, negative means it preferred to stop
    and was outvoted. The shares sum to the gap, so this is an identity rather
    than an attribution heuristic.
    """

    endpoints = np.linalg.norm(candidates[:, -1, :2], axis=1)
    stop_mask = endpoints < stop_endpoint_m
    if not stop_mask.any():
        return (
            f"[DriveSuprim] RANKSTOP k={int(total.size)} stop=0 "
            f"driven_end={endpoints[selected]:.1f}m "
            "| no stop-like candidate survived the coarse cut"
        )

    stop_indices = np.flatnonzero(stop_mask)
    # Among stop-like candidates, the one the ranking liked most -- the
    # strongest case for stopping that the model actually had available.
    best_stop = int(stop_indices[np.argmax(total[stop_indices])])

    order_total = np.argsort(-total)
    stop_rank = int(np.where(order_total == best_stop)[0][0]) + 1
    imi = terms.get("imi")
    if imi is not None:
        order_imi = np.argsort(-imi)
        stop_il_rank = int(np.where(order_imi == best_stop)[0][0]) + 1
    else:
        stop_il_rank = 0

    gap = float(total[selected] - total[best_stop])
    shares = []
    for name in (
        "imi",
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "gt_compliance",
        "ego_progress",
        "pdm_score",
    ):
        if name not in terms:
            continue
        label = _RANK_TERM_LABELS.get(name, name)
        if product_mode and name not in ("imi", "pdm_score"):
            # Factors compose by multiplication, so their share of the gap is a
            # ratio: >1 favoured the driven candidate, <1 favoured stopping.
            # These multiply to the total ratio rather than summing to the gap.
            ratio = float(terms[name][selected]) / max(
                float(terms[name][best_stop]), 1e-9
            )
            shares.append(f"{label}=x{ratio:.3f}")
        else:
            share = float(terms[name][selected] - terms[name][best_stop])
            shares.append(f"{label}={share:+.3f}")

    return (
        f"[DriveSuprim] RANKSTOP k={int(total.size)} stop={int(stop_mask.sum())} "
        f"driven_end={endpoints[selected]:.1f}m best_stop=#{best_stop} "
        f"stop_end={endpoints[best_stop]:.2f}m stop_rank={stop_rank} "
        f"stop_il_rank={stop_il_rank} gap={gap:+.4f} | " + " ".join(shares)
    )


def _rank_debug_line(stats: dict, selected: int, total: np.ndarray, count: int) -> str:
    parts = []
    for name in (
        "imi",
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "gt_compliance",
        "ego_progress",
        "pdm_score",
    ):
        if name not in stats:
            continue
        label = _RANK_TERM_LABELS.get(name, name)
        entry = stats[name]
        parts.append(
            f"{label}={entry['value']:+.3f}({entry['influence']:.1f}%"
            f",r{entry['solo_rank']})"
        )
    flips = ",".join(
        _RANK_TERM_LABELS.get(name, name)
        for name, entry in stats.items()
        if entry["loo_flip"]
    )
    return (
        f"[DriveSuprim] RANKDBG k={count} sel={selected} "
        f"total={float(total[selected]):+.4f} | " + " ".join(parts)
        + f" | flip={flips or '-'}"
    )


def _prepare_route_features(
    route_waypoints: np.ndarray | None,
    *,
    slots: int,
    norm_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert AlpaSim's prepared rig-frame route to DriveSuprim tensors."""

    if slots <= 0 or norm_m <= 0.0:
        raise ValueError(f"invalid route contract: slots={slots} norm_m={norm_m}")

    padded = np.full((slots, 2), np.nan, dtype=np.float32)
    if route_waypoints is not None:
        points = np.asarray(route_waypoints, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 2:
            raise ValueError(
                "route_waypoints must have shape [N, >=2], "
                f"got {points.shape}"
            )
        count = min(slots, len(points))
        padded[:count] = points[:count, :2]

    mask = np.isfinite(padded).all(axis=1)

    # AlpaSim normally enforces this before submit_route. Re-check it at the
    # model boundary so malformed tails cannot silently enter the attention.
    valid_indices = np.flatnonzero(mask)
    if len(valid_indices) > 1:
        for left, right in zip(valid_indices[:-1], valid_indices[1:]):
            if right != left + 1:
                mask[right:] = False
                break
            gap_m = float(np.linalg.norm(padded[right] - padded[left]))
            if not 3.5 <= gap_m <= 4.5:
                mask[right:] = False
                break

    padded[~mask] = 0.0
    return padded / np.float32(norm_m), mask.astype(np.float32)


def _stitch_three_cameras(
    left: np.ndarray,
    front: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """L0 | F0 | R0 into one 4:1 panorama, the way training builds it.

    Mirrors `DriveSuprimFeatureBuilder._get_camera_feature`. The literals in the
    original are nuPlan's 1920x1080 -- a 28-row vertical crop and 416 columns off
    each side view. NuRec ships rectified 512x256, where `[416:-416]` is a
    negative width, so the crop is derived from the frame instead: keep the front
    view whole and drop `(3W - 4H) / 4` columns from each side view, which lands
    the stitch on exactly 4:1. At 512x256 that is 128 columns and no vertical
    crop, giving 1024x256 -- already the model's input size, so the resize that
    follows is a no-op rather than a rescale.

    Getting this wrong is silent: a 1536x200 stitch resized to 1024x256 looks
    like a plausible panorama and scores like a different camera.
    """

    height, width = front.shape[:2]
    nurec_frame = width <= 1024
    vcrop = 0 if nurec_frame else 28
    if vcrop:
        front = front[vcrop : height - vcrop]
    side = (
        max(0, (3 * width - 4 * (height - 2 * vcrop)) // 4)
        if nurec_frame
        else 416
    )
    hi = height - vcrop
    left = left[vcrop:hi, side : width - side]
    right = right[vcrop:hi, side : width - side]
    return np.concatenate((left, front, right), axis=1)


def _crop_vertical(
    image: np.ndarray,
) -> np.ndarray:
    if (
        image.ndim != 3
        or image.shape[2] != 3
    ):
        raise ValueError(
            "expected an RGB HWC image, "
            f"got {image.shape}"
        )

    return (
        image[28:-28]
        if image.shape[0] > 56
        else image
    )


def _crop_side_camera(
    image: np.ndarray,
) -> np.ndarray:
    image = _crop_vertical(
        image
    )

    if image.shape[1] > 832:
        return image[:, 416:-416]

    return image
