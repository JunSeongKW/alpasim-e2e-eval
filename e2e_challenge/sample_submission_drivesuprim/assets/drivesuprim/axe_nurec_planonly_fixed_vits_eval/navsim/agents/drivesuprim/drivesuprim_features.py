import cv2
from enum import IntEnum
import json, os
import numpy as np
import numpy.typing as npt
from PIL import Image
from shapely import affinity
from shapely.geometry import Polygon, LineString
from typing import Any, Dict, List, Tuple, Union

import torch
from torchvision import transforms

from navsim.agents.drivesuprim.drivesuprim_config import DriveSuprimConfig
from navsim.agents.drivesuprim.data.transforms import GaussianBlur
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.actor_state.oriented_box import OrientedBox
from nuplan.common.actor_state.state_representation import StateSE2, TimePoint, StateVector2D
from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.common.geometry.convert import absolute_to_relative_poses
from nuplan.common.maps.abstract_map import AbstractMap, SemanticMapLayer, MapObject
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataclasses import AgentInput, Scene, Annotations
from navsim.common.driving_command import driving_command_from_route
from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.agents.transfuser.transfuser_features import BoundingBox2DIndex
from navsim.evaluate.pdm_score import transform_trajectory, get_trajectory_as_array
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types

from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from navsim.planning.metric_caching.metric_cache_processor_aug_train import ego_state_augmentation

UNKNOWN_INDEX = 3
"""Slot of the 'unknown' command in the one-hot; a derived unknown means the
frame had no usable route, so the logged command stands."""


class CalibrationMismatch(ValueError):
    """The logs and the image files come from different rectification stages.

    Its own type so DatasetSSL's retry wrapper re-raises it instead of hunting
    for a good sample: every sample carries the same broken calibration, so
    retrying only buries the message.
    """


class DriveSuprimFeatureBuilder(AbstractFeatureBuilder):
    def __init__(self, config: DriveSuprimConfig):
        self._config = config
        self.training = config.training
        self.is_bev_backbone = config.backbone_type == 'bevformer_m'

        # The offline rotation-augmentation file is only needed when the
        # rotation ensemble is active (not `only_ori_input`, e.g. the BEV path).
        if self.training and not config.only_ori_input:
            with open(config.ego_perturb.offline_aug_file, 'r') as f:
                aug_data = json.load(f)
            assert aug_data['param']['rot'] == config.ego_perturb.offline_aug_angle_boundary
            self.aug_info = aug_data['tokens']

        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        self.teacher_ori_augmentation = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        self.student_ori_augmentation = transforms.Compose(
            [
                transforms.ToTensor(),
                color_jittering,
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=0.5, p=0.2),
            ]
        )
        self.student_rotated_augmentation = transforms.Compose(
            [
                transforms.ToTensor(),
                color_jittering,
                GaussianBlur(p=0.5),
            ]
        )

        # BEVFormer-M multi-view transforms: clean for the teacher, photometric
        # augmentation for the student (self-distillation still holds via
        # photometric consistency even with the rotation ensemble disabled).
        self.bev_teacher_augmentation = transforms.Compose([transforms.ToTensor()])
        self.bev_student_augmentation = transforms.Compose(
            [
                transforms.ToTensor(),
                color_jittering,
                GaussianBlur(p=0.1),
            ]
        )
        # camera selection per bev_num_cameras (matches the stitched-view layout)
        self._bev_cam_names = {
            1: ['cam_f0'],
            3: ['cam_l0', 'cam_f0', 'cam_r0'],
            5: ['cam_l1', 'cam_l0', 'cam_f0', 'cam_r0', 'cam_r1'],
        }

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "drivesuprim_feature"

    def compute_features(self, agent_input: AgentInput, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        features = {}
        
        initial_token = scene.scene_metadata.initial_token
        if not self._config.only_ori_input and self._config.training:
            n_rotated = self._config.ego_perturb.n_student_rotation_ensemble
        else:
            n_rotated = 0
        if self.is_bev_backbone:
            features.update(self._get_bev_camera_feature(agent_input, initial_token, rotation_num=n_rotated))
        else:
            features.update(self._get_camera_feature(agent_input, initial_token, rotation_num=n_rotated))

        features["status_feature"] = self._get_status_feature(agent_input)  # [seq_len, 8]

        if self._config.use_route:
            features.update(self._get_route_feature(agent_input, initial_token, rotation_num=n_rotated))

        self._ablate(features)
        return features

    # Which inputs to blank, read once from NUREC_ABLATE_INPUTS. A comma list of
    # `images` and `route`; anything else is a typo worth failing on rather than
    # silently scoring the unablated model and reporting it as an ablation.
    _ABLATIONS = {"images": ("bev_imgs", "bev_imgs_teacher", "camera_feature"),
                  "route": ("route_feature", "route_mask", "route_feature_rotated")}

    def _ablate(self, features: Dict[str, torch.Tensor]) -> None:
        """Zero whole inputs, to measure what the model actually leans on.

        Evaluation only: the point is to score the SAME weights with a signal
        removed, so the drop is attributable to that signal. Blanking beats
        turning the branch off in the config -- ``use_route=false`` would skip
        the route encoder entirely and change the network, while a zeroed
        route_mask marks every slot as padding and leaves the architecture,
        the checkpoint and every other input exactly as they were.

        Off unless NUREC_ABLATE_INPUTS is set, so training never sees it.
        """
        raw = os.environ.get("NUREC_ABLATE_INPUTS", "").strip()
        if not raw:
            return
        wanted = [w.strip() for w in raw.split(",") if w.strip()]
        unknown = [w for w in wanted if w not in self._ABLATIONS]
        if unknown:
            raise ValueError(
                f"NUREC_ABLATE_INPUTS={raw!r}: unknown {unknown}; "
                f"known are {sorted(self._ABLATIONS)}")
        for what in wanted:
            for key in self._ABLATIONS[what]:
                value = features.get(key)
                if torch.is_tensor(value):
                    features[key] = torch.zeros_like(value)
                elif isinstance(value, list):
                    features[key] = [torch.zeros_like(v) if torch.is_tensor(v) else v
                                     for v in value]
        if not getattr(DriveSuprimFeatureBuilder, "_ablation_reported", False):
            DriveSuprimFeatureBuilder._ablation_reported = True
            print(f"[drivesuprim] ABLATION ACTIVE -- blanking {wanted}", flush=True)

    def _get_status_feature(self, agent_input: AgentInput) -> List[torch.Tensor]:
        """Per-frame [driving_command(4), velocity(2), acceleration(2)]."""
        ego_status_list = []
        for i in range(self._config.seq_len):
            idx = -(i + 1)
            ego_status = agent_input.ego_statuses[idx]
            command = ego_status.driving_command
            if not self._config.use_route and self._config.derive_command_from_route:
                # A checkpoint that steers off the command rather than the route
                # needs the command to carry the intent NAVSIM put in it. The one
                # NuRec logs is straight on 85% of frames against NAVSIM's 62%,
                # so it under-signals every manoeuvre; the route says the same
                # thing in NAVSIM's proportions. Falls back to the logged value
                # when a frame has no route.
                derived = driving_command_from_route(ego_status.route_waypoints)
                if not derived[UNKNOWN_INDEX]:
                    command = derived
            driving_command = torch.tensor(command, dtype=torch.float32)
            if self._config.use_route:
                # The route is the intent signal now. Keep the four slots zeroed
                # rather than removing them, so a checkpoint trained with the
                # command still loads _status_encoding.
                driving_command = torch.zeros_like(driving_command)
            ego_status_list.append(
                torch.concatenate(
                    [
                        driving_command,
                        torch.tensor(agent_input.ego_statuses[idx].ego_velocity, dtype=torch.float32),
                        torch.tensor(agent_input.ego_statuses[idx].ego_acceleration, dtype=torch.float32),
                    ],
                )
            )
        return ego_status_list

    def _get_route_feature(
        self, agent_input: AgentInput, initial_token: str, rotation_num: int = 0
    ) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """Route waypoints of the present frame, masked and normalised.

        The rotation ensemble turns the ego frame by ``-rot`` (the same
        ``_reverse_rotation`` the rotated trajectories use), so the route has to
        turn with it. Leaving it un-rotated would hand the augmented sample a
        BEV and a label pointing one way and an intent pointing another.
        """
        waypoints = np.asarray(agent_input.ego_statuses[-1].route_waypoints, dtype=np.float32)[:, :2]
        mask = np.isfinite(waypoints).all(axis=1)
        # Zero the padding before anything multiplies it: a single NaN reaching
        # the encoder would spread across the batch.
        waypoints = np.where(mask[:, None], waypoints, 0.0).astype(np.float32)
        scale = self._config.route_norm_m

        out: Dict[str, Union[torch.Tensor, List[torch.Tensor]]] = {
            "route_feature": torch.from_numpy(waypoints / scale),
            "route_mask": torch.from_numpy(mask.astype(np.float32)),
        }
        rotated: List[torch.Tensor] = []
        for i in range(rotation_num):
            reverse_rotation = -float(self.aug_info[initial_token][i]["rot"]) / 180.0 * np.pi
            turned = np_vector2_aug(waypoints.T, reverse_rotation).T.astype(np.float32)
            rotated.append(torch.from_numpy(turned / scale))
        out["route_feature_rotated"] = rotated
        return out

    def _get_camera_feature(self, agent_input: AgentInput, initial_token: str, rotation_num=3) -> Dict[str, Union[torch.Tensor, List[torch.Tensor]]]:
        """
        Extract teacher and student camera input from AgentInput
        :param 
          agent_input: input dataclass
          initial_token: scene token, used to get the specific rotation angle of augmentation
          rotation_num: number of rotation angle
        """
        res = dict()

        seq_len = self._config.seq_len
        cameras = agent_input.cameras[-seq_len:]  # List[Cameras]
        assert(len(cameras) == seq_len)
        
        # Teacher input
        res['ori_teacher'] = []

        # Student input
        res['ori'] = []
        res['rotated'] = [[] for _ in range(rotation_num)]

        for camera in cameras:
            n_camera = self._config.n_camera
            image = camera.cam_f0.image if n_camera == 1 else camera.cam_l0.image
            if image is not None and image.size > 0 and np.any(image):
                # Crop to ensure 4:1 aspect ratio
                f0 = camera.cam_f0.image[28:-28]
                if n_camera == 1:
                    ori_image = f0
                elif n_camera == 3:
                    l0 = camera.cam_l0.image[28:-28, 416:-416]
                    r0 = camera.cam_r0.image[28:-28, 416:-416]
                    ori_image = np.concatenate([l0, f0, r0], axis=1)
                elif n_camera == 5:
                    l1 = camera.cam_l1.image[28:-28]
                    l0 = camera.cam_l0.image[28:-28, 416:-416]
                    r0 = camera.cam_r0.image[28:-28, 416:-416]
                    r1 = camera.cam_r1.image[28:-28]
                    ori_image = np.concatenate([l1, l0, f0, r0, r1], axis=1)
                else:
                    raise NotImplementedError(f"n_camera={n_camera} is not supported")
                
                _ori_image = cv2.resize(ori_image, (self._config.camera_width, self._config.camera_height))
                res['ori_teacher'].append(self.teacher_ori_augmentation(_ori_image))
                res['ori'].append(self.student_ori_augmentation(_ori_image))
                
                # Extra side/rear cameras were historically read even when no
                # rotation samples were requested.  Keep that wider panorama
                # only for the optional rotation ensemble; ordinary 3-camera
                # NuRec training now touches L0/F0/R0 exclusively.
                if rotation_num > 0:
                    if n_camera < 5:
                        raise ValueError(
                            "pixel-space rotation augmentation is not supported with "
                            f"n_camera={n_camera}; use only_ori_input=true for the "
                            "NuRec three-camera setup"
                        )
                    l1 = camera.cam_l1.image[28:-28]
                    r1 = camera.cam_r1.image[28:-28]
                    img_3cam_w = l0.shape[1] + f0.shape[1] + r0.shape[1]
                    l1_w, r1_w = l1.shape[1], r1.shape[1]
                    l2 = camera.cam_l2.image[28:-28, :-1100]
                    r2 = camera.cam_r2.image[28:-28, 1100:]
                    b0_left = camera.cam_b0.image[28:-28, :1080]
                    b0_right = camera.cam_b0.image[28:-28, -1080:]
                    stitched_image = np.concatenate(
                        [b0_left, l2, l1, l0, f0, r0, r1, r2, b0_right], axis=1
                    )

                    whole_w = stitched_image.shape[1]
                    half_view_w = img_3cam_w + l1_w // 2 + r1_w // 2
                    img_w = img_3cam_w + l1_w + r1_w

                    if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
                        debug_dir = 'debug_viz'
                        os.makedirs(debug_dir, exist_ok=True)
                        Image.fromarray(stitched_image).save(f'{debug_dir}/stitched_image.jpg')

                    for i in range(rotation_num):
                        angle = self.aug_info[initial_token][i]['rot']
                        offset_w = int(half_view_w / 180 * angle)
                        start = int(whole_w / 2 - offset_w - img_w / 2)
                        rotated_image = stitched_image[:, start:start + img_w]
                        resized_image = cv2.resize(
                            rotated_image, (self._config.camera_width, self._config.camera_height)
                        )
                        tensor_image = self.student_rotated_augmentation(resized_image)
                        if os.getenv('ROBUST_HYDRA_DEBUG') == 'true':
                            transforms.ToPILImage()(tensor_image).save(
                                f'{debug_dir}/output_tensor_image_{i}.jpg'
                            )
                        res['rotated'][i].append(tensor_image)
                
        return res


    @staticmethod
    def _undistort_image(image: np.ndarray, intrinsics: np.ndarray, distortion: np.ndarray) -> np.ndarray:
        """Lens-distortion correction matching SafeDrive's camera preprocessing."""
        h, w = image.shape[:2]
        map1, map2 = cv2.initUndistortRectifyMap(
            intrinsics, distortion, None, intrinsics, (w, h), cv2.CV_32FC1)
        return cv2.remap(image, map1, map2, interpolation=cv2.INTER_LINEAR)

    def _assert_intrinsics_match_image(self, intrinsics: npt.NDArray[np.float64], cam) -> None:
        """Refuse a cam_intrinsic written for a different image than the one loaded.

        A prepared root can hold logs whose calibration predates rectification --
        nuPlan's pinhole, cx 960 cy 560 for 1920x1080 -- while the files beside
        it are the rectified 512x256 views. Nothing downstream notices: every
        BEV reference point simply projects outside the image and
        SpatialCrossAttention returns its residual, so the encoder trains and
        scores happily on zero image content. That combination cost a 30-epoch
        run, which reached bev_mask 0.000% for all 3,136 queries on every frame
        and still produced a plausible 0.9259.

        The cheapest invariant that catches it: a pinhole's principal point is
        inside its own image. Checked after the resize scaling, so it is the
        image the model actually receives.
        """
        cx, cy = float(intrinsics[0, 2]), float(intrinsics[1, 2])
        w, h = self._config.bev_img_width, self._config.bev_img_height
        if 0.0 < cx < w and 0.0 < cy < h:
            return
        path = getattr(cam, "camera_path", "?")
        raise CalibrationMismatch(
            f"cam_intrinsic does not belong to this image: principal point "
            f"({cx:.1f}, {cy:.1f}) falls outside the {w}x{h} view fed to the "
            f"backbone (camera {path}). The logs and the image files come from "
            f"different stages of the rectification pipeline -- check that "
            f"NUREC_PREPARED_ROOT and NUREC_SENSOR_ROOT are a matched pair."
        )

    def _build_lidar2img(self, cam, orig_h: int, orig_w: int, crop_top: int = 0) -> np.ndarray:
        """Build the 4x4 ego(lidar)->image projection for a resized view."""
        intrinsics = cam.intrinsics.astype(np.float64).copy()  # 3x3 for original res
        if crop_top:
            intrinsics[1, 2] -= crop_top
        # scale intrinsics to the resized image
        sx = self._config.bev_img_width / float(orig_w)
        sy = self._config.bev_img_height / float(orig_h)
        intrinsics[0, :] *= sx
        intrinsics[1, :] *= sy
        self._assert_intrinsics_match_image(intrinsics, cam)

        # camera->ego(lidar) extrinsics
        cam2lidar = np.eye(4, dtype=np.float64)
        cam2lidar[:3, :3] = cam.sensor2lidar_rotation
        cam2lidar[:3, 3] = cam.sensor2lidar_translation
        lidar2cam = np.linalg.inv(cam2lidar)

        viewpad = np.eye(4, dtype=np.float64)
        viewpad[:3, :3] = intrinsics
        lidar2img = viewpad @ lidar2cam
        return lidar2img.astype(np.float32)

    def _get_bev_camera_feature(self, agent_input: AgentInput, initial_token: str,
                                rotation_num: int = 0) -> Dict[str, torch.Tensor]:
        """Assemble multi-view images + calibration + ego poses for BEVFormer-M.

        Returns per-frame stacks (T = bev_seq_len, current frame last):
          bev_imgs         : [T, num_cam, 3, H, W]  (student / photometric aug)
          bev_imgs_teacher : [T, num_cam, 3, H, W]  (clean)
          lidar2img        : [T, num_cam, 4, 4]
          bev_ego_pose     : [T, 3]  each frame relative to the current frame

        Rotation augmentation (Method B): rather than rotating the image pixels
        (which would break the fixed calibration), we rotate the *projection*.
        A yaw of +theta redefines the BEV/ego frame; a reference point p_N in the
        rotated frame maps to the original lidar frame via R_z(theta), so the
        effective projection is ``lidar2img @ R_z(theta)``. The images (and thus
        the true pixels/calibration) stay consistent and only the BEV grid
        orientation turns. The angles come from the SAME offline aug file as the
        rotated trajectories / aug vocab scores, so labels line up with no
        recomputation. When ``rotation_num`` > 0 we also return:
          bev_lidar2img_rotated : list length rotation_num of [T, num_cam, 4, 4]
        """
        seq_len = self._config.bev_seq_len
        cam_names = self._bev_cam_names[self._config.bev_num_cameras]

        cameras = agent_input.cameras[-seq_len:]
        ego_statuses = agent_input.ego_statuses[-seq_len:]
        assert len(cameras) == seq_len and len(ego_statuses) == seq_len, (
            f"need {seq_len} history frames for BEVFormer-M, got "
            f"{len(cameras)} cameras / {len(ego_statuses)} ego statuses")

        imgs_student, imgs_teacher, l2i_all, poses = [], [], [], []
        for camera, ego in zip(cameras, ego_statuses):
            frame_student, frame_teacher, frame_l2i = [], [], []
            for name in cam_names:
                cam = getattr(camera, name)
                image = cam.image
                if self._config.bev_undistort:
                    image = self._undistort_image(image, cam.intrinsics, cam.distortion)
                crop_top = int(self._config.bev_crop_top_bottom)
                if crop_top:
                    image = image[crop_top:-crop_top]
                orig_h, orig_w = image.shape[:2]
                resized = cv2.resize(
                    image, (self._config.bev_img_width, self._config.bev_img_height))
                frame_teacher.append(self.bev_teacher_augmentation(resized))
                frame_student.append(self.bev_student_augmentation(resized))
                frame_l2i.append(torch.from_numpy(
                    self._build_lidar2img(cam, orig_h, orig_w, crop_top=crop_top)))
            imgs_student.append(torch.stack(frame_student))   # [num_cam, 3, H, W]
            imgs_teacher.append(torch.stack(frame_teacher))
            l2i_all.append(torch.stack(frame_l2i))            # [num_cam, 4, 4]
            poses.append(torch.tensor(ego.ego_pose[:3], dtype=torch.float32))

        lidar2img = torch.stack(l2i_all)                      # [T, num_cam, 4, 4]
        out = {
            'bev_imgs': torch.stack(imgs_student),            # [T, num_cam, 3, H, W]
            'bev_imgs_teacher': torch.stack(imgs_teacher),
            'lidar2img': lidar2img,
            'bev_ego_pose': torch.stack(poses),               # [T, 3]
        }

        # Rotation augmentation: rotate the projection (not the pixels). Kept as a
        # python list (length rotation_num) so the default collate preserves the
        # rotation axis; each element collates to [bs, T, num_cam, 4, 4].
        rotated_l2i: List[torch.Tensor] = []
        rotated_yaw: List[torch.Tensor] = []
        for i in range(rotation_num):
            theta = float(self.aug_info[initial_token][i]['rot']) * np.pi / 180.0
            rotated_l2i.append(lidar2img @ _yaw_rot4x4(theta))   # [T, num_cam, 4, 4]
            rotated_yaw.append(torch.tensor(theta, dtype=torch.float32))
        out['bev_lidar2img_rotated'] = rotated_l2i
        # per-rotation yaw (radians); keeps the temporal warp consistent with the
        # rotated projection. Kept as a list so it collates to a per-rotation
        # [bs] tensor, indexed the same way as bev_lidar2img_rotated.
        out['bev_aug_yaw'] = rotated_yaw
        return out


class DriveSuprimTargetBuilder(AbstractTargetBuilder):
    def __init__(self, config: DriveSuprimConfig):
        
        self._config = config
        self.v_params = get_pacifica_parameters()
        self.training = config.training
        # Auxiliary dense supervision (BEV seg + agent detection) is only built
        # for the BEV backbone during training; the GT is rasterized onto the
        # BEVFormer BEV grid (point_cloud_range) so it aligns with the head output.
        self.build_aux_targets = (
            getattr(config, 'use_aux_heads', False)
            and config.backbone_type == 'bevformer_m'
            and self.training
        )

        if self.training and not config.only_ori_input:
            with open(config.ego_perturb.offline_aug_file, 'r') as f:
                aug_data = json.load(f)
                assert aug_data['param']['rot'] == config.ego_perturb.offline_aug_angle_boundary
                self.aug_info = aug_data['tokens']

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "transfuser_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""

        initial_token = scene.scene_metadata.initial_token
        future_traj = scene.get_future_trajectory(
            num_trajectory_frames=int(4 / 0.5)
        )  # [8, 3] (local pose, do not include present pose)

        if not self._config.only_ori_input and self._config.training:
            n_rotated = self._config.ego_perturb.n_student_rotation_ensemble
            _rotations = [self.aug_info[initial_token][i]['rot'] for i in range(n_rotated)]
            _reverse_rotations = [-_rot / 180.0 * np.pi for _rot in _rotations]  # Convert degree to rad
        else:
            _rotations = [0]
            _reverse_rotations = [0]

        # Original trajectory
        ori_trajectory = torch.tensor(future_traj.poses)  # [num_poses, 3]

        # Apply rotations to get multiple trajectories
        rotated_trajectories = []
        for _reverse_rotation in _reverse_rotations:
            if self.training and abs(_reverse_rotation) > 1e-5:
                # Rotate each trajectory point around origin (ego vehicle)
                rotated_poses = []
                for pose in future_traj.poses:
                    # Rotate x,y coordinates
                    rotated_xy = np_vector2_aug(pose[:2], _reverse_rotation)
                    # Adjust heading and normalize to [-pi, pi]
                    rotated_heading = (pose[2] + _reverse_rotation + np.pi) % (2*np.pi) - np.pi
                    rotated_poses.append(np.array([rotated_xy[0], rotated_xy[1], rotated_heading]))
                rotated_poses = np.array(rotated_poses)
                rotated_trajectories.append(torch.tensor(rotated_poses))  # [num_poses, 3]
            else:
                rotated_trajectories.append(ori_trajectory)  # Use original trajectory when no rotation
                    
        ret = {
            "ori_trajectory": ori_trajectory,
            "rotated_trajectories": rotated_trajectories,
        }

        if self.build_aux_targets:
            # Un-rotated ("ori") ego frame at the current (last history) frame,
            # matching the un-rotated BEV the student produces for aux supervision.
            frame_idx = scene.scene_metadata.num_history_frames - 1
            annotations = scene.frames[frame_idx].annotations
            ego_pose = StateSE2(*scene.frames[frame_idx].ego_status.ego_pose)
            agent_states, agent_labels, agent_classes, selected_tracks = \
                self._compute_agent_targets(annotations)
            ret["agent_states"] = agent_states
            ret["agent_labels"] = agent_labels      # objectness (kept for the transfuser head)
            ret["agent_classes"] = agent_classes    # class index per slot
            ret["bev_seg_map"] = self._compute_bev_seg_map(
                annotations, scene.map_api, ego_pose
            )
            if getattr(self._config, "aux_bev_dual_seg", False):
                # SafeDrive supervises BEV semantic and drivable-area segmentation
                # as two separate terms, so emit the binary drivable mask too.
                _, drivable_layers = self._config.bev_semantic_classes[1]
                ret["bev_drivable_map"] = torch.Tensor(
                    self._compute_map_polygon_mask(
                        scene.map_api, ego_pose, drivable_layers).astype(np.int64)
                )
            if getattr(self._config, "aux_predict_agents", False):
                agent_future, agent_future_mask = self._compute_agent_future(
                    scene, frame_idx, selected_tracks
                )
                ret["agent_future_trajectory"] = agent_future
                ret["agent_future_mask"] = agent_future_mask

        return ret

    # ------------------------------------------------------------------ #
    # Auxiliary GT (BEV segmentation + agent detection), BEVFormer grid.  #
    # Ported from TransfuserTargetBuilder but rasterized directly onto    #
    # the BEV grid defined by point_cloud_range / (bev_h, bev_w) so it    #
    # aligns pixel-for-pixel with the BEVFormer encoder output.           #
    # ------------------------------------------------------------------ #
    def _compute_agent_targets(self, annotations: Annotations):
        """2D agent boxes (x, y, heading, length, width) in the ego frame,
        filtered to the BEV extent and padded to num_bounding_boxes. Also returns
        the per-slot track_token (aligned to the padded 30 rows, '' for padding)
        so agent futures can be linked across frames in the SAME ordering."""
        max_agents = self._config.num_bounding_boxes
        x_min, y_min, _, x_max, y_max, _ = self._config.point_cloud_range

        class_names = tuple(self._config.aux_agent_classes)
        class_index = {n: i for i, n in enumerate(class_names)}
        box_3d = getattr(self._config, "aux_agent_box_3d", False)
        velocities = getattr(annotations, "velocity_3d", None)
        if velocities is None:
            velocities = np.zeros((len(annotations.boxes), 3), dtype=np.float32)

        agent_states_list: List[npt.NDArray[np.float32]] = []
        class_list: List[int] = []
        track_list: List[str] = []
        for box, name, track, vel in zip(
            annotations.boxes, annotations.names, annotations.track_tokens, velocities
        ):
            box_x, box_y, box_heading, box_length, box_width = (
                box[BoundingBoxIndex.X],
                box[BoundingBoxIndex.Y],
                box[BoundingBoxIndex.HEADING],
                box[BoundingBoxIndex.LENGTH],
                box[BoundingBoxIndex.WIDTH],
            )
            if name in class_index and (x_min <= box_x <= x_max) and (y_min <= box_y <= y_max):
                if box_3d:
                    # BEVFormer normalize_bbox layout:
                    #   [cx, cy, log(l), log(w), cz, log(h), sin(rot), cos(rot), vx, vy]
                    box_z = box[BoundingBoxIndex.Z]
                    box_height = box[BoundingBoxIndex.HEIGHT]
                    eps = 1e-4
                    agent_states_list.append(np.array([
                        box_x, box_y,
                        # length before width, as in BEVFormer's normalize_bbox:
                        # it reads bboxes[..., 3] (= dx, the x-extent = length)
                        # into slot 2 and bboxes[..., 4] (= dy = width) into
                        # slot 3.
                        np.log(max(float(box_length), eps)),
                        np.log(max(float(box_width), eps)),
                        box_z,
                        np.log(max(float(box_height), eps)),
                        np.sin(box_heading), np.cos(box_heading),
                        vel[0], vel[1],
                    ], dtype=np.float32))
                else:
                    agent_states_list.append(
                        np.array([box_x, box_y, box_heading, box_length, box_width], dtype=np.float32)
                    )
                class_list.append(class_index[name])
                track_list.append(track)

        agents_states_arr = np.array(agent_states_list)
        box_dim = 10 if box_3d else BoundingBox2DIndex.size()
        agent_states = np.zeros((max_agents, box_dim), dtype=np.float32)
        agent_labels = np.zeros(max_agents, dtype=bool)
        # Class index per slot; only meaningful where agent_labels is True.
        agent_classes = np.zeros(max_agents, dtype=np.int64)
        selected_tracks: List[str] = ["" for _ in range(max_agents)]
        if len(agents_states_arr) > 0:
            # cx, cy are slots 0/1 in both layouts.
            distances = np.linalg.norm(agents_states_arr[..., 0:2], axis=-1)
            argsort = np.argsort(distances)[:max_agents]
            agents_states_arr = agents_states_arr[argsort]
            agent_states[: len(agents_states_arr)] = agents_states_arr
            agent_labels[: len(agents_states_arr)] = True
            for slot, src in enumerate(argsort):
                selected_tracks[slot] = track_list[src]
                agent_classes[slot] = class_list[src]

        return (torch.tensor(agent_states), torch.tensor(agent_labels),
                torch.tensor(agent_classes), selected_tracks)

    def _compute_agent_future(self, scene, current_frame_idx: int, selected_tracks: List[str]):
        """Future trajectory (x, y in the CURRENT ego frame) for each selected
        agent over aux_pred_num_poses steps (one per future frame, 0.5s apart).

        Agents are linked across future frames by track_token; each future box
        (in that frame's ego frame) is mapped ego_f -> global -> ego_current.
        Returns (future [num_bb, T, 2], mask [num_bb, T]) with mask=0 where the
        agent is absent at that step (occluded/exited or frame unavailable)."""
        max_agents = self._config.num_bounding_boxes
        T = self._config.aux_pred_num_poses
        future = np.zeros((max_agents, T, 2), dtype=np.float32)
        mask = np.zeros((max_agents, T), dtype=np.float32)

        ex, ey, eh = [float(v) for v in scene.frames[current_frame_idx].ego_status.ego_pose[:3]]
        cos_c, sin_c = np.cos(eh), np.sin(eh)
        slot_of_track = {t: i for i, t in enumerate(selected_tracks) if t}
        if not slot_of_track:
            return torch.from_numpy(future), torch.from_numpy(mask)

        n_frames = len(scene.frames)
        for step in range(T):
            f = current_frame_idx + 1 + step
            if f >= n_frames:
                break
            frame = scene.frames[f]
            efx, efy, efh = [float(v) for v in frame.ego_status.ego_pose[:3]]
            cos_f, sin_f = np.cos(efh), np.sin(efh)
            ann = frame.annotations
            for box, track in zip(ann.boxes, ann.track_tokens):
                slot = slot_of_track.get(track)
                if slot is None:
                    continue
                ax, ay = float(box[BoundingBoxIndex.X]), float(box[BoundingBoxIndex.Y])
                # frame-f ego-local -> global
                gx = efx + cos_f * ax - sin_f * ay
                gy = efy + sin_f * ax + cos_f * ay
                # global -> current ego frame
                dx, dy = gx - ex, gy - ey
                px = cos_c * dx + sin_c * dy
                py = -sin_c * dx + cos_c * dy
                future[slot, step] = (px, py)
                mask[slot, step] = 1.0
        return torch.from_numpy(future), torch.from_numpy(mask)

    def _ego_is_on_road(self, mask: npt.NDArray[np.bool_]) -> bool:
        """Does the drivable mask cover the ground the car is standing on?

        A band ahead of the rear axle rather than the single cell under it: the
        rear axle sits on the window's x edge, where one cell either way is
        boundary noise, and the body extends forward of it anyway.
        """
        if not getattr(self._config, "aux_bev_seg_require_ego_on_road", False):
            return True
        band = float(getattr(self._config, "aux_bev_seg_ego_band_m", 6.0))
        half = float(getattr(self._config, "aux_bev_seg_ego_half_width_m", 2.0))
        corners = np.array([[0.0, -half], [band, half]], dtype=np.float64)
        (c0, r0), (c1, r1) = self._coords_to_pixel(corners.reshape((-1, 1, 2))).reshape(2, 2)
        c0, c1 = sorted((int(c0), int(c1)))
        r0, r1 = sorted((int(r0), int(r1)))
        h, w = mask.shape
        window = mask[max(r0, 0):min(r1 + 1, h), max(c0, 0):min(c1 + 1, w)]
        return bool(window.size and window.any())

    def _compute_bev_seg_map(
        self, annotations: Annotations, map_api: AbstractMap, ego_pose: StateSE2
    ) -> torch.Tensor:
        """Rasterize map layers + boxes into a [bev_h, bev_w] class-index map,
        aligned to the BEVFormer BEV grid (row=h=metric y, col=w=metric x).

        drivable_only=True -> binary {0,1} mask of the drivable area (road):
        the polygons of aux_bev_drivable_layers."""
        bev_h, bev_w = self._config.bev_h, self._config.bev_w
        if getattr(self._config, 'aux_bev_drivable_only', False):
            mask = self._compute_map_polygon_mask(
                map_api, ego_pose, self._drivable_layers())
            if not self._ego_is_on_road(mask):
                # -1 everywhere: _bev_seg_loss drops labels outside
                # [0, num_classes) and averages over the survivors, so this
                # frame contributes nothing to the segmentation term while its
                # trajectory labels are untouched.
                return torch.full((bev_h, bev_w), -1, dtype=torch.float32)
            return torch.Tensor(mask.astype(np.int64))  # 1 = drivable, 0 = not
        bev_seg_map = np.zeros((bev_h, bev_w), dtype=np.int64)
        semantic_classes = (
            self._config.active_bev_semantic_classes()
            if hasattr(self._config, "active_bev_semantic_classes")
            else self._config.bev_semantic_classes
        )
        for label, (entity_type, layers) in semantic_classes.items():
            if entity_type == "polygon":
                entity_mask = self._compute_map_polygon_mask(map_api, ego_pose, layers)
            elif entity_type == "linestring":
                entity_mask = self._compute_map_linestring_mask(map_api, ego_pose, layers)
            else:
                entity_mask = self._compute_box_mask(annotations, layers)
            bev_seg_map[entity_mask] = label
        return torch.Tensor(bev_seg_map)

    def _coords_to_pixel(self, coords: np.ndarray) -> np.ndarray:
        """Ego-local (x forward, y left) -> BEV pixel (col=x, row=y) indices.

        The BEVFormer encoder maps its flattened [H, W] queries so that the W
        axis spans metric x in [x_min, x_max] and the H axis spans metric y in
        [y_min, y_max], both increasing min->max. Mapping GT the same way keeps
        the seg target aligned with the predicted BEV pixel-for-pixel.
        """
        x_min, y_min, _, x_max, y_max, _ = self._config.point_cloud_range
        bev_h, bev_w = self._config.bev_h, self._config.bev_w
        xs = coords[..., 0]
        ys = coords[..., 1]
        col = (xs - x_min) / (x_max - x_min) * bev_w  # metric x -> W axis
        row = (ys - y_min) / (y_max - y_min) * bev_h  # metric y -> H axis
        # cv2 expects (col, row) point ordering; array is indexed [row, col].
        return np.stack([col, row], axis=-1).astype(np.int32)

    def _drivable_layers(self) -> List[SemanticMapLayer]:
        """Polygon layers the binary drivable target is built from.

        Hydra hands YAML lists over as strings, and a string silently matches no
        layer: get_proximal_map_objects returns nothing for it and the target
        comes out entirely zero, which trains perfectly happily on a blank map.
        Names are resolved here, and an unknown one raises rather than vanishing.
        """
        layers = getattr(self._config, "aux_bev_drivable_layers", None)
        if not layers:
            # inherited default: bev_semantic_classes[1] == ("polygon", [...])
            return list(self._config.bev_semantic_classes[1][1])
        resolved: List[SemanticMapLayer] = []
        for layer in layers:
            if isinstance(layer, SemanticMapLayer):
                resolved.append(layer)
                continue
            try:
                resolved.append(SemanticMapLayer[str(layer)])
            except KeyError as exc:
                raise ValueError(
                    f"aux_bev_drivable_layers: {layer!r} is not a SemanticMapLayer. "
                    f"Expected one of {[m.name for m in SemanticMapLayer]}"
                ) from exc
        return resolved

    def _compute_map_polygon_mask(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        radius = max(abs(v) for v in self._config.point_cloud_range[:2] + self._config.point_cloud_range[3:5])
        map_object_dict = map_api.get_proximal_map_objects(point=ego_pose.point, radius=radius, layers=layers)
        mask = np.zeros((self._config.bev_h, self._config.bev_w), dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                polygon: Polygon = self._geometry_local_coords(map_object.polygon, ego_pose)
                exterior = np.array(polygon.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(mask, [exterior], color=255)
        return mask > 0

    def _compute_map_linestring_mask(
        self, map_api: AbstractMap, ego_pose: StateSE2, layers: List[SemanticMapLayer]
    ) -> npt.NDArray[np.bool_]:
        radius = max(abs(v) for v in self._config.point_cloud_range[:2] + self._config.point_cloud_range[3:5])
        map_object_dict = map_api.get_proximal_map_objects(point=ego_pose.point, radius=radius, layers=layers)
        mask = np.zeros((self._config.bev_h, self._config.bev_w), dtype=np.uint8)
        for layer in layers:
            for map_object in map_object_dict[layer]:
                linestring: LineString = self._geometry_local_coords(
                    map_object.baseline_path.linestring, ego_pose
                )
                points = np.array(linestring.coords).reshape((-1, 1, 2))
                points = self._coords_to_pixel(points)
                cv2.polylines(mask, [points], isClosed=False, color=255, thickness=2)
        return mask > 0

    def _compute_box_mask(self, annotations: Annotations, layers) -> npt.NDArray[np.bool_]:
        mask = np.zeros((self._config.bev_h, self._config.bev_w), dtype=np.uint8)
        for name_value, box_value in zip(annotations.names, annotations.boxes):
            agent_type = tracked_object_types[name_value]
            if agent_type in layers:
                x, y, heading = box_value[0], box_value[1], box_value[-1]
                box_length, box_width, box_height = box_value[3], box_value[4], box_value[5]
                agent_box = OrientedBox(StateSE2(x, y, heading), box_length, box_width, box_height)
                exterior = np.array(agent_box.geometry.exterior.coords).reshape((-1, 1, 2))
                exterior = self._coords_to_pixel(exterior)
                cv2.fillPoly(mask, [exterior], color=255)
        return mask > 0

    @staticmethod
    def _geometry_local_coords(geometry: Any, origin: StateSE2) -> Any:
        """Transform a shapely geometry from global into ego-local coordinates."""
        a = np.cos(origin.heading)
        b = np.sin(origin.heading)
        d = -np.sin(origin.heading)
        e = np.cos(origin.heading)
        translated = affinity.affine_transform(geometry, [1, 0, 0, 1, -origin.x, -origin.y])
        rotated = affinity.affine_transform(translated, [a, b, d, e, 0, 0])
        return rotated


def np_vector2_aug(arr, angle):
    # angle: rad, positive means turning left
    _sin, _cos = np.sin(angle), np.cos(angle)
    x_rotated = arr[0] * _cos - arr[1] * _sin
    y_rotated = arr[0] * _sin + arr[1] * _cos
    return np.array([x_rotated, y_rotated])


def _yaw_rot4x4(angle: float) -> torch.Tensor:
    """4x4 homogeneous yaw (about +z) by ``angle`` radians, positive = left/CCW.

    Post-multiplying lidar2img by this maps a point expressed in the yaw-rotated
    ego frame back to the original lidar frame before projection, i.e. it turns
    the BEV grid by +angle to match the rotated trajectory / vocab-score labels
    (whose xy use R_z(-angle) on the original poses -> same +angle frame).
    """
    c, s = float(np.cos(angle)), float(np.sin(angle))
    return torch.tensor(
        [[c, -s, 0.0, 0.0],
         [s,  c, 0.0, 0.0],
         [0.0, 0.0, 1.0, 0.0],
         [0.0, 0.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
