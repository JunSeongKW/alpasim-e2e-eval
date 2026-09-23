import json
import numpy as np
import os, pickle
from collections import OrderedDict
from typing import Any, Dict, List, Union

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.drivesuprim.drivesuprim_config import DriveSuprimConfig
from navsim.agents.drivesuprim.drivesuprim_features import DriveSuprimFeatureBuilder, DriveSuprimTargetBuilder
from navsim.agents.drivesuprim.ssl_meta_arch import SSLMetaArch
from navsim.agents.drivesuprim.drivesuprim_loss_fn import drivesuprim_agent_loss_first_stage, drivesuprim_agent_loss_single_refine_stage, drivesuprim_aux_loss
from navsim.agents.drivesuprim.drivesuprim_model import DriveSuprimModel
from navsim.common.dataclasses import SensorConfig
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)

DEVKIT_ROOT = os.getenv('NAVSIM_DEVKIT_ROOT')
TRAJ_PDM_ROOT = os.getenv('NAVSIM_TRAJPDM_ROOT')


class _LazyTokenScores:
    """Dict-like lazy loader for per-token ori vocab PDM scores.

    Loads ``{dir}/{token}.pkl`` on demand and keeps a small LRU cache, so each
    process holds only a handful of tokens instead of the ~15GB monolithic
    pickle. Drop-in for the ``{token: {metric: array}}`` dict: callers keep
    doing ``scores[token][metric]`` unchanged.
    """

    def __init__(self, directory: str, cache_size: int = 512):
        self._dir = directory
        self._cache: "OrderedDict[str, Any]" = OrderedDict()
        self._cache_size = cache_size

    def __getitem__(self, token: str):
        cached = self._cache.get(token)
        if cached is not None:
            self._cache.move_to_end(token)
            return cached
        with open(os.path.join(self._dir, f"{token}.pkl"), "rb") as f:
            value = pickle.load(f)
        self._cache[token] = value
        if len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return value


class DriveSuprimAgent(AbstractAgent):
    def __init__(
            self,
            config: DriveSuprimConfig,
            lr: float,
            checkpoint_path: str = None,
            pdm_split=None,
            metrics=None,
    ):
        super().__init__(
            trajectory_sampling=config.trajectory_sampling
        )
        # The agent used to overwrite the weights unconditionally, which made
        # every per-dataset override in the config file dead on arrival. Only
        # the traffic-light entry still has to be derived here, because it
        # depends on another config flag.
        config.trajectory_pdm_weight = dict(config.trajectory_pdm_weight)
        if not config.use_traffic_light_compliance:
            config.trajectory_pdm_weight['traffic_light_compliance'] = 0.0

        self._config = config
        self._lr = lr
        # The sub-score names live in exactly one place: config.pdm_heads. The
        # agent YAML also carries a `metrics:` list, and the two silently
        # disagreed once the NuRec label dropped four of the eight -- the soft
        # teacher loss read `ori_vocab_pdm_score_full[token]['time_to_collision
        # _within_bound']` for a head that no longer exists. Deriving from the
        # config keeps them from drifting; the default pdm_heads is that same
        # eight-name list, so nothing changes for NAVSIM.
        metric_names = list(config.pdm_heads)
        if getattr(config, 'pdm_aggregate_head', False):
            # The soft-teacher path builds its distillation targets from
            # self.metrics; leaving the aggregate out of it would hand that head
            # `prediction.sum() * 0` there and supervise it on the hard label
            # only.
            metric_names.append('pdm_score')
        self.metrics = tuple(metric_names)
        self._checkpoint_path = checkpoint_path
        teacher_model = DriveSuprimModel(config)
        student_model = DriveSuprimModel(config)
        self.model = SSLMetaArch(config, teacher_model, student_model)
        self.vocab_size = config.vocab_size
        self.backbone_wd = config.backbone_wd
        self.training = config.training

        if self.training:
            if getattr(config, 'ori_vocab_pdm_score_dir', ''):
                # per-token lazy loading (avoids the ~15GB resident pickle)
                self.ori_vocab_pdm_score_full = _LazyTokenScores(config.ori_vocab_pdm_score_dir)
            else:
                self.ori_vocab_pdm_score_full = pickle.load(open(f'{config.ori_vocab_pdm_score_full_path}', 'rb'))
            # Offline rotation-augmentation artifacts are only needed when the
            # rotation ensemble is active (not `only_ori_input`, e.g. BEV path).
            if not config.only_ori_input:
                self.aug_vocab_pdm_score_dir = config.aug_vocab_pdm_score_dir

                with open(config.ego_perturb.offline_aug_file, 'r') as f:
                    aug_data = json.load(f)
                assert aug_data['param']['rot'] == config.ego_perturb.offline_aug_angle_boundary
                self.aug_info = aug_data['tokens']
        
        self.only_ori_input = config.only_ori_input
        self.n_rotation_angle = config.ego_perturb.n_student_rotation_ensemble
        self.load_pretrained_weights()
        self.freeze_perception_modules()

    def load_pretrained_weights(self) -> None:
        """Weights-only init from a previous stage's checkpoint.

        The training entry point (run_training_ssl.py) never calls
        ``initialize()``, so ``checkpoint_path`` alone had no effect during
        training -- a staged run configured with it would silently start from
        scratch. This is the equivalent of EAD's ``agent.checkpoint_path=`` for
        its phase 2: parameters are restored, but the optimizer state, epoch
        counter and LR schedule all start fresh.

        That distinction matters for stage 2. Lightning's full resume
        (``resume_ckpt_path`` / RESUME_CKPT) also restores the epoch counter, so
        resuming a 20-epoch stage-1 run into a 5-epoch stage 2 would stop
        immediately at max_epochs. Use full resume only to continue an
        interrupted run of the SAME stage.
        """
        if not self.training or not self._checkpoint_path:
            return
        ckpt = torch.load(self._checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("state_dict", ckpt)
        state_dict = {k.replace("agent.", "", 1): v for k, v in state_dict.items()}

        # strict=False suppresses missing and unexpected KEYS; a key that is
        # present with the wrong SHAPE still raises. Two config knobs legitimately
        # resize tensors across a stage boundary -- vocab_size resizes
        # _trajectory_head.vocab, bev_keyval_grid resizes _keyval_embedding -- and
        # both of those live entirely in the trajectory head, which stage 1 never
        # trains (planning_loss_weight=0). Dropping them keeps the fresh init,
        # which is what stage 2 would have started from anyway.
        own = self.state_dict()
        resized = {k: (tuple(v.shape), tuple(own[k].shape)) for k, v in state_dict.items()
                   if k in own and own[k].shape != v.shape}
        for k in resized:
            del state_dict[k]
        for k, (was, now) in resized.items():
            print(f"[load_pretrained_weights] shape 불일치로 제외, 새 init 사용: "
                  f"{k} {was} -> {now}")

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"[load_pretrained_weights] {self._checkpoint_path}: "
              f"loaded={len(state_dict)} missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) > 0.5 * len(list(self.state_dict())):
            raise RuntimeError(
                f"{len(missing)} tensors missing from {self._checkpoint_path} -- "
                "this is not a checkpoint of this architecture. Refusing to start "
                "a staged run on mostly-random weights.")

    # What stage 2 freezes: the shared perception trunk only.
    #
    # This follows what EAD's freeze_perception ACTUALLY does, which is not what
    # its own comment ("perception prefix list") suggests. Its prefix list names
    # `_agent_head`, `cls_branches` and `reg_branches`, none of which exist on its
    # model -- the real modules are `detection_head`, `det_cls_branches` and
    # `det_reg_branches`, and `"det_cls_branches".startswith("cls_branches")` is
    # False. Its drivable-area head (`drivable_area_bev_seg_head`) is not listed
    # at all. So in EAD's phase 2 the backbone is fixed while the detection and
    # drivable heads keep training alongside the planner.
    #
    # Two things EAD does freeze that we do not, because in this model they live
    # inside the head rather than beside it: the detection query embedding
    # (`det_query_emb`) and the per-head input projections (`proj_det` etc.).
    #
    # Matched on whole dotted-path components rather than substrings, because the
    # real parameter names are prefixed by the SSL wrapper
    # ("student.model._backbone....", "teacher.model....").
    PERCEPTION_MODULES = (
        "_backbone",
    )

    def _is_perception(self, dotted_name: str) -> bool:
        return any(part in self.PERCEPTION_MODULES for part in dotted_name.split("."))

    def freeze_perception_modules(self) -> None:
        """Stage 2 of the three-stage schedule trains planning on a fixed
        perception, as EAD's phase 2 does: requires_grad=False AND eval mode, so
        neither the weights nor any normalisation statistics move."""
        if not getattr(self._config, "freeze_perception", False):
            return
        n_par = 0
        for name, p in self.named_parameters():
            if self._is_perception(name):
                p.requires_grad = False
                n_par += 1
        n_mod = 0
        for name, m in self.named_modules():
            if name and self._is_perception(name):
                m.eval()
                n_mod += 1
        print(f"[freeze_perception] froze {n_par} parameters across {n_mod} modules")

    def train(self, mode: bool = True):
        # Keep the frozen perception in eval no matter how often Lightning flips
        # the module back into train mode between epochs.
        super().train(mode)
        if mode and getattr(self._config, "freeze_perception", False):
            for name, m in self.named_modules():
                if name and self._is_perception(name):
                    m.eval()
        return self

    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def reset_stream(self) -> None:
        """Clear the BEV temporal cache on both branches.

        Only meaningful with ``bev_streaming``. Call it whenever the next frame
        does not continue the previous one -- a new log, a new drive, a seek, or
        any dropped frames. Forgetting to call it makes the model fuse a stale
        BEV warped by an ego pose that never happened; calling it unnecessarily
        only costs one uncached forward.

        Safe to call when streaming is off, or on a backbone that has no cache.
        """
        for branch in ("teacher", "student"):
            m = getattr(getattr(self.model, branch, None), "model", None)
            enc = getattr(getattr(m, "_backbone", None), "image_encoder", None)
            if enc is not None and hasattr(enc, "reset_stream"):
                enc.reset_stream()

    def initialize(self) -> None:
        """Inherited, see superclass."""
        state_dict: Dict[str, Any] = torch.load(self._checkpoint_path, map_location=torch.device("cpu"))["state_dict"]
        self.load_state_dict({k.replace("agent.", ""): v for k, v in state_dict.items()}, strict=False)

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        history = [0, 1, 2, 3]
        camera_frames = {name: [] for name in ("cam_f0", "cam_l0", "cam_l1", "cam_l2", "cam_r0", "cam_r1", "cam_r2", "cam_b0")}
        camera_count = (
            self._config.bev_num_cameras
            if self._config.backbone_type == "bevformer_m"
            else self._config.n_camera
        )
        selected = {
            1: ("cam_f0",),
            3: ("cam_l0", "cam_f0", "cam_r0"),
            5: ("cam_l1", "cam_l0", "cam_f0", "cam_r0", "cam_r1"),
        }[camera_count]
        for name in selected:
            camera_frames[name] = history

        # Pixel-space rotation augmentation cannot be synthesized from only
        # three cameras because no pixels exist outside L0/F0/R0's FoV.
        if self._config.backbone_type != "bevformer_m" and not self._config.only_ori_input:
            if camera_count < 5:
                raise ValueError(
                    "three-camera DriveSuprim requires only_ori_input=true; "
                    "pixel-space rotation augmentation needs a wider camera rig"
                )
            for name in ("cam_l2", "cam_r2", "cam_b0"):
                camera_frames[name] = history
        return SensorConfig(
            **camera_frames,
            lidar_pc=[],
        )

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        return [DriveSuprimTargetBuilder(config=self._config)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        return [DriveSuprimFeatureBuilder(config=self._config)]

    def _rotated_route(self, features, index: int) -> Dict[str, torch.Tensor]:
        """Route for rotation ``index`` of the student ensemble.

        The waypoints turn with the augmentation while the validity mask does
        not: rotating a route does not change which of its slots are padding.
        """
        if not self._config.use_route:
            return {}
        return {
            'route_feature': features['route_feature_rotated'][index],
            'route_mask': features['route_mask'],
        }

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        
        features, targets, tokens = batch
        kwargs = {'tokens': tokens}

        teacher_ori_features = dict()
        student_ori_features = dict()

        if self._config.backbone_type == 'bevformer_m':
            # BEV front-end: teacher gets the clean multi-view images, student the
            # photometric-augmented ones; calibration / ego poses are shared.
            teacher_ori_features['bev_imgs'] = features['bev_imgs_teacher']
            student_ori_features['bev_imgs'] = features['bev_imgs']
            for _k in ('lidar2img', 'bev_ego_pose'):
                teacher_ori_features[_k] = features[_k]
                student_ori_features[_k] = features[_k]
        else:
            teacher_ori_features['camera_feature'] = features['ori_teacher']
            student_ori_features['camera_feature'] = features['ori']

        teacher_ori_features['status_feature'] = features['status_feature']
        student_ori_features['status_feature'] = features['status_feature']
        if self._config.use_route:
            for _dict in (teacher_ori_features, student_ori_features):
                _dict['route_feature'] = features['route_feature']
                _dict['route_mask'] = features['route_mask']

        # Aux heads (detection + BEV seg) run ONLY on the student's un-rotated
        # ("ori") forward: that BEV is in the un-rotated ego frame the GT raster
        # is aligned to, and it avoids the extra compute on the teacher and the
        # rotation-ensemble forwards. Gated per-forward via this flag.
        if self._config.use_aux_heads and self._config.training:
            student_ori_features['compute_aux'] = True

        student_feat_dict_lst = []
        student_feat_dict_lst.append(student_ori_features)

        if not self.only_ori_input and self._config.training:
            for i in range(self.n_rotation_angle):
                if self._config.backbone_type == 'bevformer_m':
                    # Same images, projection rotated by aug angle i -> the BEV is
                    # built in the rotated ego frame, matching rotated_trajectories
                    # and the aug vocab scores (both keyed on the same angles).
                    student_feat_dict_lst.append(
                        {
                            'bev_imgs': features['bev_imgs'],
                            'lidar2img': features['bev_lidar2img_rotated'][i],
                            'bev_ego_pose': features['bev_ego_pose'],
                            'bev_aug_yaw': features['bev_aug_yaw'][i],
                            'status_feature': features['status_feature'],
                            **self._rotated_route(features, i),
                        }
                    )
                else:
                    student_feat_dict_lst.append(
                        {
                            'camera_feature': features['rotated'][i],
                            'status_feature': features['status_feature'],
                            **self._rotated_route(features, i),
                        }
                    )

        teacher_pred, student_preds = self.model(teacher_ori_features, student_feat_dict_lst, **kwargs)
        
        return teacher_pred, student_preds

    def compute_loss(
            self,
            features: Dict[str, torch.Tensor],
            targets: Dict[str, torch.Tensor],
            predictions: List[Dict[str, torch.Tensor]],
            tokens=None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the student loss using ground truth.
        """

        # ori data
        ori_targets = { 'trajectory': targets['ori_trajectory'] }
        ori_predictions = predictions[0]
        scores = {}
        for k in self.metrics:
            tmp = [self.ori_vocab_pdm_score_full[token][k][None] for token in tokens]
            scores[k] = (torch.from_numpy(np.concatenate(tmp, axis=0))
                        .to(ori_predictions['trajectory'].device))
        ori_loss = drivesuprim_agent_loss_first_stage(ori_targets, ori_predictions, self._config, scores)

        # Auxiliary detection + BEV segmentation loss (BEVFormer only). The
        # student's ori prediction carries `bev_seg_map` / `agent_states` /
        # `agent_labels`; fold the aux loss into the "ori" term so it flows
        # through existing logging and the total-loss sum.
        if self._config.use_aux_heads and 'bev_seg_map' in ori_predictions:
            aux_total, aux_dict = drivesuprim_aux_loss(targets, ori_predictions, self._config)
            ori_loss = (ori_loss[0] + aux_total, {**ori_loss[1], **aux_dict})
            # Quality metrics (detection AP / centre-distance errors, seg IoU)
            # alongside the losses, so detection and segmentation can be tracked
            # during training rather than only inferred from a falling loss.
            # Detached and no-grad; skipped when the metric interval says so.
            if getattr(self._config, 'aux_log_metrics', True):
                from navsim.agents.drivesuprim.drivesuprim_det_metrics import aux_metrics
                # Separate counters and intervals per phase. `self.model.training`
                # is what Lightning toggles between fit and validate, so it is the
                # phase flag; one shared counter would let a training interval of
                # N silently thin the validation metrics N-fold as well.
                in_val = not self.model.training
                attr = '_aux_metric_step_val' if in_val else '_aux_metric_step'
                key = ('aux_metric_every_n_steps_val' if in_val
                       else 'aux_metric_every_n_steps')
                setattr(self, attr, getattr(self, attr, 0) + 1)
                every = max(1, int(getattr(self._config, key, 1)))
                if getattr(self, attr) % every == 0:
                    detached = {k: (v.detach() if torch.is_tensor(v) else v)
                                for k, v in ori_predictions.items()}
                    ori_loss[1].update(aux_metrics(targets, detached, self._config))

        if self._config.only_ori_input:
            return { "ori": ori_loss }

        # aug data
        _aug_vocab_pdm_score = {}
        for token in tokens:
            with open(os.path.join(self.aug_vocab_pdm_score_dir, f'{token}.pkl'), 'rb') as f:
                _aug_vocab_pdm_score[token] = pickle.load(f)
        aug_loss = []
        for idx in range(self._config.ego_perturb.n_student_rotation_ensemble):
            aug_targets = { 'trajectory': targets['rotated_trajectories'][idx] }
            scores = {}
            for k in self.metrics:
                tmp = [_aug_vocab_pdm_score[token][idx][k][None] for token in tokens]
                scores[k] = (torch.from_numpy(np.concatenate(tmp, axis=0))
                            .to(predictions[idx+1]['trajectory'].device))
            aug_loss.append(drivesuprim_agent_loss_first_stage(aug_targets, predictions[idx+1], self._config, scores))
        
        # Calculate average loss and loss dict
        avg_aug_loss = torch.mean(torch.stack([loss[0] for loss in aug_loss]))
        avg_aug_loss_dict = {}
        for key in aug_loss[0][1].keys():
            avg_aug_loss_dict[key] = torch.mean(torch.stack([loss[1][key] for loss in aug_loss]))
        return {
            "ori": ori_loss,
            "aug": (avg_aug_loss, avg_aug_loss_dict),
        }
    
    def compute_loss_soft_teacher(
            self,
            teacher_pred: Dict[str, torch.Tensor],
            student_pred: Dict[str, torch.Tensor],
            targets,
            tokens=None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the student loss using teacher soft label.
        """
        # Stage 1 skips the teacher forward entirely (see SSLMetaArch.forward),
        # so there is no soft label to distil from. Return a real zero tensor
        # rather than a float so the caller can still add and log it.
        if teacher_pred is None:
            zero = student_pred['trajectory'].sum() * 0.0
            return zero, {'imi_loss': zero.detach()}

        sampled_timepoints = [5 * ii - 1 for ii in range(1, 9)]
        if self._config.soft_label_traj == 'first':
            traj_diff = teacher_pred['trajectory'][:, sampled_timepoints] - targets['ori_trajectory']
        elif self._config.soft_label_traj == 'final':
            traj_diff = teacher_pred['final_traj'][:, sampled_timepoints] - targets['ori_trajectory']

        clamped_traj_diff = torch.clamp(traj_diff, min=-self._config.soft_label_imi_diff_thresh, max=self._config.soft_label_imi_diff_thresh)
        # Apply clamped adjustment to original trajectory
        revised_targets = { 'trajectory': targets['ori_trajectory'] + clamped_traj_diff }

        scores = {}
        revised_scores = {}
        for k in self.metrics:
            tmp = [self.ori_vocab_pdm_score_full[token][k][None] for token in tokens]
            scores[k] = torch.from_numpy(np.concatenate(tmp, axis=0)).to(teacher_pred['trajectory'].device).float()
            # Calculate difference and clamp to max 0.2
            diff = teacher_pred[k].sigmoid() - scores[k]
            _soft_label_score_diff_thresh = self._config.soft_label_score_diff_thresh
            clamped_diff = torch.clamp(diff, min=-_soft_label_score_diff_thresh, max=_soft_label_score_diff_thresh)
            # Apply clamped adjustment to original scores
            revised_scores[k] = scores[k] + clamped_diff
        
        soft_loss = drivesuprim_agent_loss_first_stage(revised_targets, student_pred, self._config, revised_scores)
        return soft_loss
    
    
    def compute_loss_multi_stage(
        self,
        features,
        targets: Dict[str, torch.Tensor],
        predictions: List[Dict[str, torch.Tensor]],
        tokens=None
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:

        _refinement_metrics = self.metrics
        trajectory_vocab = predictions[0]['trajectory_vocab']
        result_dict = dict()

        # ori data
        ori_loss_lst = []
        ori_predictions = predictions[0]['refinement']
        num_refinement_stage = len(ori_predictions)
        for i in range(num_refinement_stage):
            pred_i = ori_predictions[i]
            selected_indices_i = pred_i['indices_absolute']
            scores = {}
            for k in _refinement_metrics:
                tmp = [self.ori_vocab_pdm_score_full[token][k][None] for token in tokens]
                full_scores = torch.from_numpy(np.concatenate(tmp, axis=0)).to(selected_indices_i.device)  # [bs, vocab_size]
                # Extract scores based on selected indices [bs, topk_stage_i]
                batch_size, topk = selected_indices_i.shape
                batch_indices = torch.arange(batch_size, device=selected_indices_i.device).unsqueeze(1).expand(-1, topk)
                scores[k] = full_scores[batch_indices, selected_indices_i]  # [bs, topk_stage_i]
            _kwargs = {}
            if self._config.refinement.use_imi_learning_in_refinement:
                _kwargs['targets'] = { 'trajectory': targets['ori_trajectory'] }
                pred_i['trajectory_vocab'] = trajectory_vocab
            ori_loss_i = drivesuprim_agent_loss_single_refine_stage(pred_i, self._config, scores, **_kwargs)
            ori_loss_lst.append(ori_loss_i)
        total_ori_loss = sum([loss_tup[0] for loss_tup in ori_loss_lst])  # sum over all refinement stages (we only use 1 refinement stage)
        total_ori_loss_dict = {}
        for i, loss_tup in enumerate(ori_loss_lst):
            loss_dict = loss_tup[1]
            for _key, _value in loss_dict.items():
                total_ori_loss_dict[f"stage_{i+2}_{_key}"] = _value
        result_dict['ori'] = (total_ori_loss, total_ori_loss_dict)
        if self._config.only_ori_input:
            return result_dict

        # aug data
        _aug_vocab_pdm_score = {}
        for token in tokens:
            with open(os.path.join(self.aug_vocab_pdm_score_dir, f'{token}.pkl'), 'rb') as f:
                _aug_vocab_pdm_score[token] = pickle.load(f)
        aug_loss_all_mode_lst = []  # a mode means a specific rotation angle
        for idx in range(self._config.ego_perturb.n_student_rotation_ensemble):
            aug_loss_lst = []
            aug_idx_predictions = predictions[idx+1]['refinement']
            for i in range(num_refinement_stage):
                aug_idx_pred_i = aug_idx_predictions[i]
                aug_idx_selected_indices_i = aug_idx_pred_i['indices_absolute']
                scores = {}
                for k in _refinement_metrics:
                    tmp = [_aug_vocab_pdm_score[token][idx][k][None] for token in tokens]
                    full_scores = torch.from_numpy(np.concatenate(tmp, axis=0)).to(aug_idx_selected_indices_i.device)
                    batch_size, topk = aug_idx_selected_indices_i.shape
                    batch_indices = torch.arange(batch_size, device=aug_idx_selected_indices_i.device).unsqueeze(1).expand(-1, topk)
                    scores[k] = full_scores[batch_indices, aug_idx_selected_indices_i]
                _kwargs_idx = {}
                if self._config.refinement.use_imi_learning_in_refinement:
                    _kwargs_idx['targets'] = { 'trajectory': targets['rotated_trajectories'][idx] }
                    aug_idx_pred_i['trajectory_vocab'] = trajectory_vocab
                aug_loss_lst.append(drivesuprim_agent_loss_single_refine_stage(aug_idx_pred_i, self._config, scores, **_kwargs_idx))
            aug_loss_single_mode = sum([loss_tup[0] for loss_tup in aug_loss_lst])
            aug_loss_single_mode_dict = {}
            for i, loss_tup in enumerate(aug_loss_lst):
                loss_dict = loss_tup[1]
                for _key, _value in loss_dict.items():
                    aug_loss_single_mode_dict[f"stage_{i+2}_{_key}"] = _value
            aug_loss_all_mode_lst.append((aug_loss_single_mode, aug_loss_single_mode_dict))
        
        # Calculate average loss and loss dict across all (3) rotation angles
        avg_aug_loss = torch.mean(torch.stack([loss[0] for loss in aug_loss_all_mode_lst]))
        avg_aug_loss_dict = {}
        for key in aug_loss_all_mode_lst[0][1].keys():
            avg_aug_loss_dict[key] = torch.mean(torch.stack([loss[1][key] for loss in aug_loss_all_mode_lst]))
        result_dict['aug'] = (avg_aug_loss, avg_aug_loss_dict)

        return result_dict

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        backbone_params_name = '_backbone.image_encoder'
        # For backbone_type='bevformer_m' the whole BEV front-end lives under
        # `_backbone.image_encoder` -- the per-view image backbone AND the
        # from-scratch pyramid / BEV encoder / temporal fusion. Fine-tuning a
        # pretrained ViT-L at the same LR as those from-scratch modules wrecks it,
        # so `lr_mult_img_backbone` optionally splits the per-view image backbone
        # into its own group. Leave it None to keep the original two-group layout
        # (for backbone_type='vit'/'resnet*' there is no `.image_backbone`
        # submodule, so this branch is a no-op there either way).
        vit_params_name = '_backbone.image_encoder.image_backbone'
        lr_mult_img = getattr(self._config, 'lr_mult_img_backbone', None)
        # Frozen (stage-2) parameters are dropped here rather than left in the
        # optimizer with requires_grad=False: Adam would otherwise still hold
        # state for them, and an empty param group is easier to spot than a
        # silently-not-updating one.
        named = [(k, v) for k, v in self.model.student.model.named_parameters()
                 if v.requires_grad]

        default_params = [v for k, v in named if backbone_params_name not in k]
        if lr_mult_img is None:
            img_backbone_params = [v for k, v in named if backbone_params_name in k]
            params_lr_dict = [
                {'params': default_params, 'lr_scale': 1.0},
                {
                    'params': img_backbone_params,
                    'lr': self._lr * self._config.lr_mult_backbone,
                    'lr_scale': self._config.lr_mult_backbone,
                    'weight_decay': self.backbone_wd
                }
            ]
        else:
            vit_params = [v for k, v in named if vit_params_name in k]
            bev_params = [
                v for k, v in named
                if backbone_params_name in k and vit_params_name not in k
            ]
            params_lr_dict = [
                {'params': default_params, 'lr_scale': 1.0},
                {
                    'params': bev_params,
                    'lr': self._lr * self._config.lr_mult_backbone,
                    'lr_scale': self._config.lr_mult_backbone,
                    'weight_decay': self.backbone_wd
                },
                {
                    'params': vit_params,
                    'lr': self._lr * lr_mult_img,
                    'lr_scale': lr_mult_img,
                    'weight_decay': self.backbone_wd
                },
            ]
            print(
                f"[optimizer] groups: default={len(default_params)} "
                f"bev_frontend={len(bev_params)} (lr x{self._config.lr_mult_backbone}) "
                f"img_backbone={len(vit_params)} (lr x{lr_mult_img})")
        opt_type = getattr(self._config, 'optimizer_type', 'adam')
        wd = float(getattr(self._config, 'weight_decay', 0.0))
        if opt_type == 'adamw':
            optimizer = torch.optim.AdamW(params_lr_dict, lr=self._lr, weight_decay=wd)
        elif opt_type == 'adam':
            optimizer = torch.optim.Adam(params_lr_dict, lr=self._lr, weight_decay=wd)
        else:
            raise ValueError(f"Unknown optimizer_type: {opt_type}")
        print(f"[optimizer] {opt_type} lr={self._lr} weight_decay={wd} "
              f"(backbone groups keep backbone_wd={self.backbone_wd})")

        # EAD's schedule: linear warmup then cosine to min_lr, stepped per epoch.
        # 'none' keeps the previous behaviour (constant LR for the whole run),
        # so existing experiments are unaffected unless they opt in.
        sched_type = getattr(self._config, 'scheduler_type', 'none')
        if sched_type == 'none':
            return optimizer
        if sched_type != 'cos':
            raise ValueError(f"Unknown scheduler_type: {sched_type}")

        from navsim.agents.drivesuprim.scheduler import WarmupCosLR
        epochs = int(self._config.scheduler_epoch)
        warm = int(self._config.scheduler_warm_epoch)
        # scheduler_epoch is the length of the cosine, NOT the length of the run.
        # EAD leaves it at 30 for all three phases while running them for 20 / 5 /
        # 30 epochs, i.e. phases 1 and 2 deliberately stop part-way down the
        # curve. Do not "fix" one to match the other.
        print(f"[optimizer] WarmupCosLR: lr={self._lr} warmup={warm}ep "
              f"cosine_to={self._config.scheduler_min_lr} over {epochs}ep "
              f"(the run may stop earlier -- EAD truncates the cosine on purpose)")
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self._lr,
            min_lr=float(self._config.scheduler_min_lr),
            epochs=epochs,
            warmup_epochs=warm,
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    def get_training_callbacks(self) -> List[pl.Callback]:
        return [
            # TransfuserCallback(self._config),
            ModelCheckpoint(
                save_top_k=30,
                monitor="val/loss-ori",
                mode="min",
                dirpath=f"{os.environ.get('NAVSIM_EXP_ROOT')}/{self._config.ckpt_path}/",
                filename="{epoch:02d}-{step:04d}",
            )
        ]
