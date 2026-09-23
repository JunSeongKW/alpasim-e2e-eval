import logging
import os
from typing import Dict, Tuple

import pytorch_lightning as pl
import torch
from torch import Tensor

from nuplan.common.actor_state.vehicle_parameters import get_pacifica_parameters
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.agents.abstract_agent import AbstractAgent
from navsim.agents.drivesuprim.drivesuprim_config import DriveSuprimConfig
from navsim.agents.drivesuprim.drivesuprim_agent import DriveSuprimAgent
from navsim.common.dataclasses import Trajectory
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator


_LEGACY_PREDICTION_HEADS = (
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "time_to_collision_within_bound",
    "ego_progress",
    "driving_direction_compliance",
    "lane_keeping",
    "traffic_light_compliance",
)
"""Sub-score heads of agents predating ``config.pdm_heads``, in the order they
used to be unpacked here."""


logger = logging.getLogger(__name__)


class AgentLightningModuleSSL(pl.LightningModule):
    """Pytorch lightning wrapper for learnable agent."""

    def __init__(self,
                 cfg: DriveSuprimConfig,
                 agent: AbstractAgent):
        """
        Initialise the lightning module wrapper.
        :param agent: agent interface in NAVSIM
        """
        super().__init__()

        self._cfg = cfg
        self.agent: DriveSuprimAgent = agent
        self.simulator = PDMSimulator(
            TrajectorySampling(num_poses=40, interval_length=0.1)
        )
        self.v_params = get_pacifica_parameters()

        self.only_ori_input = cfg.only_ori_input


    def _step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], logging_prefix: str) -> Tensor:
        """
        Propagates the model forward and backwards and computes/logs losses and metrics.
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param logging_prefix: prefix where to log step
        :return: scalar loss
        """
        features, targets, tokens = batch

        teacher_pred, student_preds = self.agent.forward(batch)

        # compute loss for student (using ground truth)
        loss_student = self.agent.compute_loss(features, targets, student_preds, tokens)

        ori_loss = loss_student['ori']
        for k, v in ori_loss[1].items():
            self.log(f"{logging_prefix}/{k}-ori", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{logging_prefix}/loss-ori", ori_loss[0], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        loss = ori_loss[0]

        if not self.only_ori_input:
            aug_loss = loss_student['aug']
            for k, v in aug_loss[1].items():
                self.log(f"{logging_prefix}/{k}-aug", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/loss-aug", aug_loss[0], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            loss = loss + aug_loss[0]

        # compute loss for student (using teacher soft label)
        loss_soft_teacher = self.agent.compute_loss_soft_teacher(teacher_pred, student_preds[0], targets, tokens)
        for k, v in loss_soft_teacher[1].items():
            self.log(f"{logging_prefix}/{k}-soft", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{logging_prefix}/loss-soft", loss_soft_teacher[0], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        loss = loss + loss_soft_teacher[0]

        if self._cfg.refinement.use_multi_stage:
            loss_refinement = self.agent.compute_loss_multi_stage(features, targets, student_preds, tokens)
            
            loss_refinement_ori = loss_refinement['ori']
            for k, v in loss_refinement_ori[1].items():
                self.log(f"{logging_prefix}/{k}-ori", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log(f"{logging_prefix}/loss-refinement_ori", loss_refinement_ori[0], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
            loss = loss + loss_refinement_ori[0]

            if not self.only_ori_input:
                loss_refinement_aug = loss_refinement['aug']
                for k, v in loss_refinement_aug[1].items():
                    self.log(f"{logging_prefix}/{k}-aug", v, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                self.log(f"{logging_prefix}/loss-refinement_aug", loss_refinement_aug[0], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
                loss = loss + loss_refinement_aug[0]
        
        self.log(f"{logging_prefix}/loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        
        return loss

    def training_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int) -> Tensor:
        """
        Step called on training samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "train")
    
    def on_train_start(self):
        self.agent.model.train()
        # Optional: locate the exact module that first produces NaN/Inf.
        # Enable with env var BEVFORMER_NAN_DEBUG=1.
        from navsim.agents.backbones.bevformer.nan_debug import register_nan_hooks
        register_nan_hooks(self.agent.model)

    def optimizer_step(
        self,
        epoch: int,
        batch_idx: int,
        optimizer,
        optimizer_closure = None,
    ) -> None:

        # Iteration-based linear warmup, the way BEVFormer / UniAD do it
        # (warmup_iters=500, warmup_ratio=1/3). The epoch scheduler sets each
        # group's LR once per epoch; during warmup we scale that value and then
        # put it back, so the two do not fight. Warmup is expected to finish
        # inside the first epoch.
        warm_iters = int(getattr(self._cfg, 'scheduler_warmup_iters', 0) or 0)
        if warm_iters > 0 and self.trainer.global_step < warm_iters:
            ratio = float(getattr(self._cfg, 'scheduler_warmup_ratio', 1.0 / 3))
            k = self.trainer.global_step / warm_iters
            factor = 1.0 - (1.0 - k) * (1.0 - ratio)
            for g in optimizer.param_groups:
                if '_warmup_base_lr' not in g:
                    g['_warmup_base_lr'] = g['lr']
                g['lr'] = g['_warmup_base_lr'] * factor
            optimizer.step(closure=optimizer_closure)
            for g in optimizer.param_groups:
                g['lr'] = g.pop('_warmup_base_lr')
        else:
            optimizer.step(closure=optimizer_closure)

        if self._cfg.backbone_type in ('resnet34', 'resnet50'):
            if epoch < 3:
                m = 0
            elif epoch < 6:
                m = 0.992 + (epoch-3) * 0.002
            else:
                m = 0.998
        else:
            if epoch < 3:
                m = 0.992 + epoch * 0.002
            else:
                m = 0.998
        self.agent.model.update_teacher(m)

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        # The Trainer's gradient_clip_val cannot be used together with our custom
        # optimizer_step, so clip here. The precision plugin calls this hook AFTER
        # unscaling (for AMP), so it is the correct, AMP-safe place to clip. This
        # guards against fp16/bf16 gradient blow-ups that were causing NaN losses
        # partway through training.
        #
        # Drop a non-finite gradient before clipping rather than after. Clipping
        # by norm computes clip_coef = max_norm / total_norm, which is 0 when the
        # norm is inf -- and inf * 0 is NaN, so the clip that is meant to contain
        # a blow-up converts it into permanent corruption instead. fp16 has a
        # GradScaler that skips such a step; bf16-mixed has no scaler at all, so
        # without this the first bad batch writes NaN into the weights and every
        # loss reported afterwards is NaN forever.
        if self._drop_nonfinite_gradients(optimizer):
            return
        self.clip_gradients(optimizer, gradient_clip_val=35.0, gradient_clip_algorithm="norm")

    def _drop_nonfinite_gradients(self, optimizer) -> bool:
        """Zeroes the gradients if any is non-finite. True when that happened.

        Zeroing rather than skipping the step: the step is issued from inside
        ``optimizer.step(closure=...)`` and cannot be called off from here. A
        zero gradient leaves Adam applying decayed momentum only, which is a far
        smaller error than either a NaN weight or an unclipped blow-up.
        """
        parameters = [
            p
            for group in optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        if not parameters:
            return False
        if all(torch.isfinite(p.grad).all() for p in parameters):
            return False

        bad = sum(1 for p in parameters if not torch.isfinite(p.grad).all())
        self._nonfinite_grad_steps = getattr(self, "_nonfinite_grad_steps", 0) + 1
        for p in parameters:
            p.grad.zero_()
        # getattr rather than self.trainer: a guard that can itself raise is
        # worse than no guard.
        step = getattr(getattr(self, "trainer", None), "global_step", -1)
        logger.warning(
            "non-finite gradient at step %s: %d of %d tensors; gradient dropped "
            "(%d such steps so far)",
            step,
            bad,
            len(parameters),
            self._nonfinite_grad_steps,
        )
        return True

    def validation_step(self, batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]], batch_idx: int):
        """
        Step called on validation samples
        :param batch: tuple of dictionaries for feature and target tensors (batched)
        :param batch_idx: index of batch (ignored)
        :return: scalar loss
        """
        return self._step(batch, "val")

    def configure_optimizers(self):
        """Inherited, see superclass."""
        return self.agent.get_optimizers()

    def predict_step(
            self,
            batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
            batch_idx: int
    ):
        return self.predict_step_drivesuprim(batch, batch_idx)
    

    def predict_step_drivesuprim(
            self,
            batch: Tuple[Dict[str, Tensor], Dict[str, Tensor]],
            batch_idx: int
    ):
        features, targets, tokens = batch
        self.agent.eval()
        with torch.no_grad():
            predictions, _ = self.agent.forward(batch)
            if self._cfg.refinement.use_multi_stage and not self._cfg.inference.use_first_stage_traj_in_infer:
                poses = predictions['final_traj'].cpu().numpy()
            else:
                poses = predictions["trajectory"].cpu().numpy()

            # Which sub-scores exist is the agent's choice: the NuRec agent drops
            # the five that do not enter its label, so naming them here would
            # raise. The trajectory has already been chosen inside the model, by
            # the ranking these same heads feed; what is collected below is
            # per-candidate diagnostics that ride along with it.
            head_names = tuple(getattr(self._cfg, "pdm_heads", None) or _LEGACY_PREDICTION_HEADS)
            imis = predictions["imi"].softmax(-1).log().cpu().numpy()
            sub_scores = {
                name: predictions[name].sigmoid().log().cpu().numpy()
                for name in head_names
                if name in predictions
            }

        if poses.shape[1] == 40:
            interval_length = 0.1
        else:
            interval_length = 0.5


        result = {}
        for idx, (pose, imi, token) in enumerate(zip(poses, imis, tokens)):
            result[token] = {
                'trajectory': Trajectory(pose, TrajectorySampling(time_horizon=4, interval_length=interval_length)),
                'imi': imi,
            }
            for name, values in sub_scores.items():
                result[token][name] = values[idx]
        return result
