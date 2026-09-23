from typing import Dict, Optional, List
import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from scipy.optimize import linear_sum_assignment

from navsim.agents.drivesuprim.drivesuprim_config import DriveSuprimConfig
from navsim.agents.transfuser.transfuser_loss import _get_ce_cost, _get_l1_cost, _get_src_permutation_idx


def _bev_seg_loss(logits, target, config, class_weights):
    """BEV segmentation cross-entropy, following EAD's bev_semantic loss.

    Three things it does that a bare F.cross_entropy does not:

    * class weighting. EAD passes inverse-frequency weights normalised so they
      sum to the class count ([0.2121, 0.3693, 0.8473, 0.7267, 1.8056, 1.0974,
      1.9416] for its 7 classes -- sum 7.0). The equivalent measured on this
      drivable target is in ``aux_bev_seg_class_weights``.
    * a validity mask. Labels outside [0, num_classes) are dropped rather than
      clamped, and the mean is taken over the surviving cells only, so an
      unlabelled region costs nothing instead of training the head towards an
      arbitrary class.
    * NaN / inf guards on the logits before the softmax.
    """
    logits = torch.nan_to_num(logits, nan=0.0, posinf=50.0, neginf=-50.0)
    clamp = float(getattr(config, "aux_bev_logit_clamp", 0.0))
    if clamp > 0:
        logits = logits.clamp(min=-clamp, max=clamp)

    target = target.long()
    num_classes = logits.shape[1]
    valid = (target >= 0) & (target < num_classes)
    if not bool(valid.any()):
        # Nothing supervisable in this batch; keep the graph alive with a zero.
        return logits.sum() * 0.0

    w = None
    if class_weights is not None:
        w = torch.as_tensor(tuple(class_weights), dtype=torch.float32, device=logits.device)
    flat_logits = logits.permute(0, 2, 3, 1)[valid]
    flat_target = target[valid]
    loss = F.cross_entropy(flat_logits.float(), flat_target, weight=w, reduction="mean")
    return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)


def _get_l1_cost_nd(gt_states, pred_states, gt_valid, dims: int):
    """BEVFormer's BBox3DL1Cost: plain L1 over the first ``dims`` box dimensions.

    The transfuser cost this replaces compares centres only (``[..., :2]``), so
    two queries sharing a centre but differing in size, height or heading are
    indistinguishable to the matcher. BEVFormer's HungarianAssigner3D scores
    ``bbox_pred[:, :8]`` -- the whole box except vx, vy -- with no code_weights
    at the matching stage (those apply to the regression loss only).

    Returns [b, num_pred, num_gt], the same layout as ``_get_l1_cost``, with the
    cost of matching an invalid GT slot masked to zero.
    """
    g = gt_states[..., :dims].detach()
    p = pred_states[..., :dims].detach()
    cost = torch.cdist(p, g, p=1)                       # [b, num_pred, num_gt]
    return cost * gt_valid[:, None, :].to(cost.dtype)


def _agent_loss_ds(targets, predictions, config, with_prediction: bool):
    """DETR-style Hungarian-matched agent loss for DriveSuprim. Runs the matching
    ONCE (class + box cost) and computes classification, box-regression and --
    when ``with_prediction`` -- future-trajectory L1 under the SAME assignment,
    so detection and forecasting are supervised consistently per matched agent.
    Returns (ce_loss, box_loss, pred_loss|None)."""
    gt_states, gt_valid = targets["agent_states"], targets["agent_labels"]
    pred_states, pred_logits = predictions["agent_states"], predictions["agent_labels"]

    batch_dim, num_instances = pred_states.shape[:2]
    # avg_factor. BEVFormer normalises both losses by ``num_total_pos`` after
    # reduce_mean across ranks and clamp(min=1): every GPU divides by the SAME
    # average positive count, so a rank that happens to draw a crowded scene does
    # not shrink its own per-box gradient. Dividing by the local count instead
    # makes the effective loss scale rank-dependent.
    # fp32 regardless of AMP: this is a divisor, and it is all-reduced below.
    num_gt = gt_valid.sum().float()
    if (getattr(config, "aux_agent_sync_avg_factor", True)
            and dist.is_available() and dist.is_initialized()):
        num_gt = num_gt.clone()
        dist.all_reduce(num_gt)
        num_gt = num_gt / dist.get_world_size()
    num_gt = num_gt.clamp(min=1.0)

    # Multi-class head: pred_logits is [B, NQ, C]. Score each query against the
    # GT slot's own class channel, so matching is class-aware like the original
    # head's FocalLossCost. Single-class keeps the old [B, NQ] objectness path.
    multiclass = pred_logits.dim() == 3
    if multiclass:
        gt_cls = targets["agent_classes"].long()                       # [B, NB]
        # Detached like _get_ce_cost / _get_l1_cost: the cost matrix only drives
        # the Hungarian assignment and must not carry gradients.
        pred_prob = pred_logits.detach().sigmoid()                     # [B, NQ, C]
        if getattr(config, "aux_agent_use_focal_match_cost", True):
            # mmdet FocalLossCost, the cost BEVFormer's HungarianAssigner3D uses:
            #   neg = -log(1 - p) * (1 - alpha) * p^gamma
            #   pos = -log(p)     * alpha       * (1 - p)^gamma
            #   cost[q, g] = pos[q, cls(g)] - neg[q, cls(g)]
            # Subtracting the negative term is what makes assigning a query to a
            # GT account for the background loss it stops paying; a plain -p cost
            # ignores that and matches differently on confident queries.
            gamma = config.aux_agent_focal_gamma
            alpha = config.aux_agent_focal_alpha
            eps = 1e-12
            neg = -(1 - pred_prob + eps).log() * (1 - alpha) * pred_prob.pow(gamma)
            pos = -(pred_prob + eps).log() * alpha * (1 - pred_prob).pow(gamma)
            gather_idx = gt_cls[:, None, :].expand(-1, pred_prob.shape[1], -1)
            ce_cost = (pos.gather(2, gather_idx) - neg.gather(2, gather_idx))
        else:
            # cost[b, q, g] = -p(query q is class of gt g)
            ce_cost = -pred_prob.gather(
                2, gt_cls[:, None, :].expand(-1, pred_prob.shape[1], -1))
        ce_cost = ce_cost * gt_valid[:, None, :].to(ce_cost.dtype)
        # objectness view (max over classes) for the per-query BCE fallbacks below
        pred_obj_logits = pred_logits.max(-1).values
    else:
        ce_cost = _get_ce_cost(gt_valid, pred_logits)
        pred_obj_logits = pred_logits
    match_dims = int(getattr(config, "aux_agent_match_cost_dims", 0))
    if match_dims and pred_states.shape[-1] >= match_dims and gt_states.shape[-1] >= match_dims:
        l1_cost = _get_l1_cost_nd(gt_states, pred_states, gt_valid, match_dims)
    else:
        # 5-dim transfuser layout, or matching explicitly restricted to centres.
        l1_cost = _get_l1_cost(gt_states, pred_states, gt_valid)
    cost = (config.aux_agent_class_weight * ce_cost + config.aux_agent_box_weight * l1_cost).cpu()

    indices = [linear_sum_assignment(c) for c in cost]
    matching = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    idx = _get_src_permutation_idx(matching)

    # box regression (matched, masked by GT validity)
    pred_states_idx = pred_states[idx]
    gt_states_idx = torch.cat([t[i] for t, (_, i) in zip(gt_states, indices)], dim=0)
    pred_valid_idx = pred_obj_logits[idx]
    gt_valid_idx = torch.cat([t[i] for t, (_, i) in zip(gt_valid, indices)], dim=0).float()

    # Per-dimension weights on the box L1, mirroring BEVFormer's code_weights.
    code_w = (getattr(config, "aux_agent_code_weights_3d", None)
              if getattr(config, "aux_agent_box_3d", False)
              else getattr(config, "aux_agent_code_weights", None))
    box_l1 = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none")
    if code_w is not None:
        w = torch.as_tensor(tuple(code_w), dtype=box_l1.dtype, device=box_l1.device)
        box_l1 = box_l1 * w
    # BEVFormer drops rows whose target is not finite (`isnotnan`) before the L1:
    # log(0) on a degenerate zero-extent box would otherwise poison the batch.
    # The row has to be zeroed BEFORE the reduction -- masking afterwards would
    # multiply an inf by 0 and produce NaN.
    finite = torch.isfinite(gt_states_idx).all(dim=-1)
    box_l1 = torch.where(finite[:, None], box_l1, torch.zeros_like(box_l1))
    box_loss = box_l1.sum(-1) * gt_valid_idx
    box_loss = box_loss.view(batch_dim, -1).sum() / num_gt

    def _focal(logits, target):
        """Sigmoid focal loss, FocalLoss(gamma=2.0, alpha=0.25) as in the head."""
        gamma, alpha = config.aux_agent_focal_gamma, config.aux_agent_focal_alpha
        p = torch.sigmoid(logits)
        ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p_t = p * target + (1 - p) * (1 - target)
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        return alpha_t * ((1 - p_t) ** gamma) * ce

    use_focal = getattr(config, "aux_agent_use_focal", False)
    if multiclass:
        # One-hot over classes on matched queries; unmatched queries are all-zero
        # (background) across every channel -- the DETR sigmoid-focal convention.
        num_cls = pred_logits.shape[-1]
        gt_cls_idx = torch.cat([t[i] for t, (_, i) in zip(targets["agent_classes"].long(), indices)], dim=0)
        # Cast AFTER the multiply: gt_valid_idx is fp32, so casting the one-hot
        # first lets type promotion pull the product back to fp32 and the
        # index_put into the fp16 target_full fails under AMP.
        onehot_matched = (F.one_hot(gt_cls_idx, num_cls) * gt_valid_idx[:, None]).to(pred_logits.dtype)
        target_full = torch.zeros_like(pred_logits)
        target_full[idx] = onehot_matched
        loss_map = _focal(pred_logits, target_full) if use_focal else \
            F.binary_cross_entropy_with_logits(pred_logits, target_full, reduction="none")
        ce_loss = loss_map.sum() / num_gt
    elif use_focal:
        ce_loss = _focal(pred_valid_idx, gt_valid_idx).view(batch_dim, -1).sum() / num_gt
    else:
        ce_loss = F.binary_cross_entropy_with_logits(
            pred_valid_idx, gt_valid_idx, reduction="none").view(batch_dim, -1).mean()

    pred_loss = None
    if with_prediction:
        pred_fut = predictions["agent_future"]          # [B, num_inst, T, 2]
        gt_fut = targets["agent_future_trajectory"]     # [B, num_bb, T, 2]
        gt_fmask = targets["agent_future_mask"]         # [B, num_bb, T]
        pred_fut_idx = pred_fut[idx]                                                # [M, T, 2]
        gt_fut_idx = torch.cat([t[i] for t, (_, i) in zip(gt_fut, indices)], dim=0)   # [M, T, 2]
        gt_fmask_idx = torch.cat([t[i] for t, (_, i) in zip(gt_fmask, indices)], dim=0)  # [M, T]
        # only matched real agents AND future steps where the agent is present
        step_mask = gt_fmask_idx * gt_valid_idx[:, None]
        fut_l1 = F.l1_loss(pred_fut_idx, gt_fut_idx, reduction="none").sum(-1) * step_mask  # [M, T]
        pred_loss = fut_l1.sum() / step_mask.sum().clamp(min=1.0)

    # BEVFormer guards both terms the same way before returning them.
    ce_loss = torch.nan_to_num(ce_loss)
    box_loss = torch.nan_to_num(box_loss)
    return ce_loss, box_loss, pred_loss


def drivesuprim_aux_loss(
        targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: DriveSuprimConfig
):
    """Auxiliary dense supervision for the BEVFormer BEV features:
      - BEV segmentation cross-entropy (target is the binary drivable area, or
        the multi-class static map when aux_bev_drivable_only=False),
      - agent detection (Hungarian-matched class + box regression), and
      - agent motion prediction (future-trajectory L1 under the same matching).
    Returns (total, loss_dict)."""
    bev_seg_loss = _bev_seg_loss(
        predictions["bev_seg_map"], targets["bev_seg_map"], config,
        getattr(config, "aux_bev_seg_class_weights", None),
    )
    # Second head for the dual-seg configuration (multi-class semantic + a
    # separate drivable map); off when aux_bev_drivable_only is set.
    bev_drivable_loss = None
    if "bev_drivable_map" in predictions and "bev_drivable_map" in targets:
        bev_drivable_loss = _bev_seg_loss(
            predictions["bev_drivable_map"], targets["bev_drivable_map"], config,
            getattr(config, "aux_bev_drivable_class_weights", None),
        )
    with_prediction = getattr(config, "aux_predict_agents", False) and ("agent_future" in predictions)
    agent_class_loss, agent_box_loss, agent_pred_loss = _agent_loss_ds(
        targets, predictions, config, with_prediction
    )

    aux_decoder_loss = predictions["bev_seg_map"].new_tensor(0.0)
    if getattr(config, "aux_agent_head_type", "transfuser") == "bevformer" and "agent_states_layers" in predictions:
        layer_pairs = list(zip(predictions["agent_states_layers"][:-1], predictions["agent_labels_layers"][:-1]))
        if layer_pairs:
            layer_losses = []
            for states_l, labels_l in layer_pairs:
                cls_l, box_l, _ = _agent_loss_ds(
                    targets,
                    {"agent_states": states_l, "agent_labels": labels_l},
                    config,
                    with_prediction=False,
                )
                layer_losses.append(config.aux_agent_class_weight * cls_l + config.aux_agent_box_weight * box_l)
            # BEVFormer emits every decoder layer's loss as its own entry in the
            # loss dict, so they are SUMMED at full weight -- each intermediate
            # layer supervises as strongly as the final one. Averaging instead
            # (the previous behaviour) scales the 5 intermediate layers down to
            # 1/5 each. 'mean' is kept for the old scale.
            stacked = torch.stack(layer_losses)
            aux_decoder_loss = (stacked.sum()
                                if getattr(config, "aux_decoder_layer_reduction", "sum") == "sum"
                                else stacked.mean())

    bev_seg_final = config.aux_bev_seg_weight * bev_seg_loss
    agent_class_final = config.aux_agent_class_weight * agent_class_loss
    agent_box_final = config.aux_agent_box_weight * agent_box_loss

    total = bev_seg_final + agent_class_final + agent_box_final + aux_decoder_loss
    if bev_drivable_loss is not None:
        total = total + config.aux_bev_drivable_weight * bev_drivable_loss
    loss_dict = {
        "aux_bev_seg_loss": bev_seg_final,
        "aux_agent_class_loss": agent_class_final,
        "aux_agent_box_loss": agent_box_final,
    }
    if bev_drivable_loss is not None:
        loss_dict["aux_bev_drivable_loss"] = config.aux_bev_drivable_weight * bev_drivable_loss
    if aux_decoder_loss.detach().abs().item() > 0:
        loss_dict["aux_agent_decoder_loss"] = aux_decoder_loss
    if agent_pred_loss is not None:
        agent_pred_final = config.aux_pred_weight * agent_pred_loss
        total = total + agent_pred_final
        loss_dict["aux_agent_pred_loss"] = agent_pred_final
    return total, loss_dict


def bce_loss_with_temperature(
        predictions: torch.Tensor, targets: torch.Tensor, temperature: float = 1.0
) -> torch.Tensor:
    """
    Binary Cross Entropy loss with temperature scaling.
    :param predictions: model predictions (logits)
    :param targets: ground truth labels
    :param temperature: temperature scaling factor
    :return: computed loss value
    """
    if temperature != 1.0:
        predictions = predictions / temperature
    return F.binary_cross_entropy_with_logits(predictions, targets)


def bce_loss_with_label_smoothing(
        predictions: torch.Tensor, targets: torch.Tensor, label_smoothing_value: float = 0.1,
) -> torch.Tensor:
    """
    Binary Cross Entropy loss with label smoothing.
    :param predictions: model predictions (logits)
    :param targets: ground truth labels
    :param label_smoothing_value: value for label smoothing
    :return: computed loss value
    """
    smooth_targets = targets * (1 - label_smoothing_value) + 0.5 * label_smoothing_value
    smooth_targets = smooth_targets.clamp(min=1e-7, max=1 - 1e-7)
    return F.binary_cross_entropy_with_logits(predictions, smooth_targets, reduction='mean')




def _sub_score_losses(predictions, vocab_pdm_score, config, dtype):
    """Binary cross-entropy per configured sub-score head.

    Only the heads the config declares are read, and only where the weight is
    non-zero: a term the label holds at a constant (traffic light on NuRec, or
    anything the score no longer multiplies) contributes
    ``prediction.sum() * 0`` instead.  That keeps every parameter in the graph
    -- DDP trips over a head that receives no gradient at all -- while spending
    no supervision on a constant.
    """
    losses, total = {}, None
    names = list(config.pdm_heads)
    if getattr(config, 'pdm_aggregate_head', False):
        names.append('pdm_score')
    for name in names:
        prediction = predictions[name]
        weight = (
            config.pdm_aggregate_loss_weight if name == 'pdm_score'
            else config.trajectory_pdm_weight.get(name, 0.0)
        )
        available = name in vocab_pdm_score
        if name == 'traffic_light_compliance' and not getattr(
            config, 'use_traffic_light_compliance', True
        ):
            available = False
        if weight == 0.0 or not available:
            term = prediction.sum() * 0.0
        else:
            target = vocab_pdm_score[name].to(dtype)
            if name in config.pdm_three_class_terms:
                target = three_to_two_classes(target)
            term = weight * F.binary_cross_entropy_with_logits(prediction, target)
        losses[name] = term
        total = term if total is None else total + term
    return total, losses


_LOSS_LOG_NAME = {
    'no_at_fault_collisions': 'pdm_noc_loss',
    'drivable_area_compliance': 'pdm_da_loss',
    'time_to_collision_within_bound': 'pdm_ttc_loss',
    'ego_progress': 'pdm_progress_loss',
    'driving_direction_compliance': 'pdm_ddc_loss',
    'lane_keeping': 'pdm_lk_loss',
    'traffic_light_compliance': 'pdm_tl_loss',
    'history_comfort': 'pdm_comfort_loss',
    'pdm_score': 'pdm_aggregate_loss',
}


def drivesuprim_agent_loss_first_stage(
        targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor], config: DriveSuprimConfig,
        vocab_pdm_score
):
    """
    Helper function calculating loss of DriveSuprim first stage (coarse filtering)
    """
    
    imi = predictions['imi']
    _dtype = imi.dtype
    pdm_total, pdm_losses = _sub_score_losses(predictions, vocab_pdm_score, config, _dtype)

    vocab = predictions["trajectory_vocab"]
    target_traj = targets["trajectory"]
    sampled_timepoints = [5 * k - 1 for k in range(1, 9)]
    B = target_traj.shape[0]
    l2_distance = -((vocab[:, sampled_timepoints][None].repeat(B, 1, 1, 1) - target_traj[:, None]) ** 2) / config.sigma
    """
    vocab: [vocab_size, 40, 3]
    vocab[:, sampled_timepoints]: [vocab_size, 8, 3]
    vocab[:, sampled_timepoints][None].repeat(B, 1, 1, 1): [b, vocab_size, 8, 3]
    target_traj[:, None]: [b, 1, 8, 3]
    l2_distance: [b, vocab_size, 8, 3]
    """
    imi_loss = F.cross_entropy(imi, l2_distance.sum((-2, -1)).softmax(1))

    imi_loss_final = config.trajectory_imi_weight * imi_loss

    loss = imi_loss_final + pdm_total
    # Stage-1 (perception-only) sets planning_loss_weight=0. Scaling by zero
    # rather than skipping the call keeps the graph intact, so every trajectory
    # parameter still receives a (zero) gradient and DDP does not trip over
    # unused parameters. The logged components are scaled too, so a run where
    # planning contributes nothing does not read as if it were still training.
    pw = getattr(config, "planning_loss_weight", 1.0)
    loss = pw * loss
    logged = {'imi_loss': pw * imi_loss_final}
    logged.update({_LOSS_LOG_NAME.get(k, f'pdm_{k}_loss'): pw * v for k, v in pdm_losses.items()})
    return loss, logged


def drivesuprim_agent_loss_single_refine_stage(
        predictions: Dict[str, torch.Tensor], config: DriveSuprimConfig, vocab_pdm_score, targets=None
):
    """
    Helper function calculating loss of single refinement stage of DriveSuprim
    """

    layer_results = predictions['layer_results']
    losses = {}
    total_loss = 0.0

    for layer, layer_result in enumerate(layer_results):
        _dtype = layer_result[config.pdm_heads[0]].dtype
        loss, _ = _sub_score_losses(layer_result, vocab_pdm_score, config, _dtype)

        if config.refinement.use_imi_learning_in_refinement:
            imi = layer_result['imi']
            vocab = predictions["trajectory_vocab"]
            target_traj = targets["trajectory"]
            sampled_timepoints = [5 * k - 1 for k in range(1, 9)]
            indices_absolute = predictions['indices_absolute']
            l2_distance = -((vocab[:, sampled_timepoints][indices_absolute] - target_traj[:, None]) ** 2) / config.sigma

            imi_loss = F.cross_entropy(imi, l2_distance.sum((-2, -1)).softmax(1))
            imi_loss_final = config.trajectory_imi_weight * imi_loss
            loss += imi_loss_final
        
        total_loss += loss
        losses[f'layer_{layer+1}'] = loss

    # Same planning-stage gate as the first stage, logged values included.
    pw = getattr(config, "planning_loss_weight", 1.0)
    total_loss = pw * total_loss
    losses = {k: pw * v for k, v in losses.items()}
    return total_loss, losses


def three_to_two_classes(x):
    x[x==0.5] = 0.0
    return x
