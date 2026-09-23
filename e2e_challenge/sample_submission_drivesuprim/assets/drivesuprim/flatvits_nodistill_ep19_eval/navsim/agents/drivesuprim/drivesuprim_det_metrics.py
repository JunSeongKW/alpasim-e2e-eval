"""Detection / segmentation quality metrics logged alongside the training loss.

The losses tell you the objective is going down; they do not tell you whether the
boxes land on the objects. These are cheap per-batch proxies computed under
no_grad so they can run every step:

  det/ap@<d>   BEV-centre-distance AP, the NuScenes-style matching criterion
               (a prediction matches a GT if their BEV centres are within d
               metres), averaged over the configured distance thresholds and
               over classes that are present in the batch.
  det/recall@<d>, det/prec@<d>
  det/err_xy   mean BEV centre error over matched pairs [m]
  det/err_yaw  mean absolute heading error over matched pairs [rad]
  det/err_vel  mean velocity error over matched pairs [m/s]  (3D boxes only)
  seg/iou      foreground IoU of the BEV segmentation
  seg/recall, seg/prec

These are batch-level estimates, not a dataset-level mAP: AP is computed from the
score-sorted precision/recall curve within the batch. Use them to watch training
progress; use a proper offline evaluation for a headline number.
"""

from typing import Dict, List

import torch


def _decode_boxes(states: torch.Tensor, box_3d: bool):
    """-> (centre_xy [N,2], yaw [N], velocity [N,2] | None)"""
    if box_3d:
        # [cx, cy, log(l), log(w), cz, log(h), sin, cos, vx, vy]
        xy = states[..., 0:2]
        yaw = torch.atan2(states[..., 6], states[..., 7])
        vel = states[..., 8:10]
    else:
        # [x, y, heading, length, width]
        xy = states[..., 0:2]
        yaw = states[..., 2]
        vel = None
    return xy, yaw, vel


@torch.no_grad()
def detection_metrics(
    targets: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    box_3d: bool,
    dist_thresholds: List[float] = (0.5, 1.0, 2.0, 4.0),
    score_threshold: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """Greedy score-ordered matching by BEV centre distance, per class."""
    gt_states = targets["agent_states"]
    gt_valid = targets["agent_labels"].bool()
    gt_cls = targets.get("agent_classes")
    pred_states = predictions["agent_states"]
    pred_logits = predictions["agent_labels"]

    if pred_logits.dim() == 3:
        pred_score, pred_cls = pred_logits.sigmoid().max(-1)
    else:
        pred_score, pred_cls = pred_logits.sigmoid(), torch.zeros_like(pred_logits, dtype=torch.long)
    if gt_cls is None:
        gt_cls = torch.zeros_like(gt_valid, dtype=torch.long)

    p_xy, p_yaw, p_vel = _decode_boxes(pred_states, box_3d)
    g_xy, g_yaw, g_vel = _decode_boxes(gt_states, box_3d)

    B = gt_states.shape[0]
    out: Dict[str, torch.Tensor] = {}
    err_xy, err_yaw, err_vel = [], [], []

    for thr in dist_thresholds:
        tps, fps, scores, n_gt = [], [], [], 0
        for b in range(B):
            keep = pred_score[b] > score_threshold
            order = torch.argsort(pred_score[b][keep], descending=True)
            pi = torch.nonzero(keep, as_tuple=False).squeeze(-1)[order]
            gi = torch.nonzero(gt_valid[b], as_tuple=False).squeeze(-1)
            n_gt += int(gi.numel())
            taken = torch.zeros(gi.numel(), dtype=torch.bool, device=gt_states.device)
            for k in pi.tolist():
                same = (gt_cls[b][gi] == pred_cls[b][k]) & (~taken)
                if not bool(same.any()):
                    tps.append(0.0); fps.append(1.0); scores.append(float(pred_score[b][k]))
                    continue
                d = torch.linalg.norm(g_xy[b][gi] - p_xy[b][k][None], dim=-1)
                d = torch.where(same, d, torch.full_like(d, float("inf")))
                j = int(torch.argmin(d))
                if float(d[j]) <= thr:
                    taken[j] = True
                    tps.append(1.0); fps.append(0.0)
                    if thr == max(dist_thresholds):
                        gj = int(gi[j])
                        err_xy.append(float(d[j]))
                        dy = p_yaw[b][k] - g_yaw[b][gj]
                        err_yaw.append(float(torch.atan2(dy.sin(), dy.cos()).abs()))
                        if p_vel is not None:
                            err_vel.append(float(torch.linalg.norm(p_vel[b][k] - g_vel[b][gj])))
                else:
                    tps.append(0.0); fps.append(1.0)
                scores.append(float(pred_score[b][k]))

        dev = gt_states.device
        if n_gt == 0 or not scores:
            out[f"det/ap@{thr}"] = torch.tensor(0.0, device=dev)
            out[f"det/recall@{thr}"] = torch.tensor(0.0, device=dev)
            out[f"det/prec@{thr}"] = torch.tensor(0.0, device=dev)
            continue
        s = torch.tensor(scores, device=dev)
        t = torch.tensor(tps, device=dev)[torch.argsort(s, descending=True)]
        f = torch.tensor(fps, device=dev)[torch.argsort(s, descending=True)]
        ctp, cfp = t.cumsum(0), f.cumsum(0)
        recall = ctp / max(n_gt, 1)
        prec = ctp / (ctp + cfp).clamp(min=1e-6)
        # VOC-style: integrate precision over recall increments
        ap = torch.zeros((), device=dev)
        prev_r = torch.zeros((), device=dev)
        prec_env = torch.flip(torch.cummax(torch.flip(prec, [0]), 0).values, [0])
        for r, p in zip(recall, prec_env):
            ap = ap + (r - prev_r) * p
            prev_r = r
        out[f"det/ap@{thr}"] = ap
        out[f"det/recall@{thr}"] = recall[-1]
        out[f"det/prec@{thr}"] = prec[-1]

    out["det/mAP"] = torch.stack([out[f"det/ap@{t}"] for t in dist_thresholds]).mean()
    dev = gt_states.device
    out["det/err_xy"] = torch.tensor(sum(err_xy) / len(err_xy) if err_xy else 0.0, device=dev)
    out["det/err_yaw"] = torch.tensor(sum(err_yaw) / len(err_yaw) if err_yaw else 0.0, device=dev)
    if box_3d:
        out["det/err_vel"] = torch.tensor(sum(err_vel) / len(err_vel) if err_vel else 0.0, device=dev)
    return out


@torch.no_grad()
def segmentation_metrics(
    targets: Dict[str, torch.Tensor],
    predictions: Dict[str, torch.Tensor],
    key: str = "bev_seg_map",
) -> Dict[str, torch.Tensor]:
    """Foreground IoU / recall / precision. For the binary drivable target the
    foreground is class 1; for a multi-class map it is 'anything but background',
    which keeps a single comparable number across configurations."""
    if key not in predictions or key not in targets:
        return {}
    pred = predictions[key].argmax(1)
    gt = targets[key].long()
    p_fg, g_fg = pred > 0, gt > 0
    inter = (p_fg & g_fg).sum().float()
    union = (p_fg | g_fg).sum().float()
    return {
        "seg/iou": inter / union.clamp(min=1.0),
        "seg/recall": inter / g_fg.sum().clamp(min=1).float(),
        "seg/prec": inter / p_fg.sum().clamp(min=1).float(),
    }


@torch.no_grad()
def aux_metrics(targets, predictions, config) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    if "agent_states" in predictions:
        out.update(detection_metrics(
            targets, predictions,
            box_3d=getattr(config, "aux_agent_box_3d", False),
            dist_thresholds=tuple(getattr(config, "det_metric_thresholds", (0.5, 1.0, 2.0, 4.0))),
        ))
    out.update(segmentation_metrics(targets, predictions))
    return out
