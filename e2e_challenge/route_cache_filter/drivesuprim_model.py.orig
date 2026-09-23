import copy
from itertools import combinations
import numpy as np
import os, pickle
from sklearn.cluster import KMeans
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from navsim.agents.drivesuprim.drivesuprim_backbone_pe import DriveSuprimBackbonePE
from navsim.agents.drivesuprim.drivesuprim_config import DriveSuprimConfig
from navsim.common.route_contract import ROUTE_SLOTS
from navsim.agents.transfuser.transfuser_model import AgentHead
from navsim.agents.transfuser.transfuser_features import BoundingBox2DIndex
from navsim.agents.utils.attn import MemoryEffTransformer
from navsim.agents.utils.nerf import nerf_positional_encoding


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))



def _sub_score_heads(config, d_model: int, d_ffn: int, with_imi: bool) -> nn.ModuleDict:
    """One head per sub-score the configured label actually carries.

    Built from ``config.pdm_heads`` rather than a literal dict so a dataset can
    drop the terms its label does not score.  On NuRec that is five of eight:
    lane keeping, driving direction, traffic light, time-to-collision and
    comfort do not enter ``NC x DAC x GT x EP``, and their exported values are
    held constants, so a head on them would spend capacity learning 1.0.
    """
    names = list(config.pdm_heads)
    if getattr(config, 'pdm_aggregate_head', False):
        names.append('pdm_score')
    heads = {
        name: nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, 1),
        )
        for name in names
    }
    if with_imi:
        heads['imi'] = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, 1),
        )
    return nn.ModuleDict(heads)


def config_imi_rank_weight(config) -> float:
    """Weight on the imitation term in the ranking score.

    Was hard-coded at 0.02. SafeDrive ships 0.3 (`imi_test_weight` in its
    test.sh), fifteen times more: how closely a proposal matches what a person
    did is a stronger prior than that. Configurable because, like the rest of
    the ranking, it can be swept without retraining.
    """
    return float(getattr(config, 'pdm_imi_rank_weight', 0.02))


def config_imi_rank_weight_stage(config, stage: str) -> float:
    """Imitation weight for one selector stage.

    The coarse cut ranks 4,096 candidates, many of them hopeless, so the product
    of the gates spans a real range there and the two terms are commensurable.
    The refinement stage ranks what survived that cut -- all decent by
    construction -- where the product spans ~0.18 against normalised imi's 1.0.
    The weight therefore has to differ per stage, not globally.
    """
    if stage == "refine":
        w = getattr(config, 'pdm_imi_rank_weight_refine', None)
        if w is not None:
            return float(w)
    return config_imi_rank_weight(config)


def _rank_product(config, result, use_traffic_light: bool, stage: str = "coarse"):
    """Rank by the product the label is made of, rather than a sum of logs.

    The label is `NC x DAC x GT x EP`, and a product has no way to trade a bad
    factor against a good one: multiplying by 0.005 costs the same whatever the
    other four say. A sum of logs does allow that trade, and at the shipped
    weights it is not hypothetical -- 34.2% of the 256 that survive the coarse
    cut at a standstill carry a collision label of zero.

    Kept in probability space rather than as a sum of logs so a head that
    saturates to exactly 0 gives exactly 0 instead of -inf, which would make
    every such candidate indistinguishable and hand the ordering to whatever
    ran next. `imi` is added by the caller either way: it is not a factor of
    the label and must not be able to veto.
    """
    score = None
    exponents = None
    if stage == "refine":
        exponents = getattr(config, 'pdm_rank_product_exponents_refine', None)
    if exponents is None:
        exponents = getattr(config, 'pdm_rank_product_exponents', None) or {}
    for name in config.pdm_rank_product_terms:
        if name not in result:
            continue
        if name == 'traffic_light_compliance' and not use_traffic_light:
            continue
        if name == 'imi':
            # imi is a softmax over the whole vocabulary, not a per-candidate
            # probability: it sums to 1, so a typical entry is 1/4096 and the
            # factor would be three orders of magnitude below the others and
            # decide the product on its own. The exponent is what brings it back
            # -- 0.004**0.2 is 0.33, beside gates that sit near 0.99 -- so a
            # small exponent here is not a weak preference, it is the only way
            # the term is commensurable at all.
            term = result[name].softmax(-1)
            if getattr(config, 'pdm_rank_imi_normalize', False):
                # Same normalisation the additive path applies, so putting imi
                # in `terms` and leaving it out do not disagree about what the
                # term is worth.
                term = term / term.amax(dim=-1, keepdim=True).clamp_min(1e-12)
        else:
            term = result[name].sigmoid()
        exponent = float(exponents.get(name, 1.0))
        if exponent != 1.0:
            term = term.clamp_min(1e-6) ** exponent
        score = term if score is None else score * term
    if score is not None and getattr(config, 'pdm_aggregate_head', False) \
            and 'pdm_score' in result and config.pdm_aggregate_rank_weight:
        # Same role as in the sum path: a bonus that ranks, not a factor that
        # vetoes, so it is added rather than multiplied in.
        score = score + config.pdm_aggregate_rank_weight * result['pdm_score'].sigmoid()
    return score


def _rank_imi_term(config, result, stage: str = "coarse"):
    """The imitation term, in whichever space the ranking is being built in.

    The sum path wants ``log softmax`` -- typically about -8.3 across a 4,096
    vocabulary -- which is the right magnitude beside gates that also arrive as
    logs. The product path is a probability in [0, 1], where that same term at
    weight 0.02 is -0.17: a sixth of the entire range of the thing it is
    supposed to nudge. So the product path takes the softmax itself, which is
    bounded like the factors it sits beside.
    """
    if getattr(config, 'pdm_rank_product', False):
        weights = result['imi'].softmax(-1)
        if getattr(config, 'pdm_rank_imi_normalize', False):
            weights = weights / weights.amax(dim=-1, keepdim=True).clamp_min(1e-12)
        return config_imi_rank_weight_stage(config, stage) * weights
    return config_imi_rank_weight_stage(config, stage) * result['imi'].softmax(-1).log()


def _rank_score(config, result, use_traffic_light: bool, stage: str = "coarse"):
    """The ranking score, assembled from the configured composition.

    Multiplicative terms enter as ``w * log(sigmoid(x))`` -- a term near zero
    drives the whole score down, which is what multiplying by it does.  The
    weighted terms share one log so their trade-off stays inside a single term,
    as in EPDMS.

    With ``pdm_rank_product`` the composition is the label's own product
    instead; see `_rank_product`. Both stages route through here, so the switch
    reaches the coarse 4096 -> 256 cut and the refinement 256 -> 1 alike.
    """
    if getattr(config, 'pdm_rank_product', False):
        return _rank_product(config, result, use_traffic_light, stage)
    score = None
    for name, weight in config.pdm_score_log.items():
        if name not in result or weight == 0.0:
            continue
        if name == 'traffic_light_compliance' and not use_traffic_light:
            continue
        term = weight * result[name].sigmoid().log()
        score = term if score is None else score + term
    inner = None
    for name, weight in config.pdm_score_sum.items():
        if name not in result or weight == 0.0:
            continue
        term = weight * result[name].sigmoid()
        inner = term if inner is None else inner + term
    if inner is not None:
        term = config.pdm_score_sum_scale * inner.log()
        score = term if score is None else score + term
    # The aggregate head is already a score in [0, 1]; adding its sigmoid keeps
    # it a bonus that ranks, where a log term would let one head veto a
    # candidate every component agrees on.
    if getattr(config, 'pdm_aggregate_head', False) and 'pdm_score' in result:
        term = config.pdm_aggregate_rank_weight * result['pdm_score'].sigmoid()
        score = term if score is None else score + term
    return score


class GRN(nn.Module):
    """Global Response Normalization (ConvNeXt V2). Operates on [B, H, W, C]."""

    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        nx = gx / (gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """ConvNeXt V2 block: 7x7 depthwise conv -> LN -> 4x pointwise -> GELU -> GRN
    -> pointwise -> residual.

    SafeDrive's BEV segmentation branch is described as "a ConvNeXt-v2 block
    with a lightweight segmentation head"; this is that block. The 7x7
    depthwise conv gives the drivable-area head the receptive field the previous
    Conv3x3 -> ReLU -> Conv1x1 stack lacked.
    """

    def __init__(self, dim: int, drop_path: float = 0.0):
        super().__init__()
        self.drop_path_prob = drop_path
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)                    # [B, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)                    # [B, C, H, W]
        if self.drop_path_prob > 0.0 and self.training:
            keep = 1.0 - self.drop_path_prob
            mask = x.new_empty(x.shape[0], 1, 1, 1).bernoulli_(keep) / keep
            x = x * mask
        return shortcut + x


class BEVDeformableCrossAttention(nn.Module):
    """Deformable cross-attention from object queries onto a single-level BEV map.

    Mirrors the original BEVFormer detection decoder, whose cross_attn is a
    CustomMSDeformableAttention: each query samples `num_points` locations
    around its own reference point instead of attending densely to all
    bev_h*bev_w tokens. That is what makes the layer-wise reference-point
    refinement actually do something -- with dense attention the refined
    reference never feeds back into where the query looks.
    """

    def __init__(self, embed_dims=256, num_heads=8, num_points=4, dropout=0.0):
        super().__init__()
        assert embed_dims % num_heads == 0
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_levels = 1
        self.head_dims = embed_dims // num_heads

        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        from navsim.agents.backbones.bevformer.spatial_cross_attention import (
            constant_init, xavier_init,
        )
        import math
        constant_init(self.sampling_offsets, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 2).repeat(1, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0.0, bias=0.0)
        xavier_init(self.value_proj, distribution="uniform", bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(self, query, value, reference_points, spatial_shapes, level_start_index,
                query_pos=None):
        """:param query: [bs, num_query, C]
        :param value: [bs, num_value, C] flattened BEV map
        :param reference_points: [bs, num_query, 2] normalized to [0, 1]
        """
        from navsim.agents.backbones.bevformer.spatial_cross_attention import _deform_attn

        bs, num_query, _ = query.shape
        residual = query
        v = self.value_proj(value).view(bs, value.shape[1], self.num_heads, self.head_dims)
        if query_pos is not None:
            query = query + query_pos

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points)

        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = (
            reference_points[:, :, None, None, None, :]
            + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        )
        out = _deform_attn(v, spatial_shapes, level_start_index,
                           sampling_locations, attention_weights)
        out = self.output_proj(out)
        return self.dropout(out) + residual


class BEVFormerStyleAgentHead(nn.Module):
    """BEVFormer-like 2D BEV object head for NAVSIM agent boxes.

    Keeps the NAVSIM/TransFuser target contract (`agent_states`, `agent_labels`)
    but uses BEVFormer detection-head ingredients: learned object queries,
    full-BEV decoder memory, per-layer cls/reg branches, and reference-point
    based xy refinement. The output box code remains ego-frame 2D:
    x, y, heading, length, width.
    """

    def __init__(self, config: DriveSuprimConfig):
        super().__init__()
        self._config = config
        self._num_objects = (
            getattr(config, "aux_bevformer_det_num_queries", None) or config.num_bounding_boxes
        )
        self._num_classes = len(getattr(config, "aux_agent_classes", ("vehicle",)))
        self._d_model = config.tf_d_model
        self._pc_range = config.point_cloud_range
        self._num_layers = config.aux_bevformer_det_layers

        from navsim.agents.backbones.bevformer.bevformer_backbone import (
            LearnedPositionalEncoding,
        )

        self._bev_h, self._bev_w = config.bev_h, config.bev_w
        self._memory_proj = nn.Conv2d(config.bev_embed_dims, self._d_model, kernel_size=1)
        # Factorised row/col encoding, as in the BEVFormer config, instead of a
        # flat nn.Embedding(bev_h*bev_w, d_model) (10.2M params at 200x200x256).
        self._bev_pos = LearnedPositionalEncoding(
            num_feats=self._d_model // 2,
            row_num_embed=config.bev_h,
            col_num_embed=config.bev_w,
        )
        # Original: query_embedding = nn.Embedding(num_query, embed_dims * 2),
        # split into (query_pos, query); reference points come from a Linear on
        # query_pos rather than being a free embedding.
        self._box_3d = getattr(config, "aux_agent_box_3d", False)
        # BEVFormer regresses 3D reference points (x, y, z); the legacy 2D path
        # keeps (x, y).
        self._ref_dim = 3 if self._box_3d else 2
        self._query_embedding = nn.Embedding(self._num_objects, self._d_model * 2)
        self._reference_proj = nn.Linear(self._d_model, self._ref_dim)
        # Original decoder layer order: self_attn -> norm -> cross_attn -> norm
        # -> ffn -> norm, with a DEFORMABLE cross_attn.
        drop = config.aux_bevformer_det_dropout
        self._self_attns = nn.ModuleList([
            nn.MultiheadAttention(self._d_model, config.vadv2_head_nhead,
                                  dropout=drop, batch_first=True)
            for _ in range(self._num_layers)
        ])
        self._cross_attns = nn.ModuleList([
            BEVDeformableCrossAttention(
                embed_dims=self._d_model, num_heads=config.vadv2_head_nhead,
                num_points=config.aux_bevformer_det_num_points, dropout=drop)
            for _ in range(self._num_layers)
        ])
        self._ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self._d_model, config.aux_bevformer_det_ffn),
                nn.ReLU(inplace=True),
                nn.Dropout(drop),
                nn.Linear(config.aux_bevformer_det_ffn, self._d_model),
                nn.Dropout(drop),
            )
            for _ in range(self._num_layers)
        ])
        self._norms = nn.ModuleList([
            nn.ModuleList([nn.LayerNorm(self._d_model) for _ in range(3)])
            for _ in range(self._num_layers)
        ])
        self._cls_branches = nn.ModuleList([self._make_cls_branch() for _ in range(self._num_layers)])
        self._reg_branches = nn.ModuleList([self._make_reg_branch() for _ in range(self._num_layers)])
        self._init_weights()

    def _init_weights(self) -> None:
        """Original head: bias_init_with_prob(0.01) on the final classification
        layer when the classifier is sigmoid-based. Starts every query at ~1%
        objectness instead of 50%, which matters here because only a handful of
        the queries are ever matched to a real box."""
        from navsim.agents.backbones.bevformer.spatial_cross_attention import xavier_init
        prior_prob = 0.01
        bias_init = float(-np.log((1 - prior_prob) / prior_prob))
        for branch in self._cls_branches:
            nn.init.constant_(branch[-1].bias, bias_init)
        xavier_init(self._reference_proj, distribution="uniform", bias=0.0)

    def _make_cls_branch(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self._d_model, self._d_model),
            nn.LayerNorm(self._d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self._d_model, self._d_model),
            nn.LayerNorm(self._d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self._d_model, self._num_classes),
        )

    def _make_reg_branch(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(self._d_model, self._d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self._d_model, self._d_model),
            nn.ReLU(inplace=True),
            nn.Linear(self._d_model, 10 if self._box_3d else BoundingBox2DIndex.size()),
        )

    def forward(self, bev_map: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = bev_map.shape[0]
        memory = self._memory_proj(bev_map).flatten(-2, -1).permute(0, 2, 1)
        memory = memory + self._bev_pos(batch_size, memory.device, memory.dtype)

        spatial_shapes = torch.tensor(
            [[self._bev_h, self._bev_w]], device=memory.device, dtype=torch.long)
        level_start_index = torch.tensor([0], device=memory.device, dtype=torch.long)

        query_pos, query = torch.split(
            self._query_embedding.weight[None].repeat(batch_size, 1, 1),
            self._d_model, dim=-1)
        reference = self._reference_proj(query_pos).sigmoid()
        layer_states = []
        layer_logits = []

        for i, (cls_branch, reg_branch) in enumerate(zip(self._cls_branches, self._reg_branches)):
            norm1, norm2, norm3 = self._norms[i]
            # torch 2.0.1's fused SDPA kernels (flash / mem-efficient) crash in
            # backward on this GPU (H200, sm_90): "illegal instruction". The
            # release predates the hardware. Pin the math backend, which is
            # exact and plenty fast for 30 queries.
            # query_pos is added to q/k only (not v), as in mmcv's MultiheadAttention.
            q = k = query + query_pos
            with torch.backends.cuda.sdp_kernel(
                    enable_flash=False, enable_mem_efficient=False, enable_math=True):
                attn_out, _ = self._self_attns[i](q, k, query, need_weights=False)
            query = norm1(query + attn_out)
            # The BEV deformable attention samples on a 2D grid, so only (x, y)
            # of the reference is used -- mirroring the original decoder's
            # `reference_points_input = reference_points[..., :2]`. The z channel
            # exists purely to be refined alongside the box.
            query = norm2(self._cross_attns[i](
                query, memory, reference[..., :2], spatial_shapes, level_start_index,
                query_pos=query_pos))
            query = norm3(query + self._ffns[i](query))
            # [bs, num_query, num_classes]; squeezed to [bs, num_query] for the
            # single-class case so the existing objectness contract is preserved.
            logits = cls_branch(query)
            if self._num_classes == 1:
                logits = logits.squeeze(-1)
            raw_state = reg_branch(query)

            if self._box_3d:
                # BEVFormer decoder: refine (x, y) from tmp[..., :2] and z from
                # tmp[..., 4:5], then sigmoid and denormalise into pc_range.
                # Output layout matches normalize_bbox:
                #   [cx, cy, log(l), log(w), cz, log(h), sin, cos, vx, vy]
                new_ref = torch.zeros_like(reference)
                new_ref[..., :2] = raw_state[..., :2] + inverse_sigmoid(reference[..., :2])
                new_ref[..., 2:3] = raw_state[..., 4:5] + inverse_sigmoid(reference[..., 2:3])
                new_ref = new_ref.sigmoid()
                x_min, y_min, z_min, x_max, y_max, z_max = self._pc_range
                cx = new_ref[..., 0:1] * (x_max - x_min) + x_min
                cy = new_ref[..., 1:2] * (y_max - y_min) + y_min
                cz = new_ref[..., 2:3] * (z_max - z_min) + z_min
                state = torch.cat([
                    cx, cy,
                    raw_state[..., 2:4],      # log(l), log(w)
                    cz,
                    raw_state[..., 5:6],      # log(h)
                    raw_state[..., 6:8],      # sin, cos
                    raw_state[..., 8:10],     # vx, vy
                ], dim=-1)
                next_ref = new_ref
            else:
                xy = (raw_state[..., BoundingBox2DIndex.POINT] + inverse_sigmoid(reference)).sigmoid()

                x_min, y_min, _, x_max, y_max, _ = self._pc_range
                point = torch.stack(
                    [
                        xy[..., 0] * (x_max - x_min) + x_min,
                        xy[..., 1] * (y_max - y_min) + y_min,
                    ],
                    dim=-1,
                )
                heading = raw_state[..., BoundingBox2DIndex.HEADING:BoundingBox2DIndex.HEADING + 1].tanh() * np.pi
                length = F.softplus(raw_state[..., BoundingBox2DIndex.LENGTH:BoundingBox2DIndex.LENGTH + 1])
                width = F.softplus(raw_state[..., BoundingBox2DIndex.WIDTH:BoundingBox2DIndex.WIDTH + 1])
                state = torch.cat([point, heading, length, width], dim=-1)

                next_ref = xy
            reference = next_ref.detach() if self._config.aux_bevformer_det_with_box_refine else reference
            layer_states.append(state)
            layer_logits.append(logits)

        return {
            "agent_states": layer_states[-1],
            "agent_labels": layer_logits[-1],
            "agent_states_layers": layer_states,
            "agent_labels_layers": layer_logits,
        }


class DriveSuprimModel(nn.Module):
    def __init__(self, config: DriveSuprimConfig):
        super().__init__()

        self._config = config
        # 'intern' / 'swin' / 'sptr' / 'eva' / 'moe' / 'moe_ult32' were experimental
        # backbone options with no supporting config in this repo; their code
        # (navsim/agents/backbones/{internimage,swin,eva}.py, ops_dcnv3/) was
        # removed as dead weight -- see DriveSuprimBackbonePE for the supported set.
        assert config.backbone_type in ['vit', 'vov', 'resnet34', 'resnet50', 'bevformer_m', 'vits_flat']
        self._backbone = DriveSuprimBackbonePE(config)

        self.is_bev_backbone = config.backbone_type == 'bevformer_m'

        img_num = 1
        # For the BEV front-end the token grid is the BEV grid (bev_h x bev_w),
        # otherwise it is the pooled image-feature grid. Optionally the first-stage
        # trajectory keyval is pooled to a smaller GxG grid (bev_keyval_grid) so
        # the 8192-vocab cross-attention stays cheap at large BEV resolutions.
        self.bev_keyval_grid = getattr(config, "bev_keyval_grid", 0) if self.is_bev_backbone else 0
        if self.is_bev_backbone:
            if self.bev_keyval_grid > 0:
                num_keyval_tokens = self.bev_keyval_grid * self.bev_keyval_grid
                self.bev_keyval_pool = nn.AdaptiveAvgPool2d((self.bev_keyval_grid, self.bev_keyval_grid))
            else:
                num_keyval_tokens = config.bev_h * config.bev_w
                self.bev_keyval_pool = None
        else:
            num_keyval_tokens = config.img_vert_anchors * config.img_horz_anchors * img_num
        self._keyval_embedding = nn.Embedding(
            num_keyval_tokens, config.tf_d_model
        )

        # usually, the BEV features are variable in size.
        self.downscale_layer = nn.Conv2d(self._backbone.img_feat_c, config.tf_d_model, kernel_size=1)
        self._status_encoding = nn.Linear((4 + 2 + 2) * config.num_ego_status, config.tf_d_model)
        # Route replaces the driving command as the intent signal, but is added
        # as its own term rather than widening _status_encoding, so a checkpoint
        # trained without it still loads.
        self._route_encoding = RouteEncoder(config.tf_d_model, config.route_hidden_dim) \
            if config.use_route else None

        self._trajectory_head = HydraTrajHead(
            num_poses=config.trajectory_sampling.num_poses,
            d_ffn=config.tf_d_ffn,
            d_model=config.tf_d_model,
            nhead=config.vadv2_head_nhead,
            nlayers=config.vadv2_head_nlayers,
            vocab_path=config.vocab_path,
            config=config
        )

        # --- Auxiliary dense heads (BEVFormer only) --------------------------
        # A BEV semantic-segmentation head + a DETR-style agent detection head
        # attached to the BEV feature grid. Only meaningful for the BEV backbone
        # (which yields a metric BEV grid aligned to point_cloud_range). Built
        # whenever the flag is set so the params exist for checkpoint load; the
        # actual forward is gated per-call by `compute_aux` (see forward()).
        self.use_aux_heads = config.use_aux_heads and self.is_bev_backbone
        if self.use_aux_heads:
            # Binary drivable-area seg (2 classes) by default, else full multi-class.
            if config.aux_bev_drivable_only:
                n_seg_classes = 2
            elif hasattr(config, "num_active_bev_classes"):
                n_seg_classes = config.num_active_bev_classes
            else:
                n_seg_classes = config.num_bev_classes
            # SafeDrive: "a ConvNeXt-v2 block with a lightweight segmentation
            # head to produce static BEV segmentation maps". One shared trunk;
            # the classifier(s) on top are the "lightweight head".
            self._seg_trunk = nn.Sequential(
                *[ConvNeXtV2Block(config.bev_embed_dims)
                  for _ in range(config.aux_seg_convnext_blocks)]
            )
            self._bev_seg_head = nn.Sequential(
                self._seg_trunk,
                nn.Conv2d(config.bev_embed_dims, n_seg_classes,
                          kernel_size=1, stride=1, padding=0, bias=True),
            )
            # Second classifier for the separate drivable-area term.
            self._dual_seg = getattr(config, "aux_bev_dual_seg", False)
            if self._dual_seg:
                self._bev_drivable_head = nn.Conv2d(
                    config.bev_embed_dims, 2, kernel_size=1, stride=1, padding=0, bias=True)
            self._aux_agent_head_type = config.aux_agent_head_type
            if self._aux_agent_head_type == "transfuser":
                self._agent_query_embedding = nn.Embedding(
                    config.num_bounding_boxes, config.tf_d_model
                )
                self._agent_tf_decoder = nn.TransformerDecoder(
                    nn.TransformerDecoderLayer(
                        config.tf_d_model, config.vadv2_head_nhead, config.tf_d_ffn,
                        dropout=0.0, batch_first=True,
                    ),
                    config.aux_agent_tf_layers,
                )
                self._agent_head = AgentHead(
                    num_agents=config.num_bounding_boxes,
                    d_ffn=config.tf_d_ffn,
                    d_model=config.tf_d_model,
                )
            elif self._aux_agent_head_type == "bevformer":
                self._agent_head = BEVFormerStyleAgentHead(config)
            else:
                raise ValueError(f"Unknown aux_agent_head_type: {self._aux_agent_head_type}")
            # Motion prediction: per-agent future trajectory (x,y) over
            # aux_pred_num_poses steps, in the current ego frame. Shares the
            # agent decoder output; raw linear regression (no tanh clamp) like
            # the ego TrajectoryHead, since futures can exceed the BEV extent.
            self.predict_agents = config.aux_predict_agents
            if self.predict_agents:
                self._agent_pred_num_poses = config.aux_pred_num_poses
                self._agent_prediction_head = nn.Sequential(
                    nn.Linear(config.tf_d_model, config.tf_d_ffn),
                    nn.ReLU(),
                    nn.Linear(config.tf_d_ffn, config.aux_pred_num_poses * 2),
                )
            else:
                self.predict_agents = False

        self.use_multi_stage = self._config.refinement.use_multi_stage
        if self.use_multi_stage:
            if self._config.refinement.refinement_approach == 'transformer_decoder':
                self._trajectory_offset_head = RefineTrajHead(
                    d_ffn=config.tf_d_ffn,
                    d_model=config.tf_d_model,
                    nhead=config.vadv2_head_nhead,
                    d_backbone=self._backbone.img_feat_c,
                    num_stage=config.refinement.num_refinement_stage,
                    stage_layers=config.refinement.stage_layers,
                    topks=config.refinement.topks,
                    config=config
                )
            else:
                raise NotImplementedError

    def img_feat_blc_dict(self, camera_feature, **kwargs):
        img_features = self._backbone(camera_feature, **kwargs)
        img_feat_dict = {
            'patch_token': img_features,  # [bs, c_img, h, w] (FULL BEV res -> aux heads + refinement)
        }
        # First-stage trajectory keyval: optionally pool the BEV feature to a
        # smaller grid so the 8192-vocab cross-attention stays cheap at 200x200.
        feat_for_keyval = img_features
        if self.is_bev_backbone and self.bev_keyval_pool is not None:
            feat_for_keyval = self.bev_keyval_pool(img_features)
        kv = self.downscale_layer(feat_for_keyval).flatten(-2, -1)  # [bs, c, hk * wk]
        kv = kv.permute(0, 2, 1)  # [bs, hk * wk, c]
        img_feat_dict['avg_feat'] = kv
        return img_feat_dict

    def forward_features_list(self, x_list):
        # Rotation-ensemble views share the same images (only lidar2img / yaw
        # differ), so extract the per-frame image features ONCE here and reuse
        # them across all views instead of re-running ResNet+FPN per view.
        if self.is_bev_backbone and len(x_list) > 1 and 'bev_imgs' in x_list[0]:
            shared = self._backbone.image_encoder.extract_img_feats_seq(x_list[0]['bev_imgs'])
            x_list = [{**x, 'bev_precomputed_img_feats': shared} for x in x_list]
        return [self.forward(x) for x in x_list]

    def _route_inputs(self, features: Dict[str, torch.Tensor]):
        """Route tensors for this forward, or ``(None, None)`` when it is off.

        A configured route that never reached the features is a wiring error,
        not something to skip: training would otherwise run for days without the
        signal it was configured to use and report nothing.
        """
        if not getattr(self._config, "use_route", False):
            return None, None
        missing = [key for key in ("route_feature", "route_mask") if features.get(key) is None]
        if missing:
            raise KeyError(f"use_route=True but the features are missing {missing}")
        return features["route_feature"], features["route_mask"]

    def forward(self,
                features: Dict[str, torch.Tensor],
                masks=None,
                tokens=None) -> Dict[str, torch.Tensor]:
        
        output: Dict[str, torch.Tensor] = {}

        status_feature: torch.Tensor = features["status_feature"][0]  # tensor.shape == [bs, 8] (features["status_feature"][0] picks present status)

        if self.is_bev_backbone:
            # BEV front-end consumes the multi-view images + calibration + ego
            # poses directly (assembled by the feature builder), not a single
            # stitched perspective image.
            backbone_input = {
                "bev_imgs": features["bev_imgs"],
                "lidar2img": features["lidar2img"],
                "ego_pose": features.get("bev_ego_pose", None),
                "bev_aug_yaw": features.get("bev_aug_yaw", None),
                "bev_precomputed_img_feats": features.get("bev_precomputed_img_feats", None),
            }
            img_feat_dict = self.img_feat_blc_dict(backbone_input)
        else:
            camera_feature: List[torch.Tensor] = features["camera_feature"]  # List[torch.Tensor], len == seq_len, tensor.shape == [bs, 3, h, w]
            if isinstance(camera_feature, list):
                camera_feature = camera_feature[-1]  # [bs, 3, h, w], [-1] means present frame camera input
            img_feat_dict = self.img_feat_blc_dict(camera_feature, masks=masks, return_class_token=False)

        route, route_mask = self._route_inputs(features)

        status_encoding = self._status_encoding(status_feature)  # [bs, 8] -> [bs, c]
        if self._route_encoding is not None:
            status_encoding = status_encoding + self._route_encoding(route, route_mask)

        keyval = img_feat_dict.pop('avg_feat')  # [bs, h_avg * w_avg, c]
        keyval += self._keyval_embedding.weight[None, ...]  # [bs, h_avg * w_avg, c]

        output.update(img_feat_dict)

        # Auxiliary detection + BEV segmentation. Run when the caller requests it
        # (student's un-rotated "ori" forward during training) OR when the
        # inference-time feasibility filter needs them. Computed BEFORE the
        # trajectory head so the detected geometry can gate candidate selection.
        # The BEV grid [bs, C, bev_h, bev_w] is in the (un-rotated) ego frame the
        # GT raster is aligned to over point_cloud_range.
        # The gate now runs during training too, so refinement is trained on the
        # same candidate distribution it sees at inference. Gating only at
        # inference leaves a train/test mismatch: the refinement stage learns from
        # an unfiltered top-k and is then handed a filtered one.
        #
        # The mask is a hard boolean with no gradient, so this changes WHICH
        # candidates reach refinement, not how anything is differentiated.
        # Enable it only once detection and segmentation are good enough to trust
        # -- an early-epoch gate throws away good candidates on bad predictions.
        want_filter = self._config.feasibility_enabled
        if self.use_aux_heads and (features.get('compute_aux', False) or want_filter):
            bev_map = img_feat_dict['patch_token']  # [bs, C, bev_h, bev_w]
            # aux_detach_bev: cut the perception losses off from the SHARED trunk,
            # so they train the detection and segmentation heads only and the BEV
            # features are shaped by the planning objective alone.
            #
            # This is the stage-3 question in isolated form. Measured on v1,
            # unfreezing perception at stage 3 costs mAP 0.4524 -> 0.4139 and seg
            # IoU 0.8475 -> 0.8097 while every PDM sub-loss improves (DAC -20%,
            # LK -14%, NC -7%) -- the planner reshapes the shared features for
            # itself. Scaling the backbone lr damps BOTH sides of that at once;
            # detaching separates them, leaving the planner free while the heads
            # re-fit whatever trunk it produces.
            #
            # Off by default: this changes what gradients exist, so every earlier
            # run stays reproducible.
            if getattr(self._config, 'aux_detach_bev', False):
                bev_map = bev_map.detach()
            if self._dual_seg:
                # share the ConvNeXt-V2 trunk between the two classifiers
                seg_feat = self._seg_trunk(bev_map)
                output['bev_seg_map'] = self._bev_seg_head[-1](seg_feat)
                output['bev_drivable_map'] = self._bev_drivable_head(seg_feat)
            else:
                output['bev_seg_map'] = self._bev_seg_head(bev_map)  # [bs, n_cls, bev_h, bev_w]
            if self._aux_agent_head_type == "transfuser":
                # this head reads keyval rather than the BEV map, so it needs the
                # same cut to keep aux_detach_bev meaning one thing
                kv = keyval.detach() if getattr(self._config, 'aux_detach_bev', False) else keyval
                agent_query = self._agent_query_embedding.weight[None].repeat(kv.shape[0], 1, 1)
                agent_feat = self._agent_tf_decoder(agent_query, kv)  # [bs, num_bb, c]
                output.update(self._agent_head(agent_feat))  # agent_states, agent_labels
            else:
                agent_feat = None
                output.update(self._agent_head(bev_map))  # agent_states, agent_labels
            if getattr(self, 'predict_agents', False):
                if agent_feat is None:
                    raise NotImplementedError("aux_predict_agents is only implemented for aux_agent_head_type='transfuser'")
                B_, N_ = agent_feat.shape[:2]
                output['agent_future'] = self._agent_prediction_head(agent_feat).view(
                    B_, N_, self._agent_pred_num_poses, 2)  # [bs, num_bb, T, 2] ego-frame xy

        # Optional inference-time candidate feasibility gate (drivable + collision).
        feasibility_mask = None
        if want_filter and 'bev_seg_map' in output:
            feasibility_mask = self._build_feasibility_mask(
                output, keyval.shape[0], keyval.device)
            # Export it. The gate is applied inside the trajectory head, but
            # without this the mask never leaves forward(), so any consumer --
            # the challenge driver's debug payload, and the video overlay that
            # reads it -- falls back to an all-ones mask and paints every one of
            # the 4,096 candidates as a survivor.
            output['feasibility_mask'] = feasibility_mask

        trajectory = self._trajectory_head(
            keyval, status_encoding, tokens=tokens, feasibility_mask=feasibility_mask,
            route=route, route_mask=route_mask)

        if self.use_multi_stage:
            img_feat = img_feat_dict['patch_token']  # [bs, c_vit, w, h]
            final_traj = self._trajectory_offset_head(
                img_feat, trajectory['refinement'], route=route, route_mask=route_mask)
            trajectory['final_traj'] = final_traj

        output.update(trajectory)

        return output

    @torch.no_grad()
    def _build_feasibility_mask(self, output, B, device):
        """Boolean [B, vocab] mask (True = keep) from the aux heads' drivable-area
        seg + agent detection, used to hard-gate candidate selection at inference.
        See DriveSuprimConfig.feasibility_* for the knobs and v1 approximations.
        """
        cfg = self._config
        vocab = self._trajectory_head.vocab  # [V, P, 3] ego frame (x fwd, y left)
        V, P = vocab.shape[0], vocab.shape[1]

        # Ego footprint rectangle (4 corners + face axes) at every vocab pose, plus
        # each corner's BEV pixel. All fixed given the (constant) vocab + ego box,
        # so cache once. Reused by BOTH the drivable and collision gates.
        s, mrg = cfg.feasibility_collision_scale, cfg.feasibility_collision_margin
        # The CV settings belong in the key: they decide how many poses the
        # collision test uses, and a stale cache silently reuses the previous
        # variant's geometry when the config is toggled between calls.
        # The pose -> box-centre offset moves every corner, so it belongs in the
        # key alongside the extents.
        dx = float(getattr(cfg, 'feasibility_ego_center_dx_m', 0.0))
        key = (s, mrg, cfg.feasibility_ego_length, cfg.feasibility_ego_width, dx,
               getattr(cfg, 'feasibility_predict_agents_cv', False),
               getattr(cfg, 'feasibility_predict_horizon', 0.0))
        if getattr(self, '_ego_key', None) != key or getattr(self, '_ego_corners', None) is None \
                or self._ego_corners.device != device:
            x_min, y_min, _, x_max, y_max, _ = cfg.point_cloud_range
            Hh, Ww = cfg.bev_h, cfg.bev_w
            xy = vocab[..., :2].to(device).float()               # [V,P,2]
            th = vocab[..., 2].to(device).float()                # [V,P]
            ehl = torch.tensor(0.5 * cfg.feasibility_ego_length * s + mrg, device=device)
            ehw = torch.tensor(0.5 * cfg.feasibility_ego_width * s + mrg, device=device)
            # A pose is the rig origin, not the box centre, so slide the
            # rectangle forward along the pose heading to sit on the real
            # vehicle. Forward only: across 100 curated_val clips the rig sits
            # on the vehicle centreline every time (lateral offset exactly 0)
            # and the rig->box rotation is always the identity, which AlpaSim
            # enforces by refusing a rotated rig outright. The driver rejects
            # the offset if either assumption ever breaks. dx = 0 reproduces the
            # pose-centred placement the checkpoint was trained with, and the
            # axes below are orientation only, so translation leaves them alone.
            cx, cy = xy[..., 0], xy[..., 1]
            if dx:
                cx = cx + dx * th.cos()
                cy = cy + dx * th.sin()
            ego_c = self._obb_corners(cx, cy, th, ehl, ehw)   # [V,P,4,2]
            self._ego_corners = ego_c
            self._ego_axes = torch.stack([
                torch.stack([th.cos(), th.sin()], -1),
                torch.stack([-th.sin(), th.cos()], -1)], dim=-2)  # [V,P,2,2]
            # BEV pixel of each ego corner (same mapping as _coords_to_pixel)
            ccol = ((ego_c[..., 0] - x_min) / (x_max - x_min) * Ww).long()     # [V,P,4]
            crow = ((ego_c[..., 1] - y_min) / (y_max - y_min) * Hh).long()
            self._corner_in_range = (ccol >= 0) & (ccol < Ww) & (crow >= 0) & (crow < Hh)  # [V,P,4]
            self._corner_row = crow.clamp(0, Hh - 1)
            self._corner_col = ccol.clamp(0, Ww - 1)
            self._vocab_xy = xy
            self._ego_key = key

        feasible = torch.ones(B, V, dtype=torch.bool, device=device)

        # --- drivable-area gate (matches NAVSIM DAC: ego bounding-box CORNER-based;
        #     a pose is off-road if ANY in-range ego corner is off the drivable seg) ---
        if cfg.feasibility_drivable and 'bev_seg_map' in output:
            driv = output['bev_seg_map'].float().softmax(1)[:, 1]        # [B,H,W] P(drivable)
            ds = max(1, int(getattr(cfg, 'feasibility_drivable_pose_stride', 1)))
            crow, ccol = self._corner_row[:, ::ds], self._corner_col[:, ::ds]
            cin = self._corner_in_range[:, ::ds]
            Pd = crow.shape[1]
            r = crow.reshape(-1)                                              # [V*Pd*4]
            c = ccol.reshape(-1)
            pc = driv[:, r, c].view(B, V, Pd, 4)                              # [B,V,Pd,4]
            off = (pc < cfg.feasibility_drivable_thresh) & cin.view(1, V, Pd, 4)
            drivable_ok = ~off.any(dim=3).any(dim=2)                          # any corner, any pose
            output['feasibility_drivable_mask'] = drivable_ok
            feasible &= drivable_ok

        # --- collision gate: ego footprint rectangle vs agent box (OBB-OBB via the
        #     Separating Axis Theorem), both slightly scaled up. Any pose whose ego
        #     rectangle overlaps a detected agent box -> drop the candidate.
        if cfg.feasibility_collision and 'agent_states' in output:
            st = output['agent_states'].float()
            # Two box layouts reach this point and they are NOT interchangeable:
            #   5-dim (legacy): [x, y, heading, length, width]
            #   10-dim (aux_agent_box_3d, what BEVFormer uses and what this repo
            #           trains): [cx, cy, log l, log w, cz, log h, sin, cos, vx, vy]
            # Reading the 10-dim layout with 5-dim indices takes log(l) as the
            # heading and leaves the extents in log space, so a 4.5 m car becomes a
            # 1.5 m box pointing in an arbitrary direction -- the gate then barely
            # catches anything.
            if getattr(cfg, 'aux_agent_box_3d', False):
                a_cx, a_cy = st[..., 0], st[..., 1]
                a_len, a_wid = st[..., 2].exp(), st[..., 3].exp()
                a_yaw = torch.atan2(st[..., 6], st[..., 7])
                a_vx, a_vy = st[..., 8], st[..., 9]
            else:
                a_cx, a_cy = st[..., 0], st[..., 1]
                a_yaw = st[..., 2]
                a_len, a_wid = st[..., 3].abs(), st[..., 4].abs()
                a_vx = a_vy = None                       # 5-dim layout has no velocity

            # Constant-velocity propagation: agents move along the horizon instead
            # of standing still for the whole 4 s. Only the first `P_cv` poses are
            # tested -- past the prediction horizon a CV guess is not worth acting
            # on, so those poses are left out of the collision test entirely
            # (the drivable gate still covers them).
            dt = float(self._config.trajectory_sampling.interval_length)
            horizon = float(getattr(cfg, 'feasibility_predict_horizon', 0.0))
            use_cv = (getattr(cfg, 'feasibility_predict_agents_cv', False)
                      and a_vx is not None and horizon > 0)
            P_cv = min(P, max(1, int(round(horizon / dt)))) if use_cv else P
            t_pose = torch.arange(P_cv, device=device, dtype=torch.float32) * dt   # [P_cv]
            # The BEVFormer-style head emits per-class logits [B,N,C]; collapse to
            # objectness with a max over classes so this stays [B,N] for both heads.
            _lab = output['agent_labels'].float()
            if _lab.dim() == 3:
                _cls = _lab.argmax(-1)                                  # [B,N] predicted class
                _lab = _lab.max(-1).values
            else:
                _cls = None
            valid = _lab.sigmoid() > cfg.feasibility_agent_conf_thresh  # [B,N]

            # Per-class collision margin, keyed by position in aux_agent_classes;
            # entries naming a class the head does not predict are ignored. Only
            # pedestrian and bicycle are widened -- they alone are both scored at
            # full weight (PDM zeroes NC for AGENT_TYPES, halves it to 0.5 for
            # static types) and free enough to leave their own box inside the
            # horizon. Falls back to
            # the global scale for any class not listed (and for the
            # single-class head, which has no class).
            per_cls = getattr(cfg, 'feasibility_class_scale', None) or {}
            if _cls is not None and per_cls:
                lut = torch.tensor(
                    [float(per_cls.get(name, s)) for name in cfg.aux_agent_classes],
                    device=device, dtype=torch.float32)
                a_scale = lut[_cls]                                     # [B,N]
            else:
                a_scale = torch.full_like(_lab, float(s))
            ego_c, ego_ax = self._ego_corners, self._ego_axes          # [V,P,4,2], [V,P,2,2] (cached above)
            egc = ego_c.unsqueeze(0)                                    # [1,V,P,4,2]
            use_future = getattr(self, 'predict_agents', False) and ('agent_future' in output)
            if use_future:
                fut = output['agent_future'].float()                   # [B,N,T,2]
                T = fut.shape[2]
                sel = (torch.arange(P, device=device).float() * T / max(P, 1)).long().clamp(0, T - 1)
            N = st.shape[1]
            # Only the poses inside the prediction horizon take part.
            # slice into NEW names: ego_c/ego_ax alias the cached tensors, and
            # rebinding them here previously left the cache holding a truncated view.
            ego_cP, ego_axP = ego_c[:, :P_cv], ego_ax[:, :P_cv]

            # Only test candidates the drivable gate has not already killed. The
            # two gates are ANDed, so a candidate that is already off-road cannot
            # change the result no matter what the collision test says -- and the
            # SAT loop is ~99% of this function's cost while the drivable lookup
            # is ~0.1%. Ordering drivable first and subsetting here is exact, not
            # an approximation.
            #
            # Across a batch the survivors differ per sample, so the UNION is
            # taken: a candidate is skipped only when every sample has already
            # rejected it. At inference (B=1) that is just the survivors.
            alive = feasible.any(dim=0)                              # [V]
            n_alive = int(alive.sum())
            if n_alive == 0:
                return feasible
            sub = n_alive < V
            if sub:
                aidx = torch.nonzero(alive, as_tuple=False).squeeze(-1)   # [Vs]
                ego_cP, ego_axP = ego_cP[aidx], ego_axP[aidx]
            Veff = ego_cP.shape[0]
            egc = ego_cP.unsqueeze(0)
            collide = torch.zeros(B, Veff, dtype=torch.bool, device=device)

            # Ego corners projected onto the ego's OWN axes. This depends only on
            # ego geometry, not on any agent, yet it used to sit inside the
            # per-agent loop and was recomputed once per detection -- a [Veff,
            # P_cv, 4] reduction times 2 axes times N agents. Hoisted here it runs
            # once. (The agent-axis projections below genuinely do depend on n.)
            ego_ext = []
            for k in range(2):
                axk = ego_axP[..., k, :]                               # [Veff,Pc,2]
                ep = (ego_cP * axk[..., None, :]).sum(-1)              # [Veff,Pc,4]
                ego_ext.append((axk, ep.amin(-1)[None], ep.amax(-1)[None]))

            # Radius of the whole ego swept set, measured on the CORNERS so the
            # vehicle's own extent is already included. Anything farther from the
            # origin than this (minus the agent's own half-diagonal) cannot touch
            # any pose of any surviving candidate, so the SAT can be skipped for
            # it outright. Conservative: it can only skip agents that were going
            # to miss, never one that would have hit.
            ego_reach = ego_cP.reshape(-1, 2).norm(dim=-1).amax()      # scalar
            # Agents are processed in CHUNKS rather than one Python iteration each.
            # A scene here averages 34 detections and peaks at 146; per-agent the
            # loop launched ~20 small CUDA kernels apiece, so launch overhead
            # dominated the arithmetic. Chunking cuts the iteration count by
            # `feasibility_agent_chunk` at the cost of holding
            # [B, C, Veff, P_cv, 4] floats -- ~15 MB at C=16 -- which is why the
            # chunk size is a knob rather than "all agents at once".
            CH = max(1, int(getattr(cfg, 'feasibility_agent_chunk', 16)))

            # Distance cull, vectorised over every agent at once (it used to be a
            # scalar test inside the loop). Conservative: an agent survives if it
            # could reach the ego swept set at ANY pose.
            ahl_all = 0.5 * a_len * a_scale + mrg                       # [B,N]
            ahw_all = 0.5 * a_wid * a_scale + mrg
            a_half_all = torch.sqrt(ahl_all ** 2 + ahw_all ** 2)
            if use_cv and a_vx is not None:
                _px = a_cx[..., None] + a_vx[..., None] * t_pose
                _py = a_cy[..., None] + a_vy[..., None] * t_pose        # [B,N,Pc]
                d_near = torch.sqrt(_px ** 2 + _py ** 2).amin(-1)       # [B,N]
            else:
                d_near = torch.sqrt(a_cx ** 2 + a_cy ** 2)
            near = (d_near - a_half_all) <= ego_reach                   # [B,N]
            nidx = torch.nonzero((valid & near).any(dim=0), as_tuple=False).squeeze(-1)

            for c0 in range(0, int(nidx.numel()), CH):
                idx = nidx[c0:c0 + CH]                                  # [C]
                C = int(idx.numel())
                cx, cy = a_cx[:, idx], a_cy[:, idx]                     # [B,C]
                ah = a_yaw[:, idx]
                ahl, ahw = ahl_all[:, idx], ahw_all[:, idx]
                if use_future:
                    ctr = fut[:, idx][:, :, sel[:P_cv], :]              # [B,C,Pc,2]
                elif use_cv:
                    ctr = torch.stack([cx[..., None] + a_vx[:, idx, None] * t_pose,
                                       cy[..., None] + a_vy[:, idx, None] * t_pose], -1)
                else:
                    ctr = torch.stack([cx, cy], -1)[:, :, None, :]      # [B,C,1,2]
                Pa = ctr.shape[2]
                ag_c = self._obb_corners(
                    ctr[..., 0], ctr[..., 1],
                    ah[..., None].expand(B, C, Pa),
                    ahl[..., None].expand(B, C, Pa),
                    ahw[..., None].expand(B, C, Pa))                    # [B,C,Pa,4,2]
                ag_c = ag_c[:, :, None]                                 # [B,C,1,Pa,4,2]
                cos_a, sin_a = ah.cos(), ah.sin()
                ag_ax = torch.stack([torch.stack([cos_a, sin_a], -1),
                                     torch.stack([-sin_a, cos_a], -1)], dim=-2)  # [B,C,2,2]

                sep = torch.zeros(B, C, Veff, P_cv, dtype=torch.bool, device=device)
                for axk, emin, emax in ego_ext:                         # ego face normals
                    a_ = (ag_c * axk[None, None, :, :, None, :]).sum(-1)   # [B,C,Veff,Pa,4]
                    sep |= (emax[:, None] < a_.amin(-1)) | (a_.amax(-1) < emin[:, None])
                for k in range(2):                                      # agent face normals
                    axk = ag_ax[:, :, k, :][:, :, None, None, None, :]  # [B,C,1,1,1,2]
                    e_ = (egc[:, None] * axk).sum(-1)                   # [B,C,Veff,P_cv,4]
                    a_ = (ag_c * axk).sum(-1)                           # [B,C,Veff,Pa,4]
                    sep |= (e_.amax(-1) < a_.amin(-1)) | (a_.amax(-1) < e_.amin(-1))
                hit = (~sep) & valid[:, idx][:, :, None, None]          # [B,C,Veff,P_cv]
                collide |= hit.any(dim=3).any(dim=1)                    # -> [B,Veff]

            if sub:
                full = torch.zeros(B, V, dtype=torch.bool, device=device)
                full[:, aidx] = collide
                collide = full
            output['feasibility_collision_mask'] = ~collide
            feasible &= ~collide

        return feasible

    @staticmethod
    def _obb_corners(cx, cy, th, hl, hw):
        """World-frame corners of oriented rectangle(s). cx, cy, th, hl, hw all
        broadcast to a common shape [...]; returns [..., 4, 2]. hl is the
        half-extent along the heading (length) axis, hw along the lateral (width)
        axis. Corner order: front-left, front-right, rear-right, rear-left."""
        cos, sin = torch.cos(th), torch.sin(th)
        sx = torch.tensor([1., 1., -1., -1.], device=cx.device)   # length-axis signs
        sy = torch.tensor([1., -1., -1., 1.], device=cx.device)   # width-axis signs
        lx = sx * hl[..., None]                                    # [..., 4]
        ly = sy * hw[..., None]
        wx = cx[..., None] + cos[..., None] * lx - sin[..., None] * ly
        wy = cy[..., None] + sin[..., None] * lx + cos[..., None] * ly
        return torch.stack([wx, wy], dim=-1)



class RouteEncoder(nn.Module):
    """Encode the 20-slot rig-frame route polyline into a single token.

    The message is a fixed ruler, not an arbitrary point set. route_contract
    spaces the slots ROUTE_SPACING_M (80/19 = 4.21 m) apart along the arclength
    and compacts the survivors of the near cutoff to the front, so slot k always
    means the same distance and the valid slots are always a prefix. Measured
    over 16,385 frames: prefix 100.00%, spacing p10 = p50 = p90 = 4.2 m, slot 0
    at x p50 = 42.1 m.

    That is why this flattens rather than pools. Pooling (a masked mean and max
    over a shared per-point MLP) buys permutation invariance and robustness to a
    varying valid count -- neither of which the contract needs -- and pays for it
    by reducing the whole polyline to two statistics. Flattening keeps the curve:
    order, per-slot geometry, curvature. Each slot gets its own weights, which is
    well defined precisely because slot k has a fixed meaning.

    The mask rides along as its own `slots` dims instead of multiplying the
    coordinates. Padding is already zeroed upstream, so it contributes nothing to
    a Linear on its own; what the flags add is the route's LENGTH, which the
    encoder could not otherwise recover -- a Linear cannot count zeros. As a
    prefix code the flags let the first Linear read any function of that length
    (weights sum to f(k) = sum of w_i for i < k), not just a linear one. A frame
    with no route at all (about 0.5% of NuRec) arrives as all-zero coordinates
    AND an all-zero mask, which is distinguishable from a valid route.

    See [[docs/nurec/SHAPES.md]] for the measurements behind these claims.
    """

    def __init__(self, d_model: int, d_hidden: int = 64, slots: int = ROUTE_SLOTS):
        super().__init__()
        self.slots = slots
        # slots*2 coordinates + slots mask flags.
        self.mlp = nn.Sequential(
            nn.Linear(slots * 3, 2 * d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(2 * d_hidden, d_model),
        )

    def forward(self, route: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """route: [bs, slots, 2] (already normalised, padding zeroed); mask: [bs, slots]."""
        if route.shape[1] != self.slots or mask.shape[1] != self.slots:
            # Flattening binds the slot count into the weights, so a Runtime that
            # changes ROUTE_SLOTS must retrain rather than silently misalign.
            raise ValueError(
                f"RouteEncoder was built for {self.slots} slots but got "
                f"route {tuple(route.shape)} / mask {tuple(mask.shape)}"
            )
        return self.mlp(torch.cat([route.flatten(1), mask], dim=-1))


class RouteTokens(nn.Module):
    """Route waypoints as extra memory tokens for the trajectory decoder.

    The trajectory tokens already cross-attend to the image; adding the route to
    the same memory lets each candidate consult the waypoints on its own terms,
    instead of every candidate receiving one pooled summary.

    Waypoints are embedded from their coordinates, never from their slot index,
    so a change to the Runtime's near cutoff or valid-slot count leaves the
    learned embedding meaningful. Padding is dropped by the decoder's
    memory_key_padding_mask rather than zeroed here.
    """

    def __init__(self, d_model: int, d_hidden: int = 64, num_bands: int = 6):
        super().__init__()
        # num_bands = 0 keeps the raw (x, y) pair, for the ablation.
        # Fourier features before the MLP, NOT because adjacent waypoints ought to
        # look dissimilar -- 4.21 m apart on a mostly straight route, they are
        # genuinely alike and a representation saying so is honest. The problem is
        # narrower. Raw coordinates through an MLP put the 20 keys on very nearly
        # one straight line in feature space (measured: PC1 explains 97.8% of
        # their variance). Collinear keys make k_i = k_mean + t_i * u, so the
        # attention logit q . k_i is AFFINE in the slot index -- and an affine
        # function is maximised at an endpoint, never in the middle. Softmax over
        # it can only ramp up or down. Measured on random queries, the argmax lands
        # on slot 0 or slot 9 for 86.6% of them.
        #
        # That is fatal for the one operation these tokens exist to support:
        # "read the route near me". 41.5% of the vocab reaches past the route's
        # first waypoint and every candidate sits somewhere different, so the
        # useful readout peaks at a MIDDLE waypoint. Peaking in the middle needs
        # the logit to be concave in the index, which needs the key curve to bend.
        #
        # Sinusoids at exponentially spaced frequencies bend it. The same 0.0526
        # normalised step between adjacent slots moves sin(2^0 x) by 0.05 rad and
        # sin(2^5 x) by 1.68 rad; the low bands carry roughly-where, the high bands
        # which-one, and winding the input line through the space supplies the
        # curvature. Six bands is the right count for this spacing: 1.68 rad stays
        # under pi, so the top band does not alias adjacent waypoints onto each
        # other. Result: PC1 drops to 48.1% and the argmax spreads to 25.8% on the
        # ends against 20% for uniform.
        #
        # NOT navsim.agents.utils.nerf.nerf_positional_encoding, which computes the
        # identical thing but loops in Python over bands and functions -- twelve
        # kernel launches on a [bs, 20, 2] tensor, 1541 us against 38 us for the
        # broadcast below. That helper is imported at the top of this module and
        # never called; this is why.
        self.num_bands = num_bands
        self.register_buffer(
            "_bands", 2.0 ** torch.arange(max(num_bands, 1), dtype=torch.float32),
            persistent=False,      # a constant, and keeping it out of state_dict
        )                          # leaves checkpoint keys untouched
        in_dim = 2 * 2 * num_bands if num_bands > 0 else 2
        self.embed = nn.Sequential(
            nn.Linear(in_dim, d_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(d_hidden, d_model),
        )
        # Lets the decoder tell a route token from an image token.
        self.type_embedding = nn.Parameter(torch.zeros(1, 1, d_model))

    def forward(self, route: torch.Tensor) -> torch.Tensor:
        """route: [bs, slots, 2], normalised, padding already zeroed."""
        if self.num_bands == 0:
            return self.embed(route) + self.type_embedding
        scaled = route.unsqueeze(-1) * self._bands.to(route.dtype)   # [bs, slots, 2, bands]
        encoded = torch.cat([scaled.sin(), scaled.cos()], dim=-1).flatten(-2)
        return self.embed(encoded) + self.type_embedding


class HydraTrajHead(nn.Module):
    def __init__(self, num_poses: int, d_ffn: int, d_model: int, vocab_path: str,
                 nhead: int, nlayers: int, config: DriveSuprimConfig = None
                 ):
        super().__init__()
        self._config = config
        self._num_poses = num_poses
        # The route gets a cross-attention of its own rather than a seat in the
        # image memory; see RouteAwareDecoderLayer for why. Built only when the
        # route is on, so a route-free config keeps the stock decoder and does
        # not carry the branch's parameters.
        _route_on = (config is not None and getattr(config, "use_route", False)
                     and getattr(config, "route_separate_attention", False))
        _decoder_cls = RouteAwareDecoder if _route_on else nn.TransformerDecoder
        _layer_cls = RouteAwareDecoderLayer if _route_on else nn.TransformerDecoderLayer
        _layer_kwargs = ({"gate_init": getattr(config, "route_gate_init", 0.0)}
                         if _route_on else {})
        self._route_separate = _route_on
        self.transformer = _decoder_cls(
            _layer_cls(
                d_model, nhead, d_ffn,
                dropout=0.0, batch_first=True, **_layer_kwargs
            ), nlayers
        )
        self.vocab = nn.Parameter(
            torch.from_numpy(np.load(vocab_path)),
            requires_grad=False
        )

        self.heads = _sub_score_heads(config, d_model, d_ffn, with_imi=True)

        self.normalize_vocab_pos = config.normalize_vocab_pos
        if self.normalize_vocab_pos:
            self.encoder = MemoryEffTransformer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.0
            )

        self.pos_embed = nn.Sequential(
            nn.Linear(num_poses * 3, d_ffn),
            nn.ReLU(),
            nn.Linear(d_ffn, d_model),
        )

        self._route_tokens = (
            RouteTokens(d_model, config.route_hidden_dim,
                        getattr(config, 'route_fourier_bands', 6))
            if config is not None and getattr(config, "use_route", False)
            else None
        )

    def forward(self, img_feature, status_encoding, tokens=None, feasibility_mask=None,
                route=None, route_mask=None) -> Dict[str, torch.Tensor]:

        vocab = self.vocab.data  # [n_vocab, 40, 3]
        L, HORIZON, _ = vocab.shape
        B = img_feature.shape[0]

        if self.normalize_vocab_pos:
            embedded_vocab = self.pos_embed(vocab.view(L, -1))[None]  # [1, n_vocab, c]
            embedded_vocab = self.encoder(embedded_vocab).repeat(B, 1, 1)  # [bs, n_vocab, c]
        else:
            embedded_vocab = self.pos_embed(vocab.view(L, -1))[None].repeat(B, 1, 1)

        # The image memory stays exactly what it was; the route travels beside it
        # to its own attention instead of being concatenated on. True marks a
        # slot to ignore, and a frame with no route at all is handled inside
        # RouteAwareDecoderLayer rather than by leaning on the image tokens to
        # keep the row non-empty.
        memory, memory_key_padding_mask, route_kwargs = img_feature, None, {}
        if self._route_tokens is not None and route is not None and route_mask is not None:
            route_tokens = self._route_tokens(route)
            if self._route_separate:
                route_kwargs = {"route_memory": route_tokens,
                                "route_key_padding_mask": route_mask == 0}
            else:
                # Every candidate may now read the waypoints as well as the image.
                memory = torch.cat([img_feature, route_tokens], dim=1)
                # True marks a position to ignore. Image tokens are always present;
                # route padding is not. A frame with no route at all masks only the
                # route block, so no row is ever fully masked.
                image_positions = torch.zeros(
                    img_feature.shape[:2], dtype=torch.bool, device=img_feature.device
                )
                memory_key_padding_mask = torch.cat(
                    [image_positions, route_mask == 0], dim=1)

        tr_out = self.transformer(
            embedded_vocab, memory,
            memory_key_padding_mask=memory_key_padding_mask, **route_kwargs
        )  # [bs, n_vocab, c]
        dist_status = tr_out + status_encoding.unsqueeze(1)  # [bs, n_vocab, c]
        result = {}
        for k, head in self.heads.items():
            result[k] = head(dist_status).squeeze(-1)

        scores = _rank_imi_term(self._config, result) + _rank_score(
            self._config, result, self._config.use_traffic_light_compliance
        )  # [bs, n_vocab]

        if feasibility_mask is not None:
            # Candidate gate from the aux heads (drivable + collision), applied
            # BEFORE the coarse->fine top-k so refinement only sees candidates
            # that survived it.
            #
            # Demote rather than mask. Filling with -inf makes every rejected
            # candidate compare equal, so when fewer than `topks` survive, topk
            # fills the remaining slots in index order rather than by score. A
            # penalty larger than the score span keeps every feasible candidate
            # ahead of every infeasible one while preserving the score ordering
            # WITHIN each group -- so a short row is topped up with the best of
            # the rejected, not with arbitrary ones. It also removes the need for
            # an all-rejected fallback: a uniformly penalised row still ranks by
            # score, so argmax/top-k stay well-defined.
            # detached: the demotion is a selection rule, not something to
            # differentiate through. Without this the penalty term would put the
            # per-row max/min scores into the graph once the gate runs in training.
            span = (scores.max(dim=1, keepdim=True).values
                    - scores.min(dim=1, keepdim=True).values).detach()
            scores = torch.where(feasibility_mask, scores, scores - (span + 1.0))
        result['scores'] = scores

        selected_indices = scores.argmax(1)
        result["trajectory"] = self.vocab.data[selected_indices]
        result["trajectory_vocab"] = self.vocab.data
        result["selected_indices"] = selected_indices

        if self._config.refinement.use_multi_stage:
            topk_str = str(self._config.refinement.topks)
            topk = int(topk_str.split('+')[0])
            topk_values, topk_indices = torch.topk(scores, k=topk, dim=1)
            result['refinement'] = []  # dicts of different refinement stages
            _dict = {}
            _dict["trajs"] = self.vocab.data[topk_indices]
            # Gather the statuses for the top-k trajectories
            batch_indices = torch.arange(B, device=topk_indices.device).view(-1, 1).expand(-1, topk)
            _dict["trajs_status"] = dist_status[batch_indices, topk_indices]
            _dict['indices_absolute'] = topk_indices

            # Store the scores for each top-k trajectory
            _dict['coarse_score'] = {}
            for score_key in self.heads.keys():
                _dict['coarse_score'][score_key] = result[score_key][batch_indices, topk_indices]

            result['refinement'].append(_dict)
        
        return result


class RefineTrajHead(nn.Module):
    def __init__(self, d_ffn: int, d_model: int, nhead: int, d_backbone: int,
                 num_stage: int, stage_layers: str, topks: str,
                 config: DriveSuprimConfig = None
                 ):
        super().__init__()
        
        stage_layers = str(stage_layers)
        topks = str(topks)
        
        self._config = config
        self.num_stage = num_stage  # the number of **refinement** stages, we choose to use only 1 refinement stage (8192->256)
        self.stage_layers = [int(sl) for sl in stage_layers.split('+')]
        self.topks = [int(topk) for topk in topks.split('+')]
        assert len(self.stage_layers) == num_stage and len(self.topks) == num_stage
        # self.nlayers = sum(self.stage_layers)

        self.use_mid_output = config.refinement.use_mid_output
        self.use_separate_stage_heads = config.refinement.use_separate_stage_heads

        downscale_layer = nn.Conv2d(d_backbone, d_model, kernel_size=1)
        if self.use_separate_stage_heads:
            self.downscale_layers = nn.ModuleList([copy.deepcopy(downscale_layer) for _ in range(num_stage)])
        else:
            self.downscale_layers = nn.ModuleList([downscale_layer for _ in range(num_stage)])

        # Positional embedding on the key/value tokens. See
        # RefinementConfig.keyval_pos_embed for why refinement lacked one.
        # Follows downscale_layers: per-stage when the stages have separate
        # heads, shared otherwise, so the two always agree on how much of the
        # stage machinery is tied together.
        self.keyval_pos_embed = None
        if config is not None and getattr(config.refinement, 'keyval_pos_embed', False):
            # The token grid is whatever feeds `img_feat` in forward(), which is
            # patch_token at FULL resolution -- the first stage's bev_keyval_grid
            # pooling never applies here.
            if config.backbone_type == 'bevformer_m':
                num_tokens = config.bev_h * config.bev_w
            else:
                num_tokens = config.img_vert_anchors * config.img_horz_anchors
            emb = nn.Embedding(num_tokens, d_model)
            if self.use_separate_stage_heads:
                self.keyval_pos_embed = nn.ModuleList(
                    [copy.deepcopy(emb) for _ in range(num_stage)])
            else:
                self.keyval_pos_embed = nn.ModuleList([emb for _ in range(num_stage)])

        # One embedding shared by every stage, unlike keyval_pos_embed above:
        # that one is per-stage because each stage sees its own image grid,
        # whereas a waypoint is the same ego-frame geometry no matter which
        # stage reads it.
        self._route_tokens = (
            RouteTokens(d_model, config.route_hidden_dim,
                        getattr(config, 'route_fourier_bands', 6))
            if config is not None and getattr(config, "use_route", False)
            else None
        )

        # Same split as the first stage. Refinement is where a shared memory hurts
        # most: 3,136 image tokens against 20 route ones.
        _route_on = (config is not None and getattr(config, "use_route", False)
                     and getattr(config, "route_separate_attention", False))
        self._route_separate = _route_on
        transformer_blocks = [
            RouteAwareDecoder(
                RouteAwareDecoderLayer(
                    d_model, nhead, d_ffn, dropout=0.0, batch_first=True,
                    gate_init=getattr(config, "route_gate_init", 0.0),
                ), layer, return_intermediate=True,
            ) if _route_on else TransformerDecoder_v2(
                nn.TransformerDecoderLayer(
                    d_model, nhead, d_ffn,
                    dropout=0.0, batch_first=True
                ), layer
            ) for layer in self.stage_layers]
        self.transformer_blocks = nn.ModuleList(transformer_blocks)

        heads = _sub_score_heads(
            self._config, d_model, d_ffn,
            with_imi=self._config.refinement.use_imi_learning_in_refinement,
        )

        if self.use_separate_stage_heads:
            self.multi_stage_heads = nn.ModuleList([copy.deepcopy(heads) for _ in range(num_stage)])
        else:
            self.multi_stage_heads = nn.ModuleList([heads for _ in range(num_stage)])

        self.normalize_vocab_pos = config.normalize_vocab_pos
        if self.normalize_vocab_pos:
            self.encoder = MemoryEffTransformer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=0.0
            )

    def forward(self, img_feat, refinement_dict, route=None, route_mask=None) -> Dict[str, torch.Tensor]:

        B = img_feat.shape[0]
        
        for i in range(self.num_stage):
            _img_feat_fg = self.downscale_layers[i](img_feat).flatten(2)
            _img_feat_fg = _img_feat_fg.permute(0, 2, 1)  # [bs, h_avg * w_avg, c]
            if self.keyval_pos_embed is not None:
                # Added after the 1x1 projection and before the decoder, matching
                # where the first stage adds _keyval_embedding. Not in-place: the
                # projection output is still needed unmodified by nothing here,
                # but an in-place add on a tensor that autograd tracks is a
                # needless hazard if a later stage ever reuses it.
                _img_feat_fg = _img_feat_fg + self.keyval_pos_embed[i].weight[None]
            # The surviving candidates get the same access to the route the first
            # stage had; without this the refinement would re-rank them on image
            # evidence alone and could undo a route-aware ordering. It rides its
            # own attention here too -- this is the stage where a shared softmax
            # hurt most, 20 route keys against 3,136 image ones.
            memory, memory_key_padding_mask, route_kwargs = _img_feat_fg, None, {}
            if self._route_tokens is not None and route is not None and route_mask is not None:
                route_tokens = self._route_tokens(route)
                if self._route_separate:
                    route_kwargs = {"route_memory": route_tokens,
                                    "route_key_padding_mask": route_mask == 0}
                else:
                    memory = torch.cat([_img_feat_fg, route_tokens], dim=1)
                    image_positions = torch.zeros(
                        _img_feat_fg.shape[:2], dtype=torch.bool,
                        device=_img_feat_fg.device
                    )
                    memory_key_padding_mask = torch.cat(
                        [image_positions, route_mask == 0], dim=1)

            status_encoding = refinement_dict[-1]['trajs_status']  # [bs, topk_stage_i, c]
            tr_out_lst = self.transformer_blocks[i](
                status_encoding, memory,
                memory_key_padding_mask=memory_key_padding_mask, **route_kwargs
            )  # [layer_stage_i, bs, topk_stage_i, c]

            # Compute scores for each refinement decoder layer
            layer_results = []
            for j, dist_status in enumerate(tr_out_lst):
                layer_result = {}
                for k, head in self.multi_stage_heads[i].items():
                    layer_result[k] = head(dist_status).squeeze(-1)
                layer_results.append(layer_result)
            
            if not self.use_mid_output:
                layer_results = layer_results[-1:]
            refinement_dict[-1]['layer_results'] = layer_results
            
            last_layer_result = layer_results[-1]

            scores = _rank_score(
                self._config, last_layer_result,
                self._config.use_traffic_light_compliance,
                stage="refine",
            )

            if self._config.refinement.use_imi_learning_in_refinement:
                imi_term = _rank_imi_term(
                    self._config, last_layer_result, stage="refine")
                # Diagnostic for REFINEMENT_IMI_WEIGHT_PATCH.md section 5: is the
                # imitation term still deciding the refinement argmax on its own?
                # Enabled by env var so it costs nothing in a scoring run.
                if os.environ.get("DRIVESUPRIM_REFINE_DEBUG", "0") not in ("0", "", "false"):
                    with torch.no_grad():
                        total = scores + imi_term
                        # (a) as this codebase actually ranks: _rank_score()'s
                        # output, whatever composition it is configured for.
                        rk = scores
                        # (b) the patch document's section-5 measurement: the raw
                        # sigmoid product of the three heads and the normalised
                        # imi, independent of how _rank_score is composed. The two
                        # disagree when the configured ranking is not the plain
                        # product, so print both rather than assume.
                        prod = None
                        for _k in ("no_at_fault_collisions",
                                   "drivable_area_compliance", "ego_progress"):
                            if _k not in last_layer_result:
                                prod = None
                                break
                            _t = last_layer_result[_k].sigmoid()
                            prod = _t if prod is None else prod * _t
                        imi_n = last_layer_result['imi'].softmax(-1)
                        imi_n = imi_n / imi_n.amax(-1, keepdim=True).clamp_min(1e-12)
                        w = config_imi_rank_weight_stage(self._config, 'refine')
                        doc_total = None if prod is None else prod + w * imi_n
                        def _sp(x):
                            return (x.amax(1) - x.amin(1)).mean().item()
                        print(
                            "[DriveSuprim] REFDBG:"
                            f" w_refine={w:.4f}"
                            f" exp_refine={getattr(self._config, 'pdm_rank_product_exponents_refine', None)}"
                            f" exp_coarse={getattr(self._config, 'pdm_rank_product_exponents', None)}"
                            f" w_coarse={config_imi_rank_weight(self._config):.4f}"
                            f" k={rk.shape[1]}"
                            f" | rank_spread={_sp(rk):.4f}"
                            f" imi_term_spread={_sp(imi_term):.4f}"
                            f" imi_decides={(imi_term.argmax(1) == total.argmax(1)).float().mean().item():.3f}"
                            + ("" if prod is None else
                               f" | doc_product_spread={_sp(prod):.4f}"
                               f" doc_imi_spread={_sp(imi_n):.4f}"
                               f" doc_imi_decides={(imi_n.argmax(1) == doc_total.argmax(1)).float().mean().item():.3f}"
                               f" doc_product_decides={(prod.argmax(1) == doc_total.argmax(1)).float().mean().item():.3f}"),
                            flush=True,
                        )
                scores = scores + imi_term

            if i != self.num_stage-1:
                _next_topk = self.topks[i+1]
                _, select_indices = torch.topk(scores, k=_next_topk, dim=1)
                batch_indices = torch.arange(B, device=select_indices.device).view(-1, 1).expand(-1, _next_topk)

                _next_layer_dict = {}
                _next_layer_dict["trajs"] = refinement_dict[-1]['trajs'][batch_indices, select_indices]
                _next_layer_dict["trajs_status"] = tr_out_lst[-1][batch_indices, select_indices]
                _next_layer_dict['indices_absolute'] = refinement_dict[-1]['indices_absolute'][batch_indices, select_indices]
                refinement_dict.append(_next_layer_dict)
            
            else:
                select_indices = scores.argmax(1)
                batch_indices = torch.arange(B, device=select_indices.device)
                final_traj = refinement_dict[-1]['trajs'][batch_indices, select_indices]  # [bs, 40, 3]
                # filtered_scores = scores
        
        return final_traj


class RouteAwareDecoderLayer(nn.TransformerDecoderLayer):
    """A stock decoder layer plus a cross-attention of its own over the route.

    Why its own and not concatenated into the image memory, which is what this
    replaces: a shared memory means one softmax, and the route has to win
    probability mass away from the image tokens to be heard at all -- 20 keys
    against 784 in the first stage and 3,136 in refinement, where the route is
    0.63% of the keys. Nothing in that arrangement guarantees it is ever read,
    and the pressure not to bother is highest exactly here, fine-tuning from a
    checkpoint that already scores well with no route at all.

    A separate attention makes the question moot. The route's softmax is over
    the route alone, so its 20 waypoints hold the full mass however many image
    tokens there are. This does NOT collapse the waypoints: every proposal to
    fold the curve into one vector was rejected because the useful operation is
    per-candidate -- 4,096 queries reading one curve from 4,096 different places
    (41.5% of the vocab reaches past the route's first waypoint, so the distance
    from a candidate to the route is a real quantity that varies by candidate).
    Twenty keys are what make that readable; one key would return the same value
    to every candidate.

    Gated with tanh(g) and g initialised to zero, applied Flamingo-style to a
    pre-normed branch so the layer is EXACTLY the identity at initialisation.
    A checkpoint trained without a route therefore starts where it left off and
    the branch opens only if it pays for itself. The gate doubles as the
    diagnostic: read tanh(g) per layer after training to see how much route the
    model chose to use.
    """

    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.0,
                 batch_first=True, gate_init: float = 0.0):
        super().__init__(d_model, nhead, dim_feedforward,
                         dropout=dropout, batch_first=batch_first)
        # Submodule names above are inherited unchanged (self_attn, multihead_attn,
        # linear1/2, norm1/2/3) so a pre-route checkpoint still loads; only the
        # three names below are new, and they arrive as missing keys.
        self.route_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=batch_first)
        self.route_norm = nn.LayerNorm(d_model)
        self.route_gate = nn.Parameter(torch.full((1,), float(gate_init)))

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                route_memory=None, route_key_padding_mask=None):
        out = super().forward(
            tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        if route_memory is None:
            return out

        # The concatenated memory could never be fully masked because the image
        # tokens were always present; a route-only softmax has no such floor, and
        # the ~0.5% of NuRec frames with no route at all would divide by nothing.
        # Unmask those rows so the softmax is well defined, then discard what it
        # produced -- masking after the fact would not help, NaN * 0 is NaN.
        if route_key_padding_mask is not None:
            empty = route_key_padding_mask.all(dim=1, keepdim=True)   # [bs, 1]
            safe_mask = route_key_padding_mask & ~empty
        else:
            empty, safe_mask = None, None

        attended, _ = self.route_attn(
            self.route_norm(out), route_memory, route_memory,
            key_padding_mask=safe_mask, need_weights=False,
        )
        if empty is not None:
            attended = attended.masked_fill(empty.unsqueeze(-1), 0.0)
        return out + torch.tanh(self.route_gate) * attended


class RouteAwareDecoder(nn.TransformerDecoder):
    """``nn.TransformerDecoder`` that also hands the route to each layer.

    ``self.layers`` is built by the base class, so parameter names stay
    ``layers.N.<submodule>`` and pre-route checkpoints keep loading.
    ``return_intermediate`` reproduces TransformerDecoder_v2 for refinement,
    which scores every layer rather than only the last.
    """

    def __init__(self, decoder_layer, num_layers, return_intermediate: bool = False):
        super().__init__(decoder_layer, num_layers)
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                route_memory=None, route_key_padding_mask=None):
        output = tgt
        outputs = []
        for mod in self.layers:
            output = mod(
                output, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                route_memory=route_memory,
                route_key_padding_mask=route_key_padding_mask,
            )
            outputs.append(output)
        return outputs if self.return_intermediate else output


class TransformerDecoder_v2(nn.TransformerDecoder):

    def forward(self, tgt, memory, tgt_mask = None, memory_mask = None, tgt_key_padding_mask = None, memory_key_padding_mask = None):
        output = tgt

        output_lst = []

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)
            output_lst.append(output)

        return output_lst
