"""Spatial cross-attention for the BEVFormer-M front-end.

Adapted from the official BEVFormer (Apache-2.0,
https://github.com/fundamentalvision/BEVFormer) to a self-contained form that
does not depend on the mmcv/mmdet config registry. The multi-scale deformable
sampling itself reuses mmcv's pure-PyTorch kernel
(`multi_scale_deformable_attn_pytorch`), which is available in the installed
mmcv 1.7.2.

Simplification vs. the original: we do NOT re-batch queries per camera by
visibility (the original packs only the BEV queries visible to each camera to
save memory). Instead every BEV query is run against every camera and the
per-camera outputs are masked and averaged. This is a little heavier but far
simpler / more robust, which matters while bringing the front-end up. The
numerical result is equivalent because invisible queries are masked out before
averaging.
"""
import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, xavier_init
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch

try:
    # CUDA kernel used by the original BEVFormer: far more memory-efficient than
    # the pure-PyTorch fallback (which materializes per-level grid_sample outputs
    # and OOMs at fp32 / larger batch). Falls back to pytorch on CPU.
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction
    _HAS_CUDA_MSDA = True
except Exception:  # pragma: no cover
    _HAS_CUDA_MSDA = False


def _deform_attn(value, spatial_shapes, level_start_index, sampling_locations,
                 attention_weights, im2col_step=64):
    """Dispatch to the CUDA deformable-attention kernel on GPU, else PyTorch."""
    if _HAS_CUDA_MSDA and value.is_cuda:
        # mmcv's CUDA kernel (ms_deform_attn_cuda.cu) does
        #   im2col_step_ = min(batch, im2col_step)
        #   assert batch % im2col_step_ == 0
        # which only clamps the chunk size DOWN to im2col_step -- it does not
        # search for a divisor. That's harmless while batch <= im2col_step
        # (im2col_step_ becomes batch itself, trivially divisible), but once
        # batch > im2col_step (e.g. a ragged last-of-epoch batch pushes
        # per_gpu_batch_size * num_cams above the default 64) it asserts
        # unless batch happens to be an exact multiple of im2col_step, and
        # crashes with "batch(%d) must divide im2col_step(%d)" otherwise.
        # Batches through this BEV encoder are always small, so there is no
        # real memory benefit to chunking here: fall back to using the whole
        # batch as the chunk size (im2col_step_ = batch, always divisible)
        # exactly in the case where the default would fail, and leave the
        # normal (possibly chunked) path alone otherwise.
        batch = value.shape[0]
        if batch % min(batch, im2col_step) != 0:
            im2col_step = batch
        # mmcv's kernel is compiled for fp32/fp16 only -- bf16 raises
        # "ms_deform_attn_forward_cuda not implemented for 'BFloat16'". Under a
        # bf16 autocast, run just this call in fp32 and hand the result back in
        # the caller's dtype: the surrounding projections, FFN and norms stay
        # bf16, which is where most of the encoder's arithmetic is anyway. The
        # pure-PyTorch fallback would take bf16 but materialises per-level
        # grid_sample outputs and OOMs at this BEV size.
        if value.dtype == torch.bfloat16:
            out = MultiScaleDeformableAttnFunction.apply(
                value.float(), spatial_shapes, level_start_index,
                sampling_locations.float(), attention_weights.float(), im2col_step)
            return out.to(value.dtype)
        return MultiScaleDeformableAttnFunction.apply(
            value, spatial_shapes, level_start_index, sampling_locations,
            attention_weights, im2col_step)
    return multi_scale_deformable_attn_pytorch(
        value, spatial_shapes, sampling_locations, attention_weights)


class MSDeformableAttention3D(nn.Module):
    """Multi-scale deformable attention with 3D (pillar) reference points.

    Each BEV query carries ``num_z_anchors`` reference points (one per height
    sample in its pillar). Sampling offsets are predicted per (head, level,
    point) and added to the projected pillar points before bilinear sampling
    the multi-view image feature maps.
    """

    def __init__(self, embed_dims=256, num_heads=8, num_levels=4, num_points=8,
                 num_z_anchors=4, batch_first=True):
        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f"embed_dims {embed_dims} must be divisible by num_heads {num_heads}")
        if num_points % num_z_anchors != 0:
            raise ValueError(
                f"num_points ({num_points}) must be divisible by num_z_anchors ({num_z_anchors})")

        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.num_z_anchors = num_z_anchors
        self.batch_first = batch_first
        self.head_dims = embed_dims // num_heads

        self.sampling_offsets = nn.Linear(embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        # NOTE: the output projection lives in SpatialCrossAttention, matching
        # the original design (deformable attention returns un-projected feats).
        self.init_weights()

    def init_weights(self):
        constant_init(self.sampling_offsets, 0.0)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1
        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0.0, bias=0.0)
        xavier_init(self.value_proj, distribution="uniform", bias=0.0)

    def forward(self, query, value, reference_points, spatial_shapes, level_start_index):
        """
        :param query: [bs, num_query, C]
        :param value: [bs, num_value, C]
        :param reference_points: [bs, num_query, num_z_anchors, 2] in [0, 1]
        :param spatial_shapes: [num_levels, 2] (H, W) per level
        :param level_start_index: [num_levels]
        :return: [bs, num_query, C]
        """
        bs, num_query, _ = query.shape
        _, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        value = value.view(bs, num_value, self.num_heads, self.head_dims)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points)

        # reference_points: [bs, num_query, num_z_anchors, 2]
        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        num_z_anchors = reference_points.shape[2]
        # [bs, num_query, 1, 1, 1, num_z_anchors, 2]
        reference_points = reference_points[:, :, None, None, None, :, :]
        sampling_offsets = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        # split the num_points axis into (points_per_z, num_z_anchors)
        points_per_z = self.num_points // num_z_anchors
        sampling_offsets = sampling_offsets.view(
            bs, num_query, self.num_heads, self.num_levels, points_per_z, num_z_anchors, 2)
        sampling_locations = reference_points + sampling_offsets
        sampling_locations = sampling_locations.view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)

        output = _deform_attn(
            value, spatial_shapes, level_start_index, sampling_locations, attention_weights)
        return output


class SpatialCrossAttention(nn.Module):
    """BEV queries attend to multi-view image features at projected points."""

    def __init__(self, embed_dims=256, num_cams=3, num_heads=8, num_levels=4,
                 num_points=8, num_z_anchors=4, dropout=0.1):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.dropout = nn.Dropout(dropout)
        self.deformable_attention = MSDeformableAttention3D(
            embed_dims=embed_dims, num_heads=num_heads, num_levels=num_levels,
            num_points=num_points, num_z_anchors=num_z_anchors, batch_first=True)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(self, query, value, reference_points_cam, bev_mask,
                spatial_shapes, level_start_index, query_pos=None, residual=None):
        """
        :param query: [bs, num_query, C]
        :param value: [num_cams, bs, num_value, C]
        :param reference_points_cam: [num_cams, bs, num_query, num_z_anchors, 2]
        :param bev_mask: [num_cams, bs, num_query, num_z_anchors] bool
        :param spatial_shapes: [num_levels, 2]
        :param level_start_index: [num_levels]
        :return: [bs, num_query, C]

        Memory note: like the original BEVFormer, each camera only runs
        deformable attention over the BEV queries that actually project into it
        (``queries_rebatch``), instead of all num_query queries for every
        camera. This greatly reduces peak activation memory. The camera
        calibration is identical across the batch (fixed rig), so the visibility
        indices are taken from batch element 0 and reused for the whole batch,
        matching the reference implementation.
        """
        # Run the whole spatial cross-attention in fp32 (autocast off), matching
        # the original UniAD/BEVFormer @force_fp32 on this module. Under fp16 AMP
        # the projected sampling / deformable attention can overflow fp16's range
        # and produce NaN; fp32 here keeps it numerically stable.
        orig_dtype = query.dtype
        with torch.cuda.amp.autocast(enabled=False):
            out = self._forward_fp32(
                query.float(), value.float(), reference_points_cam.float(), bev_mask,
                spatial_shapes, level_start_index,
                None if query_pos is None else query_pos.float(),
                None if residual is None else residual.float(),
            )
        return out.to(orig_dtype)

    def _forward_fp32(self, query, value, reference_points_cam, bev_mask,
                      spatial_shapes, level_start_index, query_pos, residual):
        if residual is None:
            residual = query
        inp_residual = residual
        slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()
        num_value = value.shape[2]
        D = reference_points_cam.size(3)  # num_z_anchors

        # visible BEV-query indices per camera (from batch elem 0)
        indexes = []
        for i in range(self.num_cams):
            idx = bev_mask[i, 0].sum(-1).nonzero().squeeze(-1)
            indexes.append(idx)
        max_len = max((len(each) for each in indexes), default=0)
        max_len = max(max_len, 1)  # guard against a fully-empty camera

        # pack only the visible queries / reference points per camera
        queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
        ref_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, D, 2])
        for j in range(bs):
            for i in range(self.num_cams):
                idx = indexes[i]
                if len(idx) == 0:
                    continue
                queries_rebatch[j, i, :len(idx)] = query[j, idx]
                ref_rebatch[j, i, :len(idx)] = reference_points_cam[i, j, idx]

        # value: [num_cams, bs, L, C] -> [bs*num_cams, L, C] (bs-major to match queries)
        value_flat = value.permute(1, 0, 2, 3).reshape(bs * self.num_cams, num_value, self.embed_dims)

        queries = self.deformable_attention(
            queries_rebatch.view(bs * self.num_cams, max_len, self.embed_dims),
            value_flat,
            ref_rebatch.view(bs * self.num_cams, max_len, D, 2),
            spatial_shapes, level_start_index,
        ).view(bs, self.num_cams, max_len, self.embed_dims)

        # scatter results back to their query positions and average over cameras
        for j in range(bs):
            for i in range(self.num_cams):
                idx = indexes[i]
                if len(idx) == 0:
                    continue
                slots[j, idx] += queries[j, i, :len(idx)]

        count = (bev_mask.sum(-1) > 0).permute(1, 2, 0).sum(-1)  # [bs, num_query]
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]

        slots = self.output_proj(slots)
        return self.dropout(slots) + inp_residual
