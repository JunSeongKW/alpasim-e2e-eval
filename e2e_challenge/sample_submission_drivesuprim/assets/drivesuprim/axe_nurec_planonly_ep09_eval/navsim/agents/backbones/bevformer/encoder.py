"""BEVFormer encoder: BEV queries, deformable self-attention over the BEV grid,
spatial cross-attention onto multi-view image features, and the projection
utilities that map BEV pillar points into each camera.

Self-contained adaptation of the official BEVFormer encoder (Apache-2.0). The
temporal self-attention of the original is replaced by a plain 2D deformable
self-attention over the current BEV grid; temporal fusion is handled outside
the encoder (concat + conv over frames) as described by SafeDrive.
"""
import math

import torch
import torch.nn as nn
from mmcv.cnn import constant_init, xavier_init
from navsim.agents.backbones.bevformer.spatial_cross_attention import SpatialCrossAttention, _deform_attn


class BEVSelfAttention(nn.Module):
    """2D deformable self-attention over the BEV grid (single level)."""

    def __init__(self, embed_dims=256, num_heads=8, num_points=4, dropout=0.1):
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

    def forward(self, query, bev_pos, reference_points, spatial_shapes, level_start_index):
        """
        :param query: [bs, num_query, C]
        :param bev_pos: [bs, num_query, C]
        :param reference_points: [bs, num_query, 2] normalized grid location
        :param spatial_shapes: [1, 2] the (H, W) of the BEV grid
        :param level_start_index: [1]
        """
        bs, num_query, _ = query.shape
        residual = query
        value = self.value_proj(query).view(bs, num_query, self.num_heads, self.head_dims)
        q = query + bev_pos

        sampling_offsets = self.sampling_offsets(q).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(q).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points)

        offset_normalizer = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
        sampling_locations = (
            reference_points[:, :, None, None, None, :]
            + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        )
        out = _deform_attn(
            value, spatial_shapes, level_start_index, sampling_locations, attention_weights)
        out = self.output_proj(out)
        return self.dropout(out) + residual


class BEVFormerLayer(nn.Module):
    """One encoder layer: deformable self-attn -> spatial cross-attn -> FFN."""

    def __init__(self, embed_dims=256, num_heads=8, num_cams=3, num_levels=4,
                 feedforward_channels=512, sca_num_points=8, ssa_num_points=4,
                 num_z_anchors=4, dropout=0.1):
        super().__init__()
        self.self_attn = BEVSelfAttention(
            embed_dims, num_heads=num_heads, num_points=ssa_num_points, dropout=dropout)
        self.cross_attn = SpatialCrossAttention(
            embed_dims=embed_dims, num_cams=num_cams, num_heads=num_heads,
            num_levels=num_levels, num_points=sca_num_points,
            num_z_anchors=num_z_anchors, dropout=dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dims, feedforward_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feedforward_channels, embed_dims),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

    def forward(self, query, value, bev_pos, ref_2d, ref_cam, bev_mask,
                bev_spatial_shapes, bev_level_start_index,
                img_spatial_shapes, img_level_start_index):
        query = self.self_attn(query, bev_pos, ref_2d, bev_spatial_shapes, bev_level_start_index)
        query = self.norm1(query)
        query = self.cross_attn(
            query, value, ref_cam, bev_mask, img_spatial_shapes, img_level_start_index,
            query_pos=bev_pos)
        query = self.norm2(query)
        query = query + self.ffn(query)
        query = self.norm3(query)
        return query


class BEVFormerEncoder(nn.Module):
    """Stacks BEVFormer layers and owns the reference-point / projection logic."""

    def __init__(self, num_layers=3, embed_dims=256, num_heads=8, num_cams=3,
                 num_levels=4, feedforward_channels=512, num_z_anchors=4,
                 sca_num_points=8, ssa_num_points=4, point_cloud_range=None,
                 dropout=0.1, pillar_z_range=None):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dims = embed_dims
        self.num_z_anchors = num_z_anchors
        self.point_cloud_range = point_cloud_range
        # Vertical extent the pillar anchors span. Defaults to point_cloud_range's
        # z, which is what BEVFormer does, but the two answer different questions
        # and the best value differs:
        #
        #   pillars    decide where in the IMAGE each BEV cell samples. Anchors
        #              outside the objects' height band project onto sky or road,
        #              so this wants to be TIGHT around real object heights.
        #   pc_range's z additionally denormalises the detection head's sigmoid
        #              box z (drivesuprim_model.py), so it wants to be WIDE
        #              enough that no object centre is unrepresentable and none
        #              sits where sigmoid saturates.
        #
        # Measured on navtest by projecting the anchors through lidar2img and
        # asking whether they land inside each object's 2D extent: over -3..5
        # only 0.70 of 4 anchors hit, and 31.1% of objects get zero. Over -1..2
        # that becomes 2.17 and 5.4%. Meanwhile navtrain box-centre z has p99
        # 2.08 and a max of 4.99, so -1..2 would leave 1.7% of centres outside
        # the regression range entirely -- hence separating them rather than
        # picking one compromise.
        self.pillar_z_range = tuple(pillar_z_range) if pillar_z_range is not None \
            else (point_cloud_range[2], point_cloud_range[5])
        self.layers = nn.ModuleList([
            BEVFormerLayer(
                embed_dims=embed_dims, num_heads=num_heads, num_cams=num_cams,
                num_levels=num_levels, feedforward_channels=feedforward_channels,
                sca_num_points=sca_num_points, ssa_num_points=ssa_num_points,
                num_z_anchors=num_z_anchors, dropout=dropout)
            for _ in range(num_layers)
        ])

    @staticmethod
    def get_reference_points_3d(H, W, num_z, z_range, bs, device, dtype):
        """Pillar reference points normalized to [0, 1] in (x, y, z).

        The z anchors follow the original BEVFormer formula
        ``linspace(0.5, Z - 0.5, D) / Z`` where Z is the METRIC height of the
        pillar (pc_range[5] - pc_range[2]), not the number of anchors. With
        Z=8m and D=4 that samples z = [-2.5, -0.17, 2.17, 4.5]; normalising by D
        instead would compress the anchors to [-2, 0, 2, 4] and never sample
        near the top/bottom of the pillar.

        :return: [bs, num_z, H*W, 3]
        """
        zs = (torch.linspace(0.5, z_range - 0.5, num_z, dtype=dtype, device=device)
              / z_range).view(-1, 1, 1).expand(num_z, H, W)
        xs = (torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
              / W).view(1, 1, W).expand(num_z, H, W)
        ys = (torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device)
              / H).view(1, H, 1).expand(num_z, H, W)
        ref = torch.stack((xs, ys, zs), -1)  # [num_z, H, W, 3]
        ref = ref.reshape(num_z, H * W, 3)
        return ref[None].repeat(bs, 1, 1, 1)  # [bs, num_z, H*W, 3]

    @staticmethod
    def get_reference_points_2d(H, W, bs, device, dtype):
        """BEV-grid reference points normalized to [0, 1] in (x, y).

        :return: [bs, H*W, 2]
        """
        ys = (torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device)
              / H).view(H, 1).expand(H, W)
        xs = (torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device)
              / W).view(1, W).expand(H, W)
        ref = torch.stack((xs, ys), -1).reshape(H * W, 2)
        return ref[None].repeat(bs, 1, 1)  # [bs, H*W, 2]

    def _report_visibility_once(self, bev_mask) -> None:
        """Print, once per process, how much of the BEV grid the cameras reach.

        This is the difference between a model that learns from images and one
        that cannot. A 30-epoch run reached 0.000% here -- the logs it trained on
        carried a pre-rectification pinhole against rectified files, so every
        reference point projected outside the image, SpatialCrossAttention
        returned its residual, and nothing in the loss, the score or the
        checkpoints said a word. It ended at a plausible 0.9259.

        Two numbers, because the mask is [num_cam, bs, num_query, num_z] and the
        two useful reductions of it say different things:

        ``any``   the share of BEV cells at least one camera can see. This is
                  the coverage of the grid, and on this rig it is ~98%: the only
                  cells no camera reaches are within about five metres of the
                  car, under the 37-degree vertical field of the rectified view.
        ``mean``  the share of (cell, camera, z-anchor) triples that project
                  inside an image, ~33%. It is lower by construction -- with
                  three cameras each covering 68 degrees horizontally, roughly
                  one of them sees any given cell -- and it is the number that
                  goes to zero when the calibration and the images are not the
                  same pair. Do not read it as coverage.
        """
        if getattr(self, "_visibility_reported", False):
            return
        self._visibility_reported = True
        # [num_cam, bs, num_query, num_z] -> per query, over cameras and anchors
        any_cam = float(bev_mask.any(dim=0).any(dim=-1).float().mean())
        mean = float(bev_mask.float().mean())
        note = "" if mean > 0.05 else \
            "   <-- near zero: the encoder is reading no image content at all"
        print(f"[bevformer] BEV cells reached by any camera: {100 * any_cam:.3f}%"
              f"  (cell-camera-anchor mean {100 * mean:.3f}%){note}", flush=True)

    def point_sampling(self, reference_points_3d, lidar2img, img_h, img_w):
        """Project pillar points into every camera.

        :param reference_points_3d: [bs, num_z, num_query, 3] normalized
        :param lidar2img: [bs, num_cam, 4, 4]
        :return: (ref_cam [num_cam, bs, num_query, num_z, 2],
                  bev_mask [num_cam, bs, num_query, num_z])
        """
        # x/y come from point_cloud_range; z from pillar_z_range, which defaults
        # to it. They are separable because the two are used for different jobs:
        # here the z limits decide WHERE IN THE IMAGE a pillar samples, while the
        # detection head reuses point_cloud_range to denormalise its predicted
        # box z. See BEVFormerEncoder.__init__ for why that matters.
        pc_range = self.point_cloud_range
        z_lo, z_hi = self.pillar_z_range
        # Do the camera projection in float32 with autocast disabled: lidar2img
        # entries (focal length x metric coords) easily exceed fp16's max
        # (65504), producing inf/NaN under 16-mixed AMP. The original BEVFormer
        # casts this matmul to float32 for the same reason.
        # The original also switches TF32 off around this matmul: on Ampere+ the
        # default TF32 path keeps only 10 mantissa bits, and lidar2img mixes
        # focal lengths with metric coordinates, so the projected pixel
        # coordinates lose precision exactly where the bev_mask cut-off sits.
        allow_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        allow_tf32_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        with torch.cuda.amp.autocast(enabled=False):
            ref = reference_points_3d.float().clone()
            ref[..., 0] = ref[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
            ref[..., 1] = ref[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
            ref[..., 2] = ref[..., 2] * (z_hi - z_lo) + z_lo
            ref = torch.cat([ref, torch.ones_like(ref[..., :1])], -1)  # [bs, num_z, num_query, 4]

            bs, num_z, num_query, _ = ref.shape
            num_cam = lidar2img.shape[1]
            ref = ref.view(bs, 1, num_z, num_query, 4, 1).repeat(1, num_cam, 1, 1, 1, 1)
            l2i = lidar2img.float().view(bs, num_cam, 1, 1, 4, 4).repeat(1, 1, num_z, num_query, 1, 1)
            ref_cam = torch.matmul(l2i, ref).squeeze(-1)  # [bs, num_cam, num_z, num_query, 4]

            eps = 1e-5
            mask = ref_cam[..., 2:3] > eps
            ref_cam = ref_cam[..., 0:2] / torch.clamp(ref_cam[..., 2:3], min=eps)
            ref_cam[..., 0] /= img_w
            ref_cam[..., 1] /= img_h
            mask = (mask
                    & (ref_cam[..., 0:1] > 0.0) & (ref_cam[..., 0:1] < 1.0)
                    & (ref_cam[..., 1:2] > 0.0) & (ref_cam[..., 1:2] < 1.0))
            mask = torch.nan_to_num(mask.float()).bool()
            ref_cam = torch.nan_to_num(ref_cam)

        torch.backends.cuda.matmul.allow_tf32 = allow_tf32_matmul
        torch.backends.cudnn.allow_tf32 = allow_tf32_cudnn

        # [bs, num_cam, num_z, num_query, 2] -> [num_cam, bs, num_query, num_z, 2]
        ref_cam = ref_cam.permute(1, 0, 3, 2, 4).contiguous()
        mask = mask.squeeze(-1).permute(1, 0, 3, 2).contiguous()
        return ref_cam, mask

    def forward(self, bev_query, value, bev_pos, bev_h, bev_w, lidar2img,
                img_spatial_shapes, img_level_start_index, img_h, img_w,
                n_layers=None):
        """
        :param bev_query: [bs, num_query, C]
        :param value: [num_cam, bs, num_value, C]
        :param bev_pos: [bs, num_query, C]
        :param n_layers: run only the first N layers (weight-shared, no new
            parameters). Used to give history frames a cheaper encoder than the
            current frame -- they are fused by a conv afterwards, so they do not
            need the full depth. None runs all of them.
        :return: [bs, num_query, C]
        """
        bs = bev_query.shape[0]
        device, dtype = bev_query.device, bev_query.dtype

        z_range = self.pillar_z_range[1] - self.pillar_z_range[0]
        ref_3d = self.get_reference_points_3d(
            bev_h, bev_w, self.num_z_anchors, z_range, bs, device, dtype)
        ref_2d = self.get_reference_points_2d(bev_h, bev_w, bs, device, dtype)
        ref_cam, bev_mask = self.point_sampling(ref_3d, lidar2img, img_h, img_w)
        self._report_visibility_once(bev_mask)

        bev_spatial_shapes = torch.tensor([[bev_h, bev_w]], device=device, dtype=torch.long)
        bev_level_start_index = torch.tensor([0], device=device, dtype=torch.long)

        query = bev_query
        layers = self.layers if n_layers is None else self.layers[:max(1, n_layers)]
        for layer in layers:
            query = layer(
                query, value, bev_pos, ref_2d, ref_cam, bev_mask,
                bev_spatial_shapes, bev_level_start_index,
                img_spatial_shapes, img_level_start_index)
        return query
