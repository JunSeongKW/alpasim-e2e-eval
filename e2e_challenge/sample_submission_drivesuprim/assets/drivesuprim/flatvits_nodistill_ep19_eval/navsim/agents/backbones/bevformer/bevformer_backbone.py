"""Top-level BEVFormer-M front-end module.

Pipeline (camera-only, temporal), matching SafeDrive's front-end description:
  multi-view images -> ResNet backbone -> FPN neck -> BEVFormer encoder
  (learnable BEV queries do deformable spatial cross-attention using camera
  calibration) -> per-frame BEV feature -> temporal align + concat + conv
  -> spatio-temporal BEV feature map [bs, C, bev_h, bev_w].

The module exposes ``img_feat_c`` (channel count of the BEV map) and a
``forward(features)`` that reads the multi-view images / calibration / ego
poses assembled by the feature builder, so it drops into ``DriveSuprimBackbonePE``
in place of an image encoder.
"""
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from navsim.agents.backbones.bevformer.encoder import BEVFormerEncoder


class SimpleFPN(nn.Module):
    """Dependency-free reimplementation of mmdet's FPN, configured the way the
    original BEVFormer config does:

        img_neck=dict(type='FPN', in_channels=[512, 1024, 2048], out_channels=256,
                      start_level=0, add_extra_convs='on_output', num_outs=4,
                      relu_before_extra_convs=True)

    Verified numerically identical to ``mmdet.models.necks.FPN`` (max abs diff
    0.0 across four in_channels / num_outs configurations when both are given the
    same weights), including the two details that are easy to get wrong:
      * the extra levels use stride-2 3x3 convs, and
      * ``relu_before_extra_convs`` applies from the SECOND extra level onwards --
        the first extra conv consumes ``outs[-1]`` directly.
    Weights follow mmdet's ``init_cfg=dict(type='Xavier', layer='Conv2d',
    distribution='uniform')`` rather than the PyTorch default.

    Kept as local code (instead of importing mmdet.FPN) so the BEV front-end does
    not depend on mmdet's registry/ConvModule machinery.
    """

    def __init__(self, in_channels_list, out_channels, num_outs=None,
                 relu_before_extra_convs=True):
        super().__init__()
        if num_outs is None:
            num_outs = len(in_channels_list)
        # mmdet FPN in the BEVFormer config sets relu_before_extra_convs=True.
        self.relu_before_extra_convs = relu_before_extra_convs
        self.num_ins = len(in_channels_list)
        self.num_outs = num_outs
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1) for c in in_channels_list])
        self.output_convs = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, 3, padding=1) for _ in in_channels_list])
        # Extra levels for num_outs > len(in_channels) (BEVFormer: 4 outs from 3
        # stages). mmdet builds these as stride-2 3x3 convs and feeds them from
        # outs[-1] (add_extra_convs='on_output').
        self.extra_convs = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1)
             for _ in range(num_outs - self.num_ins)])
        self.init_weights()

    def init_weights(self):
        """mmdet's FPN declares
        ``init_cfg=dict(type='Xavier', layer='Conv2d', distribution='uniform')``,
        which is applied to every Conv2d in the neck. PyTorch's default is
        Kaiming-uniform, so without this the neck starts from a different
        distribution than the reference implementation."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, feats):
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, feats)]
        # top-down pathway (coarse -> fine)
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest")
        outs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        # Extra coarser levels from the last output (add_extra_convs='on_output').
        # NOTE mmdet applies the ReLU only from the SECOND extra level onwards --
        # the first extra conv consumes outs[-1] directly. With num_outs=4 and 3
        # backbone stages there is exactly one extra level, so no ReLU is used.
        for i, extra in enumerate(self.extra_convs):
            src = outs[-1]
            if i > 0 and self.relu_before_extra_convs:
                src = F.relu(src)
            outs.append(extra(src))
        # num_outs BELOW the number of backbone stages: keep the COARSEST levels
        # and drop the fine ones. Deformable attention costs one sample per level
        # but its value tensor is the concatenation of every level, and a stride-8
        # map has 4x the tokens of stride-16 -- so dropping the finest level is
        # where the saving is. The cost is small/distant objects, which is why
        # detection recall has to be re-measured when this is reduced.
        if len(outs) > self.num_outs:
            outs = outs[-self.num_outs:]
        return outs


class LearnedPositionalEncoding(nn.Module):
    """BEV positional encoding, factorised over rows and columns.

    Port of mmdet's LearnedPositionalEncoding, which is what the original
    BEVFormer config uses (num_feats=_dim_//2, row/col_num_embed=bev_h/bev_w).
    A row embedding and a column embedding of ``num_feats`` each are
    concatenated to give ``2*num_feats`` per cell.

    This replaces a flat nn.Embedding(bev_h*bev_w, embed_dims): for a 200x200
    grid at 256 dims that is 400x2x128 = 51.2K parameters instead of
    40000x256 = 10.2M, and it ties cells that share a row or a column instead
    of learning every cell independently.
    """

    def __init__(self, num_feats, row_num_embed, col_num_embed):
        super().__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed
        # mmdet declares init_cfg=dict(type='Uniform', layer='Embedding'),
        # i.e. mmcv's UniformInit with its a=0, b=1 defaults.
        nn.init.uniform_(self.row_embed.weight, 0.0, 1.0)
        nn.init.uniform_(self.col_embed.weight, 0.0, 1.0)

    def forward(self, bs, device, dtype):
        """:return: [bs, row_num_embed * col_num_embed, 2*num_feats], flattened
        row-major (index = row * n_cols + col) to match the BEV reference-point
        ordering in BEVFormerEncoder.get_reference_points_2d."""
        h, w = self.row_num_embed, self.col_num_embed
        x = torch.arange(w, device=device)
        y = torch.arange(h, device=device)
        x_embed = self.col_embed(x)                      # [w, num_feats]
        y_embed = self.row_embed(y)                      # [h, num_feats]
        pos = torch.cat(
            (x_embed.unsqueeze(0).repeat(h, 1, 1),
             y_embed.unsqueeze(1).repeat(1, w, 1)), dim=-1)   # [h, w, 2*num_feats]
        pos = pos.reshape(h * w, 2 * self.num_feats)          # row-major
        return pos[None].expand(bs, -1, -1).to(dtype)


class DinoV2ImageBackbone(nn.Module):
    """Per-view backbone built on DriveSuprim's own DinoVisionTransformer
    (navsim.agents.utils.vit) rather than timm.

    Same output contract as ViTImageBackbone: a single stride=patch map
    [B, C, H/p, W/p]. This is the ViT-L the DriveSuprim agents use --
    patch-16, no register tokens, weights from da_vitl16.pth. Patch-16 divides
    256x512 exactly, so it removes the /14 constraint that forced 252x504.
    """

    def __init__(self, ckpt="", grad_checkpoint=False, freeze=False):
        # NOTE grad_checkpoint is accepted for signature parity with
        # ViTImageBackbone but is not a switch here: DinoVisionTransformer's
        # _get_intermediate_layers_not_chunked wraps every block in
        # torch.utils.checkpoint unconditionally. See forward(), which turns it
        # off for the frozen case (no grads to recompute for).
        super().__init__()
        # Imported lazily: this module pulls in xformers-backed attention and is
        # only needed on the 'dinov2' path.
        from functools import partial

        from navsim.agents.utils.vit import (
            Block,
            DinoVisionTransformer,
            MemEffAttention,
        )

        self.vit = DinoVisionTransformer(
            patch_size=16,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4,
            init_values=1.0,
            ffn_layer="mlp",
            block_chunks=0,
            img_size=518,
            num_register_tokens=0,
            interpolate_antialias=False,
            interpolate_offset=0.1,
            block_fn=partial(Block, attn_class=MemEffAttention),
        )
        self.embed_dim = 1024
        self.patch = 16
        self._frozen = freeze

        if ckpt:
            sd = torch.load(ckpt, map_location="cpu")
            sd = sd.get("state_dict", sd)
            # da_vitl16.pth carries the depth head and (in agent checkpoints) an
            # 'agent.vadv2_model._backbone.image_encoder.pretrained' prefix.
            clean = {}
            for k, v in sd.items():
                if "depth_head" in k or "mask_token" in k:
                    continue
                k = k.replace(
                    "agent.vadv2_model._backbone.image_encoder.pretrained.", "")
                k = k.replace("pretrained.", "")
                clean[k] = v
            missing, unexpected = self.vit.load_state_dict(clean, strict=False)
            print(f"[DinoV2ImageBackbone] loaded {ckpt}: "
                  f"missing={len(missing)} unexpected={len(unexpected)}")
            if len(missing) > 50:
                raise RuntimeError(
                    f"da_vitl16 load looks wrong: {len(missing)} missing tensors. "
                    f"first few: {missing[:5]}")

        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False
            self.vit.eval()

    def train(self, mode=True):
        super().train(mode)
        if self._frozen:
            self.vit.eval()
        return self

    def forward(self, x):
        # reshape=True already returns [B, C, H/p, W/p].
        # The DINOv2 impl checkpoints every block unconditionally; when the ViT
        # is frozen there is nothing to recompute, so skip it under no_grad and
        # avoid paying for a second forward pass through 24 blocks.
        if self._frozen:
            with torch.no_grad():
                feats = self.vit.get_intermediate_layers(x, 1, None, reshape=True)
        else:
            feats = self.vit.get_intermediate_layers(x, 1, None, reshape=True)
        return feats[-1]


class ViTImageBackbone(nn.Module):
    """Plain-ViT per-view backbone: returns a SINGLE stride=patch feature map
    [B, C, H/p, W/p]. timm ViT with dynamic_img_size handles the non-square
    NAVSIM resolution and pos-embed interpolation; prefix (cls/reg) tokens are
    dropped and the patch tokens reshaped to a spatial grid."""

    def __init__(self, name, pretrained=False, ckpt="", grad_checkpoint=False, freeze=False):
        super().__init__()
        self.vit = timm.create_model(
            name, pretrained=pretrained, num_classes=0, dynamic_img_size=True)
        self.embed_dim = self.vit.embed_dim
        self.patch = self.vit.patch_embed.patch_size[0]
        self.num_prefix = getattr(self.vit, "num_prefix_tokens", 1)
        self._frozen = freeze
        if grad_checkpoint and not freeze and hasattr(self.vit, "set_grad_checkpointing"):
            self.vit.set_grad_checkpointing()
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False
            self.vit.eval()
        if ckpt:
            sd = torch.load(ckpt, map_location="cpu")
            sd = sd.get("state_dict", sd.get("model", sd))
            sd = {k.replace("image_encoder.", "").replace("backbone.", ""): v for k, v in sd.items()}
            missing, unexpected = self.vit.load_state_dict(sd, strict=False)
            print(f"[ViTImageBackbone] loaded {ckpt}: missing={len(missing)} unexpected={len(unexpected)}")

    def train(self, mode=True):
        # A frozen ViT stays in eval mode even inside the training loop, so its
        # dropout / stochastic-depth do not perturb the (detached) features.
        super().train(mode)
        if self._frozen:
            self.vit.eval()
        return self

    def forward(self, x):
        B, _, H, W = x.shape
        tokens = self.vit.forward_features(x)          # [B, prefix + h*w, C]
        tokens = tokens[:, self.num_prefix:]           # drop cls/reg
        h, w = H // self.patch, W // self.patch
        return tokens.transpose(1, 2).reshape(B, self.embed_dim, h, w)


class SimpleFeaturePyramid(nn.Module):
    """ViTDet-style pyramid: turn one stride-p ViT map into ``num_levels``
    feature maps at decreasing resolution (scales 2x, 1x, 0.5x, 0.25x for
    num_levels=4), each projected to ``out_channels``. Gives BEVFormer's
    multi-scale deformable cross-attention the levels it expects from a
    single-scale ViT."""

    def __init__(self, in_channels, out_channels, num_levels=4):
        super().__init__()
        # scale factors from finest to coarsest, centered on the native stride
        scales = [2.0, 1.0, 0.5, 0.25, 0.125][:num_levels]
        self.branches = nn.ModuleList()
        for s in scales:
            layers = []
            if s == 2.0:
                layers.append(nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2))
            elif s == 1.0:
                pass
            elif s == 0.5:
                layers.append(nn.MaxPool2d(2, 2))
            elif s == 0.25:
                layers += [nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2)]
            elif s == 0.125:
                layers += [nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2), nn.MaxPool2d(2, 2)]
            layers += [
                nn.Conv2d(in_channels, out_channels, 1),
                nn.GroupNorm(32, out_channels),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
            ]
            self.branches.append(nn.Sequential(*layers))

    def forward(self, x):
        # x is a single [B, C, h, w] map -> list of num_levels maps
        return [branch(x) for branch in self.branches]


class BEVFormerM(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dims = config.bev_embed_dims
        self.bev_h = config.bev_h
        self.bev_w = config.bev_w
        self.num_cams = config.bev_num_cameras
        self.num_z_anchors = config.bev_num_points_in_pillar
        self.seq_len = config.bev_seq_len
        self.img_h = config.bev_img_height
        self.img_w = config.bev_img_width
        self.use_temporal_align = config.bev_use_temporal_align
        self.use_grad_checkpoint = getattr(config, 'bev_use_grad_checkpoint', False)
        self.pc_range = tuple(config.point_cloud_range)

        # --- per-view image backbone + neck ---
        self.num_levels = config.bev_num_feature_levels
        backbone_kind = getattr(config, "bevformer_img_backbone_type", "cnn")
        if backbone_kind == "vit":
            # ViT (single-scale) + ViTDet Simple Feature Pyramid -> num_levels maps.
            vit_impl = getattr(config, "bevformer_vit_impl", "timm")
            if vit_impl == "dinov2":
                self.image_backbone = DinoV2ImageBackbone(
                    ckpt=config.bevformer_vit_ckpt,
                    grad_checkpoint=self.use_grad_checkpoint,
                    freeze=config.bevformer_vit_freeze,
                )
            elif vit_impl == "timm":
                self.image_backbone = ViTImageBackbone(
                    config.bevformer_vit_name,
                    pretrained=config.bevformer_vit_pretrained,
                    ckpt=config.bevformer_vit_ckpt,
                    grad_checkpoint=self.use_grad_checkpoint,
                    freeze=config.bevformer_vit_freeze,
                )
            else:
                raise ValueError(f"Unknown bevformer_vit_impl: {vit_impl}")
            self.neck = SimpleFeaturePyramid(
                self.image_backbone.embed_dim, self.embed_dims, num_levels=self.num_levels)
        elif backbone_kind == "vits":
            # DINOv3 ViT-Small/patch16, ported from EADv1.1's actual deployed
            # ViT-S setup. See vits_dinov3_backbone.py's module docstring for
            # the full port-vs-adapt rationale (ADD-ONLY: this branch is new;
            # the 'cnn'/'vit'/'vov' branches above/below are untouched).
            from navsim.agents.backbones.bevformer.vits_dinov3_backbone import (
                ViTSDinoV3Backbone,
                ViTSFeaturePyramid,
            )

            self.image_backbone = ViTSDinoV3Backbone(
                name=config.bevformer_vit_name,
                pretrained=config.bevformer_vit_pretrained,
                ckpt=config.bevformer_vit_ckpt,
                grad_checkpoint=self.use_grad_checkpoint,
                freeze=config.bevformer_vit_freeze,
                learned_resize2=bool(getattr(config, "bevformer_vits_learned_resize2", False)),
                resize_hidden_channels=int(getattr(config, "bevformer_vits_resize_hidden_channels", 16)),
                resize_blocks=int(getattr(config, "bevformer_vits_resize_blocks", 1)),
                vit_amp=bool(getattr(config, "bevformer_vits_amp", True)),
                flash_dtype=str(getattr(config, "bevformer_vits_flash_dtype", "bf16")),
            )
            self.neck = ViTSFeaturePyramid(
                self.image_backbone.embed_dim, self.embed_dims,
                ms_up=int(getattr(config, "bevformer_vits_ms_up", 1)),
                ms_down=int(getattr(config, "bevformer_vits_ms_down", 1)),
            )
        elif backbone_kind == "vov":
            # DriveSuprim's V2-99 (VoVNetV2-99, dd3d-pretrained). A CNN, so it
            # emits genuine multi-scale features and the neck can be a real FPN
            # -- the same shape as the original BEVFormer front-end
            # (ResNet101-DCN stages 8/16/32 -> FPN num_outs=4).
            from navsim.agents.backbones.vov import VoVNet

            out_features = list(config.bevformer_vov_out_features)
            init_cfg = None
            if config.bevformer_vov_ckpt:
                init_cfg = dict(type="Pretrained",
                                checkpoint=config.bevformer_vov_ckpt,
                                prefix="img_backbone.")
            self.image_backbone = VoVNet(
                spec_name="V-99-eSE",
                out_features=out_features,
                norm_eval=True,
                with_cp=self.use_grad_checkpoint,
                init_cfg=init_cfg,
            )
            if init_cfg is not None:
                self.image_backbone.init_weights()
            in_chs = [self.image_backbone._out_feature_channels[f] for f in out_features]
            self.neck = SimpleFPN(in_chs, self.embed_dims, num_outs=self.num_levels)
        else:
            # Select stages by STRIDE, not by a fixed index. ResNet's features_only
            # stages are reductions [2,4,8,16,32] so 8/16/32 is (2,3,4), but
            # ConvNeXt has only four stages, [4,8,16,32], where the same strides
            # are (1,2,3) -- the hard-coded tuple silently grabbed the wrong maps
            # (and ran off the end) on any non-ResNet backbone.
            want = tuple(getattr(config, "bevformer_cnn_out_strides", (8, 16, 32)))
            probe = timm.create_model(config.bevformer_img_backbone,
                                      pretrained=False, features_only=True)
            red = list(probe.feature_info.reduction())
            del probe
            missing = [s for s in want if s not in red]
            if missing:
                raise ValueError(
                    f"{config.bevformer_img_backbone} has stages at strides {red}; "
                    f"cannot provide {missing}. Set bevformer_cnn_out_strides.")
            out_indices = tuple(red.index(s) for s in want)
            self.image_backbone = timm.create_model(
                config.bevformer_img_backbone,
                pretrained=bool(getattr(config, "bevformer_cnn_pretrained", False)),
                features_only=True, out_indices=out_indices)
            nuimg = getattr(config, "bevformer_cnn_nuimg_ckpt", "") or ""
            if nuimg:
                self._load_nuimg_backbone_weights(self.image_backbone, nuimg)
            in_chs = self.image_backbone.feature_info.channels()
            # BEVFormer/UniAD use a 4-level FPN (num_outs=4) from 3 backbone stages.
            self.neck = SimpleFPN(in_chs, self.embed_dims, num_outs=self.num_levels)

        # --- input normalisation ---
        # The feature builder hands us ToTensor() output: RGB in [0, 1]. Each
        # pretrained backbone was trained under its own convention, and with
        # norm_eval=True the BatchNorms cannot re-centre themselves, so the
        # convention has to be reproduced here.
        #
        # 'dd3d': the V2-99 checkpoint is a raw DD3D detector and carries its own
        #   pixel_mean / pixel_std tensors ([103.530, 116.280, 123.675] /
        #   [57.375, 57.120, 58.395], BGR). VoVNet.forward reverses the channels
        #   itself, so the constants are registered flipped to RGB and applied
        #   before that swap. Measured on real NAVSIM images, this keeps notably
        #   more of the pretrained representation alive than feeding [0, 1]
        #   directly (pooled-feature participation ratio at stage4/stage5:
        #   10.5/3.2 vs 8.2/1.9 over a 40-image batch).
        # 'none': what DriveSuprim itself does -- no normalisation at all. Kept
        #   as an option because DriveSuprim reaches 85.9 EPDMS that way; with an
        #   unfrozen backbone the first conv can absorb the scale.
        norm_kind = getattr(config, "bevformer_img_norm", "none")
        if norm_kind == "dd3d":
            mean_bgr = torch.tensor([103.530, 116.280, 123.675])
            std_bgr = torch.tensor([57.375, 57.120, 58.395])
            self.register_buffer("img_mean", mean_bgr.flip(0).view(1, 3, 1, 1) / 255.0)
            self.register_buffer("img_std", std_bgr.flip(0).view(1, 3, 1, 1) / 255.0)
        elif norm_kind == "imagenet":
            # timm's ImageNet convention, which every FCMAE / IN-1k ConvNeXt
            # checkpoint was trained under. Applied to the [0,1] RGB the feature
            # builder produces.
            self.register_buffer("img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("img_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        elif norm_kind == "none":
            self.img_mean = None
            self.img_std = None
        else:
            raise ValueError(f"Unknown bevformer_img_norm: {norm_kind}")

        # --- BEV queries / positional & source embeddings ---
        num_query = self.bev_h * self.bev_w
        self.bev_query = nn.Embedding(num_query, self.embed_dims)
        self.bev_pos = LearnedPositionalEncoding(
            num_feats=self.embed_dims // 2,
            row_num_embed=self.bev_h,
            col_num_embed=self.bev_w,
        )
        self.cams_embeds = nn.Parameter(torch.zeros(self.num_cams, self.embed_dims))
        self.level_embeds = nn.Parameter(torch.zeros(self.num_levels, self.embed_dims))
        nn.init.normal_(self.cams_embeds, std=0.02)
        nn.init.normal_(self.level_embeds, std=0.02)

        self.encoder = BEVFormerEncoder(
            num_layers=config.bev_num_encoder_layers,
            embed_dims=self.embed_dims,
            num_cams=self.num_cams,
            num_levels=self.num_levels,
            feedforward_channels=self.embed_dims * 2,
            num_z_anchors=self.num_z_anchors,
            # Sampling points per head inside the spatial cross-attention. This is
            # 57% of an encoder layer's time -- the single largest item in the BEV
            # front-end -- so it is worth exposing rather than leaving at the
            # BEVFormer default. That default (8) goes with BEVFormer's FOUR
            # feature levels; this config runs three, so the total sample count
            # was already off their operating point.
            sca_num_points=int(getattr(config, "bev_sca_num_points", 8)),
            ssa_num_points=int(getattr(config, "bev_ssa_num_points", 4)),
            point_cloud_range=self.pc_range,
            # None -> pillars span point_cloud_range's z, i.e. BEVFormer's
            # behaviour and what every existing config gets.
            pillar_z_range=getattr(config, "bev_pillar_z_range", None),
        )

        # --- temporal fusion (concat over frames + conv) ---
        self.temporal_fuse = nn.Sequential(
            nn.Conv2d(self.embed_dims * self.seq_len, self.embed_dims, 3, padding=1),
            nn.GroupNorm(32, self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dims, self.embed_dims, 3, padding=1),
        )

        # --- streaming temporal state -------------------------------------
        # Each forward encodes bev_seq_len frames from scratch, but in sequential
        # operation frame t-1 was already encoded on the previous call. When
        # `bev_streaming` is on we keep the per-frame BEVs (each in ITS OWN ego
        # frame, un-warped) and reuse them, so only the current frame runs the
        # backbone + encoder: bev_seq_len x cost -> 1x.
        #
        # Storing them un-warped is what makes the cache time-independent: the
        # warp uses ego_pose supplied by the CURRENT call, so the same cached
        # tensor is valid however far it has since receded.
        #
        # Training always takes the full path -- the cache would leak activations
        # across samples and the batch order is shuffled anyway.
        # Encoder compute dtype. 'fp32' (default) or 'bf16'; see encode_bev_from_feats.
        _ed = str(getattr(config, "bev_encoder_dtype", "fp32")).lower()
        if _ed not in ("fp32", "bf16"):
            raise ValueError(f"bev_encoder_dtype must be fp32 or bf16, got {_ed}")
        self.encoder_dtype = torch.bfloat16 if _ed == "bf16" else torch.float32

        self.bev_streaming = bool(getattr(config, "bev_streaming", False))
        self._stream_bev = []          # oldest -> newest, length <= seq_len - 1
        if self.bev_streaming and getattr(config, "training", False):
            raise ValueError(
                "bev_streaming is an INFERENCE-only optimisation: it reuses BEV "
                "features across forward calls, which would leak state between "
                "training samples and break the shuffled batch order. Set it only "
                "on the config used for deployment.")

        self.img_feat_c = self.embed_dims

    @staticmethod
    def _load_nuimg_backbone_weights(timm_feat_model, ckpt_path: str) -> None:
        """Overwrite the image backbone with a nuImages DETECTION checkpoint.

        Mirrors EAD's _load_nuimg_backbone_weights (ead_backbone.py:138-170),
        which is itself StreamPETR's recipe: keep only `backbone.*`, strip that
        prefix, drop `fc.*`, load with strict=False. `module.` is stripped first
        so a DDP-saved checkpoint works too.

        timm's features_only wrapper holds the real backbone in `.model`, so the
        keys have to go there rather than into the wrapper.

        Runs AFTER timm's own pretrained load, so it REPLACES ImageNet rather
        than merging with it -- which is the point: the nuImages weights come
        from a detector trained on driving imagery.
        """
        import os
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"bevformer_cnn_nuimg_ckpt={ckpt_path} not found. Set it to '' to "
                "keep the plain ImageNet weights.")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state = ckpt.get("state_dict", ckpt)
        target = getattr(timm_feat_model, "model", timm_feat_model)
        new_sd = {}
        for k, v in state.items():
            if k.startswith("module."):
                k = k[len("module."):]
            if not k.startswith("backbone."):
                continue
            k = k[len("backbone."):]
            if k.startswith("fc."):
                continue
            new_sd[k] = v
        msg = target.load_state_dict(new_sd, strict=False)
        n_hit = len(new_sd) - len(msg.unexpected_keys)
        print(f"[nuImg preload] {ckpt_path}: backbone keys {len(new_sd)}, "
              f"loaded {n_hit}, missing {len(msg.missing_keys)}, "
              f"unexpected {len(msg.unexpected_keys)}")
        if n_hit == 0:
            raise RuntimeError(
                "nuImg checkpoint matched NOTHING in the image backbone. Its keys "
                "are torchvision-ResNet names, so this only works for a ResNet.")

    def reset_stream(self):
        """Drop the cached BEVs. Call when the frame sequence is discontinuous --
        a new log, a new scenario, or any jump in time."""
        self._stream_bev = []

    def extract_img_feat(self, imgs):
        """imgs [bs, num_cam, 3, H, W] -> (value [num_cam, bs, L, C],
        spatial_shapes [num_levels, 2], level_start_index [num_levels])."""
        bs, num_cam = imgs.shape[:2]
        x = imgs.flatten(0, 1)  # [bs*num_cam, 3, H, W]
        if self.img_mean is not None:
            x = (x - self.img_mean.to(x.dtype)) / self.img_std.to(x.dtype)
        feats = self.neck(self.image_backbone(x))

        feat_flatten, spatial_shapes = [], []
        for lvl, feat in enumerate(feats):
            _, c, h, w = feat.shape
            spatial_shapes.append((h, w))
            feat = feat.flatten(2).permute(0, 2, 1).view(bs, num_cam, h * w, c)
            feat = feat + self.cams_embeds[None, :, None, :]
            feat = feat + self.level_embeds[None, None, lvl:lvl + 1, :]
            feat_flatten.append(feat)
        feat_flatten = torch.cat(feat_flatten, dim=2)          # [bs, num_cam, L, C]
        feat_flatten = feat_flatten.permute(1, 0, 2, 3).contiguous()  # [num_cam, bs, L, C]

        spatial_shapes = torch.tensor(spatial_shapes, device=imgs.device, dtype=torch.long)
        level_start_index = torch.cat(
            [spatial_shapes.new_zeros(1), spatial_shapes.prod(1).cumsum(0)[:-1]])
        return feat_flatten, spatial_shapes, level_start_index

    def extract_img_feats_seq(self, bev_imgs):
        """Per-frame image features for the whole sequence, computed ONCE so the
        rotation-ensemble views (which share these images and differ only in
        lidar2img / yaw) reuse them instead of re-running ResNet+FPN per view.
        bev_imgs [bs, T, num_cam, 3, H, W] -> list length T of
        (value, spatial_shapes, level_start_index)."""
        return [self.extract_img_feat(bev_imgs[:, t]) for t in range(bev_imgs.shape[1])]

    def encode_bev_from_feats(self, value, spatial_shapes, level_start_index, lidar2img,
                              n_layers=None):
        """BEV encoder given precomputed image features. Returns [bs, C, H, W]
        in fp32 (the caller casts back to the working dtype).

        Runs the whole encoder (deformable self-attn + spatial cross-attn + FFN)
        in fp32: under fp16 AMP the BEV features can grow across the 6 layers and
        overflow fp16's range (max 65504) -> inf -> NaN, uncaught by grad clipping
        or the AMP scaler. The ResNet/FPN and the downstream head stay in fp16."""
        bs = value.shape[1]  # value is [num_cam, bs, L, C]
        bev_query = self.bev_query.weight[None].expand(bs, -1, -1)
        bev_pos = self.bev_pos(bs, bev_query.device, bev_query.dtype)
        # fp32 is the safe default here for the reason in the docstring. bf16 is
        # the one lower precision that is safe: it keeps fp32's exponent range
        # (~3e38), so the growth across layers that overflows fp16 at 65504
        # cannot overflow it -- only the mantissa shrinks. fp16 is deliberately
        # NOT offered.
        if self.encoder_dtype == torch.bfloat16:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                bev = self.encoder(
                    bev_query, value, bev_pos,
                    self.bev_h, self.bev_w, lidar2img.float(),
                    spatial_shapes, level_start_index, self.img_h, self.img_w,
                    n_layers=n_layers)
            bev = bev.float()
        else:
            with torch.cuda.amp.autocast(enabled=False):
                bev = self.encoder(
                    bev_query.float(), value.float(), bev_pos.float(),
                    self.bev_h, self.bev_w, lidar2img.float(),
                    spatial_shapes, level_start_index, self.img_h, self.img_w,
                    n_layers=n_layers)
        # [bs, bev_h*bev_w, C] -> [bs, C, bev_h, bev_w]
        return bev.permute(0, 2, 1).reshape(bs, self.embed_dims, self.bev_h, self.bev_w)

    def encode_bev(self, imgs, lidar2img, n_layers=None):
        """Single-frame BEV feature from raw images (non-shared path). imgs
        [bs, num_cam, 3, H, W], lidar2img [bs, num_cam, 4, 4] -> [bs, C, H, W] fp32."""
        value, spatial_shapes, level_start_index = self.extract_img_feat(imgs)
        return self.encode_bev_from_feats(value, spatial_shapes, level_start_index, lidar2img,
                                          n_layers=n_layers)

    def warp_bev(self, bev, ego_pose, yaw=None):
        """Warp a past-frame BEV (in its own ego frame) into the current ego
        frame. bev [bs, C, H, W]; ego_pose [bs, 3] = past origin (x, y, heading)
        expressed in the current frame.

        ``yaw`` [bs] (radians, optional) is the rotation-augmentation angle: when
        the projection is rotated by +yaw, BOTH the current and past BEVs live in
        their ego frames rotated by +yaw. The output grid ``p_c`` is then in the
        rotated current frame N_cur, and the past BEV is indexed in N_past, so the
        sampling map must be
            g = R_z(-yaw) · R(h)^T · (R_z(+yaw) · p_c - t)
        i.e. undo the current-frame rotation, apply the ego motion, then re-apply
        the past-frame rotation. With yaw=None/0 this reduces to R(h)^T (p_c - t),
        the un-augmented warp. (R_z and R(h) commute, so only a bounded
        translation term (I-R_z)·t was wrong before this fix.)"""
        bs, _, H, W = bev.shape
        device, dtype = bev.device, bev.dtype
        x_min, y_min, _, x_max, y_max, _ = self.pc_range

        xs = torch.linspace(x_min, x_max, W, device=device, dtype=dtype)
        ys = torch.linspace(y_min, y_max, H, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [H, W]
        p_c = torch.stack([grid_x, grid_y], -1)[None].expand(bs, -1, -1, -1)  # [bs, H, W, 2]

        px, py = p_c[..., 0], p_c[..., 1]  # output grid in the (rotated) current frame

        if yaw is not None:
            cy = torch.cos(yaw).to(dtype)[:, None, None]
            sy = torch.sin(yaw).to(dtype)[:, None, None]
            # N_cur -> L_cur : R_z(+yaw)
            px, py = cy * px - sy * py, sy * px + cy * py

        tx, ty, h = ego_pose[:, 0], ego_pose[:, 1], ego_pose[:, 2]
        cos, sin = torch.cos(h)[:, None, None], torch.sin(h)[:, None, None]
        dx = px - tx[:, None, None]
        dy = py - ty[:, None, None]
        # R(h)^T applied to (p - t): map current-frame point into past ego frame
        qx = cos * dx + sin * dy
        qy = -sin * dx + cos * dy

        if yaw is not None:
            # L_past -> N_past : R_z(-yaw)
            qx, qy = cy * qx + sy * qy, -sy * qx + cy * qy

        nx = (qx - x_min) / (x_max - x_min) * 2.0 - 1.0
        ny = (qy - y_min) / (y_max - y_min) * 2.0 - 1.0
        grid = torch.stack([nx, ny], -1)  # [bs, H, W, 2]
        return F.grid_sample(bev, grid, mode="bilinear",
                             padding_mode="zeros", align_corners=True)

    def forward(self, features):
        """features dict with:
          'bev_imgs'  : [bs, T, num_cam, 3, H, W]
          'lidar2img' : [bs, T, num_cam, 4, 4]
          'ego_pose'  : [bs, T, 3]  (each frame relative to the current frame)
        Returns spatio-temporal BEV feature map [bs, C, bev_h, bev_w].
        """
        bev_imgs = features["bev_imgs"]
        lidar2img = features["lidar2img"]
        ego_pose = features.get("ego_pose", None)
        # Rotation-augmentation yaw [bs] (radians); None/absent on the un-augmented
        # path. Needed so the temporal warp stays consistent with the rotated
        # projection (see warp_bev).
        yaw = features.get("bev_aug_yaw", None)
        # Precomputed per-frame image features shared across rotation-ensemble views
        # (all use the same images). When present we skip ResNet/FPN and only run
        # the BEV encoder per view; None -> compute image features here.
        precomp = features.get("bev_precomputed_img_feats", None)
        bs, T = bev_imgs.shape[:2]
        out_dtype = bev_imgs.dtype

        # Past frames without gradients, as BEVFormer's forward_train does: it
        # splits `prev_img = img[:, :-1]` off, runs obtain_history_bev under
        # no_grad + eval, and keeps only the current frame in the graph.
        # BEVFormer is the only verified source for this choice; what SafeDrive
        # does here is not known.
        #
        # They are computed BEFORE the main loop and the loop then skips them.
        # Interleaving a no_grad region with torch's non-reentrant checkpoint
        # (use_reentrant=False, needed here because bev_imgs carries no grad)
        # corrupts its saved-tensor bookkeeping on torch 2.0.1 and raises
        # "IndexError: list index out of range" from checkpoint.inner_pack.
        history_no_grad = self.training and getattr(self.config, "bev_history_no_grad", False)
        # History frames may run a SHALLOWER encoder than the current frame: they
        # are fused by a conv afterwards, so they contribute context rather than
        # the precise geometry the current frame carries. Same weights, first N
        # layers only -- no extra parameters. 0 keeps them at full depth.
        hist_layers = int(getattr(self.config, "bev_history_encoder_layers", 0)) or None
        precomputed_history = {}
        if history_no_grad:
            # cache_enabled=False matters. Under AMP, autocast caches the fp16
            # casts of each weight. A no_grad pass populates that cache, the
            # checkpointed forward then reuses the cached casts and saves fewer
            # tensors, and by the time backward recomputes it the cache has been
            # cleared -- so the recompute saves more tensors than the original
            # and checkpoint.inner_pack indexes past the end of its holder list.
            with torch.no_grad(), torch.autocast(
                    "cuda", dtype=torch.get_autocast_gpu_dtype(),
                    enabled=torch.is_autocast_enabled(), cache_enabled=False):
                for t in range(T - 1):
                    bev_t = (self.encode_bev_from_feats(*precomp[t], lidar2img[:, t],
                                                        n_layers=hist_layers)
                             if precomp is not None
                             else self.encode_bev(bev_imgs[:, t], lidar2img[:, t],
                                                  n_layers=hist_layers))
                    bev_t = bev_t.to(out_dtype)
                    if self.use_temporal_align and ego_pose is not None:
                        bev_t = self.warp_bev(bev_t, ego_pose[:, t], yaw=yaw)
                    precomputed_history[t] = bev_t

        # --- streaming fast path ------------------------------------------
        # Encode ONLY the current frame; the history comes from the cache. Falls
        # back to the full path whenever the cache is not usable (first call after
        # a reset, a batch-size change, or training).
        stream = (self.bev_streaming and not self.training
                  and len(self._stream_bev) == T - 1
                  and all(b.shape[0] == bs for b in self._stream_bev))
        if self.bev_streaming and not self.training and not stream:
            self._stream_bev = []       # stale or incomplete -> rebuild below

        if stream:
            cur = (self.encode_bev_from_feats(*precomp[T - 1], lidar2img[:, T - 1])
                   if precomp is not None
                   else self.encode_bev(bev_imgs[:, T - 1], lidar2img[:, T - 1]))
            cur = cur.to(out_dtype)
            bev_list = []
            for t in range(T - 1):
                b = self._stream_bev[t]
                if self.use_temporal_align and ego_pose is not None:
                    b = self.warp_bev(b, ego_pose[:, t], yaw=yaw)
                bev_list.append(b)
            bev_list.append(cur)
            # keep the newest seq_len-1 frames, un-warped
            self._stream_bev = (self._stream_bev[1:] + [cur.detach()]) if T > 1 else []
            return self.temporal_fuse(torch.cat(bev_list, dim=1))

        bev_list, bev_raw = [], []
        for t in range(T):
            if t in precomputed_history:
                bev_list.append(precomputed_history[t])
                bev_raw.append(precomputed_history[t])
                continue
            # Gradient-checkpoint the per-view BEV encoder: recompute it in backward
            # instead of storing activations. preserve_rng_state (default) keeps
            # dropout consistent; use_reentrant=False handles inputs without grad.
            # Past frames without gradients, as BEVFormer's forward_train does:
            # it splits `prev_img = img[:, :-1]` off and runs obtain_history_bev
            # under no_grad + eval, keeping only the current frame in the graph.
            # The backbone then learns from the current frame alone, while the
            # temporal fusion still trains on all three. Off by default because
            # SafeDrive's concat+conv treats the frames symmetrically, unlike
            # BEVFormer's recurrent prev_bev.
            ckpt = self.use_grad_checkpoint and self.training
            nl = hist_layers if t != T - 1 else None      # current frame keeps full depth
            if precomp is not None:
                value, ss, lsi = precomp[t]
                if ckpt:
                    bev_t = checkpoint(self.encode_bev_from_feats, value, ss, lsi,
                                       lidar2img[:, t], nl, use_reentrant=False)
                else:
                    bev_t = self.encode_bev_from_feats(value, ss, lsi, lidar2img[:, t],
                                                       n_layers=nl)
            else:
                if ckpt:
                    bev_t = checkpoint(self.encode_bev, bev_imgs[:, t], lidar2img[:, t],
                                       nl, use_reentrant=False)
                else:
                    bev_t = self.encode_bev(bev_imgs[:, t], lidar2img[:, t], n_layers=nl)
            bev_t = bev_t.to(out_dtype)
            bev_raw.append(bev_t)          # pre-warp, in this frame's own ego frame
            if self.use_temporal_align and ego_pose is not None and t != T - 1:
                bev_t = self.warp_bev(bev_t, ego_pose[:, t], yaw=yaw)
            bev_list.append(bev_t)

        if self.bev_streaming and not self.training:
            # Cache the UN-warped tensors: warping is done per call against that
            # call's ego_pose, so a warped tensor would be transformed twice.
            # Next call's history is this call's frames shifted by one.
            self._stream_bev = [b.detach() for b in bev_raw[1:]]

        bev_cat = torch.cat(bev_list, dim=1)  # [bs, C*T, bev_h, bev_w]
        return self.temporal_fuse(bev_cat)
