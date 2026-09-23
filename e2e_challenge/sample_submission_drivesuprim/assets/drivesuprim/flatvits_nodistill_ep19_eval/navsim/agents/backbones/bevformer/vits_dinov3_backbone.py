"""ViT-Small (DINOv3) per-view backbone for the BEVFormer-M front-end.

Ported from EADv1.1's ACTUAL deployed ViT-S setup
(EAD_navsim/navsim/agents/ead_simplebev_world/ead_backbone_simplebev.py --
``SimpleBEVDinoBackbone`` / ``SimpleBEVViTFeaturePyramid`` / ``LearnedRGBResize2``),
confirmed against EAD's eadv1.1_navsim_c_phase1.yaml:
    simplebev_vit_model:   "vit_small_patch16_dinov3"
    simplebev_vit_pretrained: True     (timm pulls the HF-hub weights directly --
                                         no local checkpoint file in EAD's own recipe)
    simplebev_vit_amp:     True, simplebev_vit_flash_dtype: "bf16"
    simplebev_learned_resize2: True    (EAD runs at 1920x1080 and halves it with a
                                         learned resizer before the patch16 ViT)
    opt_paramwise_cfg.name.image_encoder.lr_mult: 1.0   (no backbone LR throttling)

WHAT CARRIES OVER VERBATIM: the ViT choice (DINOv3 ViT-Small/patch16), timm
pretrained-weight loading path, the ImageNet normalisation convention, the
bf16-autocast-around-the-ViT-forward pattern, and the
learned-resize-stem + up/native/down conv pyramid-synthesis DESIGN (the classes
below are line-for-line the same shape as EAD's).

WHAT IS ADAPTED for this project's 512x256, 3-camera input (see
drivesuprim_agent_bevformer_vov_v2_cnx.yaml / v2_r50.yaml for the shared
architecture this backbone drops into):
  * The learned-resize /2 stem defaults OFF (``learned_resize2=False``). EAD's
    resizer exists because feeding a patch16 ViT full-res 1920x1080 pixels is
    prohibitively expensive; 512x256 already divides patch16 exactly
    (32x16 patches, 512 tokens/camera), so there is nothing to buy back by
    halving it first. The class is kept here, config-gated, in case a future
    higher-resolution ablation needs it -- porting the class rather than
    dropping it is what "same design pattern, re-tuned" means.
  * The feature pyramid defaults to ms_up=1, ms_down=1 -> exactly 3 levels
    (patch/2=8, patch=16, patch*2=32), matching the 8/16/32 stride convention
    the ConvNeXt-V2-Tiny (v2_cnx) and ResNet-50 (v2_r50) backbones already
    feed into FPN(num_outs=3) -- see bevformer_cnn_out_strides in those
    configs. EAD's own default (ms_up=1, ms_down=2 -> 4 levels) matched ITS
    4-level BEVFormer-style neck, not this repo's 3.
  * No internal image normalisation. BEVFormerM.extract_img_feat already
    applies ``bevformer_img_norm='imagenet'`` (the same convention v2_cnx and
    v2_r50 use) to the [0,1] RGB tensor before calling the image backbone, so
    normalising again in here would double-normalise. EAD's own class owns its
    normalisation because nothing upstream of it does; DriveSuprim's dispatcher
    already does, so this module does not repeat it (matching how
    ``ViTImageBackbone``/``DinoV2ImageBackbone`` in bevformer_backbone.py --
    the existing ViT-L path -- also expect pre-normalised input).
  * forward() returns a single [B, C, H/patch, W/patch] map directly (like
    every other per-view backbone in this file) and the pyramid class takes
    that tensor directly, rather than EAD's own (tokens, grid_hw) tuple
    interface -- EAD's caller (obtain_bev in ead_backbone_simplebev.py) is
    different from BEVFormerM.extract_img_feat, which always calls
    ``self.neck(self.image_backbone(x))``.
"""
import math

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


def _gn(channels: int) -> nn.GroupNorm:
    """Largest group count (<=32) that evenly divides ``channels``. Ported
    verbatim from EAD's ead_backbone_simplebev.py."""
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class LearnedRGBResizeBlock(nn.Module):
    """Ported verbatim from EAD's ead_backbone_simplebev.py."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            _gn(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            _gn(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class LearnedRGBResize2(nn.Module):
    """Task-driven RGB /2 resizer: bilinear skip + lightweight residual branch.

    Ported verbatim from EAD's ead_backbone_simplebev.py (itself ported from
    ead_bevdepth, the fullres+dinov3 reference). Starts as a plain bilinear /2
    downsample (to_rgb zero-init, aa_down = Gaussian blur) and learns a
    task-driven correction. Off by default here -- see module docstring --
    but kept so a future higher-resolution ViT-S ablation can turn it on
    without re-porting it.
    """

    def __init__(self, hidden_channels: int = 16, num_blocks: int = 1):
        super().__init__()
        self.aa_down = nn.Conv2d(3, 3, kernel_size=3, stride=2, padding=1, groups=3, bias=False)
        layers = [
            self.aa_down,
            nn.Conv2d(3, hidden_channels, kernel_size=3, padding=1, bias=False),
            _gn(hidden_channels),
            nn.GELU(),
        ]
        for _ in range(max(0, int(num_blocks))):
            layers.append(LearnedRGBResizeBlock(hidden_channels))
        self.to_rgb = nn.Conv2d(hidden_channels, 3, kernel_size=3, padding=1, bias=True)
        layers.append(self.to_rgb)
        self.residual = nn.Sequential(*layers)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.reset_parameters()

    def reset_parameters(self):
        kernel = torch.tensor(
            [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]],
            dtype=self.aa_down.weight.dtype,
        ) / 16.0
        with torch.no_grad():
            self.aa_down.weight.zero_()
            for c in range(3):
                self.aa_down.weight[c, 0].copy_(kernel)
            self.to_rgb.weight.zero_()
            self.to_rgb.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_size = (int(math.ceil(x.shape[-2] / 2)), int(math.ceil(x.shape[-1] / 2)))
        skip = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        residual = self.residual(x)
        if residual.shape[-2:] != target_size:
            residual = F.interpolate(residual, size=target_size, mode="bilinear", align_corners=False)
        return skip.to(dtype=residual.dtype) + self.alpha.to(dtype=residual.dtype) * residual


class ViTSDinoV3Backbone(nn.Module):
    """DINOv3 ViT-Small/patch16 per-view backbone.

    Same output CONTRACT as ``ViTImageBackbone``/``DinoV2ImageBackbone`` in
    this file: a single stride=patch feature map [B, C, H/patch, W/patch].
    Same LOADING/AMP behaviour as EAD's ``SimpleBEVDinoBackbone``: timm
    pretrained (network/HF-hub pull, verified available: timm 1.0.28 exposes
    'vit_small_patch16_dinov3', hf_hub_id 'timm/vit_small_patch16_dinov3.lvd1689m',
    embed_dim 384, num_prefix_tokens 5 [1 cls + 4 register]) OR a local
    checkpoint, and the ViT forward runs under a bf16 autocast the same way
    EAD's does.
    """

    def __init__(self, name="vit_small_patch16_dinov3", pretrained=True, ckpt="",
                 grad_checkpoint=False, freeze=False, learned_resize2=False,
                 resize_hidden_channels=16, resize_blocks=1,
                 vit_amp=True, flash_dtype="bf16", image_size=None):
        super().__init__()
        self._frozen = freeze
        self.vit_amp = bool(vit_amp)
        self.flash_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(
            str(flash_dtype).lower(), torch.bfloat16)

        # Optional full-res -> learned /2 RGB resizer front-end (see module
        # docstring: off by default at 512x256, which already divides patch16).
        self.input_stem = LearnedRGBResize2(
            hidden_channels=int(resize_hidden_channels), num_blocks=int(resize_blocks),
        ) if learned_resize2 else None

        create_kwargs = dict(num_classes=0, dynamic_img_size=True, dynamic_img_pad=True)
        if image_size is not None:
            create_kwargs["img_size"] = tuple(image_size)
        # EAD's ACTUAL deployed recipe (eadv1.1_navsim_c_phase1.yaml) sets
        # simplebev_vit_pretrained=True with no local checkpoint -- timm pulls
        # the HF-hub weights directly. Ported as-is: pretrained=True/no ckpt is
        # the common case; a local ckpt (if ever supplied) takes priority and
        # disables the network pull, same precedence as ViTImageBackbone below.
        self.vit = timm.create_model(
            name, pretrained=bool(pretrained) and not ckpt, **create_kwargs)
        self.embed_dim = int(self.vit.embed_dim)
        self.patch = self.vit.patch_embed.patch_size[0]
        self.num_prefix = getattr(self.vit, "num_prefix_tokens", 0)

        if ckpt:
            # EAD's own ckpt convention (SimpleBEVDinoBackbone): DDP-saved
            # checkpoints prefix every key with 'module.'; strip it and load
            # with strict=False. Ported verbatim.
            state = torch.load(ckpt, map_location="cpu")
            state = state.get("state_dict", state)
            state = {k.replace("module.", ""): v for k, v in state.items()}
            msg = self.vit.load_state_dict(state, strict=False)
            print(f"[ViTSDinoV3Backbone] loaded {ckpt}: "
                  f"missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")

        self._grad_checkpoint = bool(
            grad_checkpoint and not freeze and hasattr(self.vit, "set_grad_checkpointing")
        )
        if self._grad_checkpoint:
            self.vit.set_grad_checkpointing()
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False
            self.vit.eval()

    def train(self, mode=True):
        # A frozen ViT stays in eval mode even inside the training loop, so its
        # dropout / stochastic-depth do not perturb the (detached) features --
        # same convention as ViTImageBackbone/DinoV2ImageBackbone above.
        super().train(mode)
        if self._frozen:
            self.vit.eval()
        # getattr: nn.Module.train() can reach a child before its own __init__
        # has run to the line that sets this.
        if getattr(self, "_grad_checkpoint", False):
            # Gradient checkpointing exists to trade compute for activation
            # memory in the backward pass, so at inference it buys nothing --
            # and it is not merely wasteful there: Lightning's predict loop runs
            # under torch.inference_mode(), where checkpoint()'s attempt to save
            # its inputs for backward raises "Inference tensors cannot be saved
            # for backward" and the whole evaluation dies on the first batch.
            self.vit.set_grad_checkpointing(mode and not self._frozen)
        return self

    def _amp_ctx(self, x):
        enabled = self.vit_amp and x.is_cuda
        return torch.autocast(device_type=x.device.type, dtype=self.flash_dtype, enabled=enabled)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, 3, H, W], already normalised by the caller (BEVFormerM
        applies bevformer_img_norm upstream -- see module docstring on why this
        does NOT normalise again). Returns [B, embed_dim, H', W']."""
        if self.input_stem is not None:
            with self._amp_ctx(x):
                x = self.input_stem(x)
        _, _, h, w = x.shape
        gh, gw = int(math.ceil(h / self.patch)), int(math.ceil(w / self.patch))

        # When frozen, skip building the autograd graph entirely -- same
        # perf convention ViTImageBackbone/DinoV2ImageBackbone use above for
        # their frozen case (nothing to recompute a gradient for).
        forward_fn = (torch.no_grad if self._frozen else torch.enable_grad)
        with forward_fn():
            with self._amp_ctx(x):
                # forward_features handles DINOv3's dict output / register
                # tokens the same way EAD's robust (input_stem-present) path
                # does; ported verbatim.
                tokens = self.vit.forward_features(x)
        # Branching ported verbatim from EAD's SimpleBEVDinoBackbone.forward:
        # 'x_norm_patchtokens' (DINOv2-style dict output) already excludes the
        # prefix tokens by convention, so it skips the slice below; every other
        # shape (plain tensor, or a dict without that key) gets num_prefix
        # stripped off explicitly.
        skip_prefix_strip = False
        if isinstance(tokens, dict):
            if "x_norm_patchtokens" in tokens:
                tokens = tokens["x_norm_patchtokens"]
                skip_prefix_strip = True
            else:
                tokens = tokens.get("x", next(iter(tokens.values())))
        if tokens.dim() == 4:
            # NHWC patch maps (some timm ViTs) -> flatten to a token sequence
            # first so the prefix-stripping/reshape below is uniform.
            tokens = tokens.reshape(tokens.shape[0], -1, tokens.shape[-1])
        if not skip_prefix_strip:
            tokens = tokens[:, self.num_prefix:]              # drop cls/register tokens
        c = tokens.shape[-1]
        return tokens.transpose(1, 2).reshape(-1, c, gh, gw).contiguous().to(x.dtype)


class ViTSFeaturePyramid(nn.Module):
    """ViTDet/DPT-style simple pyramid, ported from EAD's
    ``SimpleBEVViTFeaturePyramid``: up/native/down conv synthesis of
    ``ms_up + 1 + ms_down`` levels from ONE ViT scale.

    Default ms_up=1, ms_down=1 -> 3 levels at strides patch/2, patch, patch*2
    (8/16/32 for patch16), matching what v2_cnx/v2_r50 already feed FPN's
    num_outs=3 -- see module docstring. EAD's own default (ms_up=1, ms_down=2)
    made 4 levels for ITS 4-level neck.

    Differs from EAD's class only in the forward() signature: takes the
    [B, C, H, W] map ``ViTSDinoV3Backbone.forward`` already returns, rather
    than EAD's own (tokens, grid_hw) pair, so it drops into
    ``self.neck(self.image_backbone(x))`` like every other backbone/neck pair
    in this file.
    """

    def __init__(self, in_channels: int, out_channels: int, ms_up: int = 1, ms_down: int = 1):
        super().__init__()
        self.ms_up_convs = nn.ModuleList(
            [nn.ConvTranspose2d(in_channels, in_channels, 2, stride=2) for _ in range(int(ms_up))])
        self.ms_down_convs = nn.ModuleList(
            [nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1) for _ in range(int(ms_down))])
        num_levels = int(ms_up) + 1 + int(ms_down)
        self.proj = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                _gn(out_channels),
                nn.GELU(),
            ) for _ in range(num_levels)
        ])

    def forward(self, x: torch.Tensor):
        ups, u = [], x
        for conv in self.ms_up_convs:
            u = conv(u)
            ups.append(u)
        maps = list(reversed(ups)) + [x]           # finest -> native
        d = x
        for conv in self.ms_down_convs:
            d = conv(d)
            maps.append(d)                          # ... -> coarsest
        return [proj(m) for proj, m in zip(self.proj, maps)]
