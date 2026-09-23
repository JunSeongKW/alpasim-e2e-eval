"""
Implements the TransFuser vision backbone.
"""
import os
import timm

from torch import nn

from navsim.agents.backbones.bevformer import BEVFormerM
from navsim.agents.backbones.vov import VoVNet
from navsim.agents.drivesuprim.drivesuprim_config import DriveSuprimConfig
from navsim.agents.utils.vit import DAViT


class DriveSuprimBackbonePE(nn.Module):
    """
    Multi-scale Fusion Transformer for image + LiDAR feature fusion
    """

    def __init__(self, config: DriveSuprimConfig):

        super().__init__()
        self.config = config
        self.backbone_type = config.backbone_type
        if config.backbone_type == 'vov':
            self.image_encoder = VoVNet(
                spec_name='V-99-eSE',
                out_features=['stage4', 'stage5'],
                norm_eval=True,
                with_cp=True,
                init_cfg=dict(
                    type='Pretrained',
                    checkpoint=config.vov_ckpt,
                    prefix='img_backbone.'
                )
            )
            vit_channels = 1024
            self.image_encoder.init_weights()
        elif config.backbone_type == 'vit':
            self.image_encoder = DAViT(ckpt=config.vit_ckpt)
            vit_channels = 1024
        elif config.backbone_type == 'bevformer_m':
            # SafeDrive-style BEV front-end. Consumes multi-view images + camera
            # calibration and returns a BEV feature map [bs, C, bev_h, bev_w].
            self.image_encoder = BEVFormerM(config)
            vit_channels = self.image_encoder.img_feat_c
        elif config.backbone_type == 'resnet34':
            self.image_encoder = timm.create_model(
                'resnet34', pretrained=False, features_only=True
            )
            vit_channels = 512
        elif config.backbone_type == 'resnet50':
            self.image_encoder = timm.create_model(
                'resnet50', pretrained=False, features_only=True
            )
            vit_channels = 2048
        else:
            raise ValueError

        self.avgpool_img = nn.AdaptiveAvgPool2d(
            (self.config.img_vert_anchors, self.config.img_horz_anchors)
        )
        self.img_feat_c = vit_channels

    def forward(self, image, **kwargs):

        if isinstance(self.image_encoder, BEVFormerM):
            # `image` is the feature dict (multi-view imgs + calibration + ego
            # pose). The BEV map is already at the target grid size, so we skip
            # the perspective-feature adaptive pooling.
            return self.image_encoder(image)
        if isinstance(self.image_encoder, DAViT):
            image_feat = self.image_encoder(image, **kwargs)[-1]
        else:
            image_feat = self.image_encoder(image)[-1]

        return self.avgpool_img(image_feat)
