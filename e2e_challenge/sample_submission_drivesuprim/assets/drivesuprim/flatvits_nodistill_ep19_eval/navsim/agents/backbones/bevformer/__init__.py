"""BEVFormer-M front-end (SafeDrive-style) for DriveSuprim.

Camera-only, temporal BEV encoder: a ResNet image backbone + FPN neck feed
multi-view image features to a BEVFormer encoder whose learnable BEV queries
perform deformable spatial cross-attention using camera calibration, followed
by a temporal concat+conv fusion over several frames.
"""
from navsim.agents.backbones.bevformer.bevformer_backbone import BEVFormerM

__all__ = ["BEVFormerM"]
