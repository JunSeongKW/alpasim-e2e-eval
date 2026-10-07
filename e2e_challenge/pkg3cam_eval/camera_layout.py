# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""Physical camera order of the three-camera AXEv1.0-NuRec stage-3 bundles.

Replaces the five-camera layout of 5cam_eval/drivesuprim_challenge, so the
same adapter (per-session ego_geom, axe-v9 operator settings) drives the
three-camera checkpoints. Order and aliases are axe-v9's.
"""

CAMERA_ORDER = ("CAM_L0", "CAM_F0", "CAM_R0")

CAMERA_ALIASES = {
    **{name: name for name in CAMERA_ORDER},
    "CAMERA_CROSS_LEFT_120FOV": "CAM_L0",
    "CAMERA_FRONT_WIDE_120FOV": "CAM_F0",
    "CAMERA_CROSS_RIGHT_120FOV": "CAM_R0",
}
