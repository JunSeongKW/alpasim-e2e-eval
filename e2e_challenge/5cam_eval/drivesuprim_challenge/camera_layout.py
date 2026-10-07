# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""Physical camera order used by the stage3_5cam_rear56 checkpoint."""

CAMERA_ORDER = ("CAM_L1", "CAM_L0", "CAM_F0", "CAM_R0", "CAM_R1")

CAMERA_ALIASES = {
    **{name: name for name in CAMERA_ORDER},
    "CAMERA_REAR_LEFT_70FOV": "CAM_L1",
    "CAMERA_CROSS_LEFT_120FOV": "CAM_L0",
    "CAMERA_FRONT_WIDE_120FOV": "CAM_F0",
    "CAMERA_CROSS_RIGHT_120FOV": "CAM_R0",
    "CAMERA_REAR_RIGHT_70FOV": "CAM_R1",
}
