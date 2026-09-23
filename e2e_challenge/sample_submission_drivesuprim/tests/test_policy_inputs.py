from __future__ import annotations

import cv2
import numpy as np
import torch
from alpasim_grpc.v0 import common_pb2, sensorsim_pb2

from drivesuprim_challenge.policy import (
    DriveSuprimPolicy,
    _extract_teacher_state,
)


def test_fast_teacher_preprocessing_matches_released_operations() -> None:
    rng = np.random.default_rng(7)
    images = {
        role: rng.integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8)
        for role in ("left", "front", "right")
    }
    policy = DriveSuprimPolicy.__new__(DriveSuprimPolicy)
    policy.camera_width = 2048
    policy.camera_height = 512

    result = policy._build_model_inputs(
        images=images,
        command=1,
        velocity_xy=(2.0, 3.0),
        acceleration_xy=(4.0, 5.0),
    )

    stitched = np.concatenate(
        [
            images["left"][28:-28, 416:-416],
            images["front"][28:-28],
            images["right"][28:-28, 416:-416],
        ],
        axis=1,
    )
    resized = cv2.resize(stitched, (2048, 512))
    expected_camera = (
        torch.from_numpy(np.ascontiguousarray(resized))
        .permute(2, 0, 1)
        .float()
        .div(255.0)
    )

    torch.testing.assert_close(result["camera_feature"], expected_camera)
    torch.testing.assert_close(
        result["status_feature"],
        torch.tensor([1, 0, 0, 0, 2, 3, 4, 5], dtype=torch.float32),
    )


def test_extract_teacher_state_drops_training_prefix_and_student() -> None:
    teacher = torch.tensor([1.0])
    checkpoint = {
        "agent.model.teacher.model.layer.weight": teacher,
        "agent.model.student.model.layer.weight": torch.tensor([2.0]),
    }

    assert _extract_teacher_state(checkpoint) == {"layer.weight": teacher}


def _pinhole_camera(logical_id: str, tx: float = 0.0):
    camera = sensorsim_pb2.AvailableCamerasReturn.AvailableCamera(
        logical_id=logical_id,
        rig_to_camera=common_pb2.Pose(
            vec=common_pb2.Vec3(x=tx), quat=common_pb2.Quat(w=1.0)
        ),
    )
    camera.intrinsics.resolution_h = 100
    camera.intrinsics.resolution_w = 200
    camera.intrinsics.opencv_pinhole_param.CopyFrom(
        sensorsim_pb2.OpenCVPinholeCameraParam(
            focal_length_x=100.0,
            focal_length_y=100.0,
            principal_point_x=100.0,
            principal_point_y=50.0,
        )
    )
    return camera


def test_bev_adapter_builds_temporal_multiview_contract() -> None:
    policy = DriveSuprimPolicy.__new__(DriveSuprimPolicy)
    policy.bev_img_width = 20
    policy.bev_img_height = 10
    policy.bev_seq_len = 3
    policy._bev_rectifiers = {}
    camera_map = {"left": "L", "front": "F", "right": "R"}
    camera_specs = {
        "L": _pinhole_camera("L", tx=-1.0),
        "F": _pinhole_camera("F"),
        "R": _pinhole_camera("R", tx=1.0),
    }
    image = np.zeros((100, 200, 3), dtype=np.uint8)

    result = policy._build_bev_model_inputs(
        {
            "bev_history": [
                {
                    "images": {role: image for role in camera_map},
                    "pose": common_pb2.Pose(
                        vec=common_pb2.Vec3(x=2.0),
                        quat=common_pb2.Quat(w=1.0),
                    ),
                }
            ],
            "camera_map": camera_map,
            "camera_specs": camera_specs,
            "command": 1,
            "velocity_xy": (2.0, 3.0),
            "acceleration_xy": (4.0, 5.0),
        }
    )

    assert result["bev_imgs"].shape == (3, 3, 3, 10, 20)
    assert result["lidar2img"].shape == (3, 3, 4, 4)
    assert result["bev_ego_pose"].shape == (3, 3)
    torch.testing.assert_close(result["bev_ego_pose"], torch.zeros(3, 3))
    torch.testing.assert_close(
        result["status_feature"],
        torch.tensor([1, 0, 0, 0, 2, 3, 4, 5], dtype=torch.float32),
    )

