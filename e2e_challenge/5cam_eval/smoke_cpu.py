# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""Load the shipped image exactly and execute a real five-camera CPU forward.

Also check the failures a mere checkpoint load misses: camera order, incomplete
or unsynchronized frames, live lens rectification, and rear extrinsics.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
from drivesuprim_challenge.camera_layout import CAMERA_ALIASES, CAMERA_ORDER
from drivesuprim_challenge.driver import (
    _VIRTUAL_K,
    DriveSuprimChallengeDriver,
    SessionState,
    _build_virtual_lidar2img,
    _rectification_target_config,
    common_pb2,
    egodriver_pb2,
    sensorsim_pb2,
)
from drivesuprim_challenge.policy import DriveSuprimPolicy
from navsim.planning.data.nurec_raw_fisheye_virtual import (
    NuRecRawFisheyeVirtualizer,
    _scale_model_to_image,
    virtual_camera_geometry,
)
from PIL import Image
from scipy.spatial.transform import Rotation
from vavam_challenge.rectification import build_ftheta_rectifier_for_resolution


def camera_proto(entry):
    params = entry["camera_model"]["parameters"]
    w, h = params["resolution"]
    spec = sensorsim_pb2.CameraSpec(
        logical_id=entry["logical_sensor_name"], resolution_w=w, resolution_h=h
    )
    ft = spec.ftheta_param
    ft.principal_point_x, ft.principal_point_y = params["principal_point"]
    ft.max_angle = params["max_angle"]
    ft.reference_poly = getattr(
        sensorsim_pb2.FthetaCameraParam, params["reference_poly"]
    )
    ft.angle_to_pixeldist_poly.extend(params["angle_to_pixeldist_poly"])
    ft.pixeldist_to_angle_poly.extend(params["pixeldist_to_angle_poly"])
    c, d, e = params.get("linear_cde") or [1, 0, 0]
    ft.linear_cde.linear_c, ft.linear_cde.linear_d, ft.linear_cde.linear_e = c, d, e
    matrix = np.asarray(entry["T_sensor_rig"], dtype=np.float64)
    qx, qy, qz, qw = Rotation.from_matrix(matrix[:3, :3]).as_quat()
    x, y, z = matrix[:3, 3]
    return sensorsim_pb2.AvailableCamerasReturn.AvailableCamera(
        logical_id=entry["logical_sensor_name"],
        intrinsics=spec,
        rig_to_camera=common_pb2.Pose(
            vec=common_pb2.Vec3(x=x, y=y, z=z),
            quat=common_pb2.Quat(x=qx, y=qy, z=qz, w=qw),
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    fixture = json.loads(args.fixture.read_text())
    bbox = fixture["rig_bbox"]
    length, width, height = bbox["dim"]
    vehicle = egodriver_pb2.DriveSessionRequest.RolloutSpec.VehicleDefinition(
        bounding_box=common_pb2.AABB(size_x=length, size_y=width, size_z=height),
        rig_to_bounding_box=common_pb2.Pose(
            vec=common_pb2.Vec3(x=bbox["centroid"][0]), quat=common_pb2.Quat(w=1)
        ),
    )
    ego_geom = DriveSuprimChallengeDriver._ego_geometry_from_vehicle(vehicle)
    np.testing.assert_allclose(
        ego_geom, [length * 0.98, width * 0.98, bbox["centroid"][0]]
    )
    images, projections, rectification = {}, {}, {}
    for index, name in enumerate(CAMERA_ORDER):
        entry = fixture["cameras"][name]
        assert CAMERA_ALIASES[entry["logical_sensor_name"].upper()] == name
        camera = camera_proto(entry)
        matrix = np.asarray(entry["T_sensor_rig"], dtype=np.float32)
        _, reference_projection = virtual_camera_geometry(
            matrix[:3, :3], matrix[:3, 3], _VIRTUAL_K, 0.0
        )
        projections[name] = _build_virtual_lidar2img(camera)
        np.testing.assert_allclose(projections[name], reference_projection, atol=0.002)
        # Exercise scene-dependent decoded widths, including 1916 vs 1920.
        for width in (1916, 1920):
            yy, xx = np.indices((1080, width))
            raw = np.stack((xx % 251, yy % 251, (xx + yy) % 251), axis=-1).astype(
                np.uint8
            )
            rectifier = build_ftheta_rectifier_for_resolution(
                camera_proto=camera,
                target_cfg=_rectification_target_config(),
                source_resolution_hw=(1080, width),
            )
            actual = rectifier.rectify(raw)
            virtualizer = NuRecRawFisheyeVirtualizer(
                raw_root=None,
                usdz_root=None,
                width=512,
                height=256,
                intrinsics=_VIRTUAL_K,
            )
            sources = {
                name: {
                    "image": raw,
                    "rotation": matrix[:3, :3],
                    "yaw": 0.0,
                    "model": _scale_model_to_image(
                        entry["camera_model"]["parameters"], raw.shape
                    ),
                },
            }
            reference = virtualizer._render_virtual(
                sources, 0.0, matrix[:3, :3], _VIRTUAL_K, prefer_index=0
            )
            mae = float(np.abs(actual.astype(float) - reference.astype(float)).mean())
            if mae > 0.1:
                raise RuntimeError(
                    f"{name}/{width}: training/live rectification MAE={mae}"
                )
            rectification[f"{name}/{width}"] = mae
        # Constant camera-specific pixels make an order swap detectable.
        images[name] = np.full((256, 512, 3), 20 * (index + 1), dtype=np.uint8)

    session = SessionState()
    for name in CAMERA_ORDER[:-1]:
        session.current_images[name] = images[name]
        session.current_frame_times[name] = 1_000_000
        DriveSuprimChallengeDriver._commit_camera_frame_locked(None, session)
        assert not session.camera_history, "Frame committed without both rear cameras"
    session.current_images[CAMERA_ORDER[-1]] = images[CAMERA_ORDER[-1]]
    session.current_frame_times[CAMERA_ORDER[-1]] = 1_100_000
    DriveSuprimChallengeDriver._commit_camera_frame_locked(None, session)
    assert not session.camera_history, "Frame committed despite >50ms camera skew"
    session.current_frame_times[CAMERA_ORDER[-1]] = 1_000_000
    DriveSuprimChallengeDriver._commit_camera_frame_locked(None, session)
    assert len(session.camera_history) == 1
    assert set(session.camera_history[0].images) == set(CAMERA_ORDER)

    # Exercise the actual protobuf session and image callbacks, including the
    # live rear rectifiers. This is the path the simulator calls.
    vehicle.available_cameras.extend(
        camera_proto(fixture["cameras"][name]) for name in CAMERA_ORDER
    )
    driver = DriveSuprimChallengeDriver(
        policy_handle=None, inference_interval_us=100_000
    )
    request = egodriver_pb2.DriveSessionRequest(
        session_uuid="five-camera-cpu-smoke",
        rollout_spec=egodriver_pb2.DriveSessionRequest.RolloutSpec(vehicle=vehicle),
    )
    driver.start_session(request, None)
    live_session = driver._sessions[request.session_uuid]
    np.testing.assert_allclose(live_session.ego_geom, ego_geom)
    for index, name in enumerate(CAMERA_ORDER):
        buffer = BytesIO()
        Image.fromarray(np.full((1080, 1920, 3), 20 * (index + 1), np.uint8)).save(
            buffer, format="PNG"
        )
        driver.submit_image_observation(
            egodriver_pb2.RolloutCameraImage(
                session_uuid=request.session_uuid,
                camera_image=egodriver_pb2.RolloutCameraImage.CameraImage(
                    logical_id=fixture["cameras"][name]["logical_sensor_name"],
                    frame_start_us=1_000_000,
                    frame_end_us=1_030_000,
                    image_bytes=buffer.getvalue(),
                ),
            ),
            None,
        )
        assert len(live_session.camera_history) == (1 if index == 4 else 0)
    images = live_session.camera_history[0].images
    projections = live_session.lidar2img_by_camera

    checkpoint = Path(os.environ["DRIVESUPRIM_CHECKPOINT_PATH"])
    assert checkpoint.name == "stage3_5cam_ep05.ckpt"
    policy = DriveSuprimPolicy(
        checkpoint_path=str(checkpoint),
        vocab_path=os.environ["DRIVESUPRIM_VOCAB_PATH"],
        config_path=os.environ["DRIVESUPRIM_CONFIG_PATH"],
        backbone_type=os.environ["DRIVESUPRIM_BACKBONE_TYPE"],
        device=args.device,
    )
    assert policy._config.bev_num_cameras == 5
    assert (policy._config.bev_h, policy._config.bev_w) == (56, 112)
    assert policy._config.use_route and not policy._use_autocast
    observed = {}
    real_run = policy._run_agent

    def capture(features, **kwargs):
        observed.update(
            {
                key: list(features[key].shape)
                for key in (
                    "bev_imgs",
                    "lidar2img",
                    "bev_ego_pose",
                    "route_feature",
                    "ego_geom",
                )
            }
        )
        assert observed["bev_imgs"] == [1, 3, 5, 3, 256, 512]
        assert observed["lidar2img"] == [1, 3, 5, 4, 4]
        np.testing.assert_allclose(features["ego_geom"].cpu().numpy(), ego_geom[None])
        np.testing.assert_allclose(
            features["bev_imgs"][0, 0, :, 0, 128, 256].cpu().numpy(),
            np.arange(1, 6) * 20 / 255,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            features["lidar2img"][0, 0].cpu().numpy(),
            np.stack([projections[n] for n in CAMERA_ORDER]),
        )
        return real_run(features, **kwargs)

    policy._run_agent = capture
    kwargs = {
        "lidar2img_by_camera": projections,
        "bev_ego_pose": np.zeros((3, 3), np.float32),
        "route_waypoints": np.column_stack((np.linspace(0, 80, 20), np.zeros((20, 2)))),
        "ego_geom": ego_geom,
    }
    try:
        policy.predict(
            [{n: images[n] for n in CAMERA_ORDER[:-1]}],
            [np.zeros(8, np.float32)],
            **kwargs,
        )
    except ValueError as error:
        assert "missing BEV camera CAM_R1" in str(error)
    else:
        raise AssertionError("Missing rear camera was accepted")
    prediction = policy.predict([images] * 3, [np.zeros(8, np.float32)] * 2, **kwargs)
    assert prediction.poses.shape == (40, 3), prediction.poses.shape
    assert np.isfinite(prediction.poses).all()
    report = {
        "passed": True,
        "device": args.device,
        "deformable_attention": "original PyTorch fallback / axe-v9 env",
        "clipgt_id": fixture["clipgt_id"],
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "camera_order": CAMERA_ORDER,
        "input_shapes": observed,
        "strict_checkpoint": "missing=0 unexpected=0",
        "rear_frame_sync_and_missing_camera_checks": "passed",
        "protobuf_session_and_five_live_image_callbacks": "passed",
        "training_vs_live_rectification_mae": rectification,
        "output_shape": list(prediction.poses.shape),
        "output_poses": prediction.poses.tolist(),
        "ego_geom_from_api": ego_geom.tolist(),
        "elapsed_seconds": time.monotonic() - started,
    }
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"PASS: strict checkpoint, live calibration, five-camera inputs, real {args.device} forward"
    )


if __name__ == "__main__":
    main()
