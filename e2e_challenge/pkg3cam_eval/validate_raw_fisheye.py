# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""Does the driver feed the img768 model the pixels and projection it trained on?

Runs inside the axe-v9 image with the bundle's navsim and the prepared driver
package mounted. For one real clip it takes the raw f-theta frames and the rig
calibration out of the usdz, then renders every camera twice:

  training  NuRecRawFisheyeVirtualizer (bev_ori_image_mode=raw_fisheye): the
            camera's own pose, K = 512x256 target K scaled to bev_img_*, own
            camera preferred, INTER_LINEAR remap
  driver    the challenge driver's f-theta -> pinhole rectifier and
            _build_virtual_lidar2img with the selected pinhole profile, fed an
            AvailableCamera proto built from the same calibration

and compares pixels (where both are valid) and the 4x4 projections. Run at
K377_512X256 as the control (the path every evaluated 512 model already uses)
and at K565P5_768X384 for the img768 bundle.
"""

import argparse
import importlib
import io
import json
import os
import sys
import zipfile
from collections import OrderedDict

import cv2
import numpy as np

sys.path.insert(0, "/app")

PROFILES = {"K377_512X256": (377.0, 256.0, 128.0, 512, 256),
            "K565P5_768X384": (565.5, 384.0, 192.0, 768, 384)}
CAMS = (("cam_l0", "camera_cross_left_120fov", "CAMERA_CROSS_LEFT_120FOV"),
        ("cam_f0", "camera_front_wide_120fov", "CAMERA_FRONT_WIDE_120FOV"),
        ("cam_r0", "camera_cross_right_120fov", "CAMERA_CROSS_RIGHT_120FOV"))


def quat_from_matrix(R):
    R = np.asarray(R, dtype=np.float64)
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w > 1e-6:
        return w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)
    i = int(np.argmax(np.diag(R)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = np.sqrt(max(0.0, 1.0 + R[i, i] - R[j, j] - R[k, k])) * 2.0
    q = [0.0, 0.0, 0.0]
    q[i] = s / 4.0
    q[j] = (R[j, i] + R[i, j]) / s
    q[k] = (R[k, i] + R[i, k]) / s
    return (R[k, j] - R[j, k]) / s, q[0], q[1], q[2]


def build_proto(sensorsim_pb2, logical_id, entry):
    cam = sensorsim_pb2.AvailableCamerasReturn.AvailableCamera()
    cam.logical_id = logical_id
    p = entry["camera_model"]["parameters"]
    intr = cam.intrinsics
    intr.resolution_w, intr.resolution_h = int(p["resolution"][0]), int(p["resolution"][1])
    f = intr.ftheta_param
    f.principal_point_x, f.principal_point_y = map(float, p["principal_point"])
    enum = f.DESCRIPTOR.fields_by_name["reference_poly"].enum_type
    f.reference_poly = enum.values_by_name[p["reference_poly"]].number
    f.pixeldist_to_angle_poly.extend(map(float, p["pixeldist_to_angle_poly"]))
    f.angle_to_pixeldist_poly.extend(map(float, p["angle_to_pixeldist_poly"]))
    f.max_angle = float(p["max_angle"])
    c, d, e = p.get("linear_cde") or (1.0, 0.0, 0.0)
    f.linear_cde.linear_c, f.linear_cde.linear_d, f.linear_cde.linear_e = float(c), float(d), float(e)
    T = np.asarray(entry["T_sensor_rig"], dtype=np.float64)
    cam.rig_to_camera.vec.x, cam.rig_to_camera.vec.y, cam.rig_to_camera.vec.z = map(float, T[:3, 3])
    w, x, y, z = quat_from_matrix(T[:3, :3])
    q = cam.rig_to_camera.quat
    q.w, q.x, q.y, q.z = float(w), float(x), float(y), float(z)
    return cam


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--usdz", required=True)
    ap.add_argument("--profile", required=True, choices=sorted(PROFILES))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ["DRIVESUPRIM_PINHOLE_PROFILE"] = a.profile
    V = importlib.import_module("navsim.planning.data.nurec_raw_fisheye_virtual")
    from alpasim_grpc.v0 import sensorsim_pb2
    D = importlib.import_module("drivesuprim_challenge.driver")
    fx, cx, cy, W, H = PROFILES[a.profile]
    assert (D._VIRTUAL_PINHOLE_WIDTH, D._VIRTUAL_PINHOLE_HEIGHT) == (W, H)
    K = np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1.0]])

    with zipfile.ZipFile(a.usdz) as z:
        rig = json.loads(z.read("rig_trajectories.json"))
        cal = {e["logical_sensor_name"]: e for e in rig["camera_calibrations"].values()}
        frames = {}
        for _, sensor, _ in CAMS:
            names = sorted(n for n in z.namelist() if n.startswith(f"frames/{sensor}/"))
            raw = np.frombuffer(z.read(names[len(names) // 2]), np.uint8)
            frames[sensor] = cv2.cvtColor(cv2.imdecode(raw, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    sources = OrderedDict()
    for name, sensor, _ in CAMS:
        T = np.asarray(cal[sensor]["T_sensor_rig"], dtype=np.float32)
        img = frames[sensor]
        sources[name] = {"image": img, "rotation": T[:3, :3], "translation": T[:3, 3],
                         "model": V._scale_model_to_image(cal[sensor]["camera_model"]["parameters"], img.shape),
                         "yaw": V._yaw_deg(T[:3, :3])}
    v = V.NuRecRawFisheyeVirtualizer.__new__(V.NuRecRawFisheyeVirtualizer)
    v.width, v.height, v.intrinsics, v._map_cache = W, H, K, OrderedDict()

    report = {"profile": a.profile, "usdz": os.path.basename(a.usdz), "cameras": {}}
    order = list(sources)
    for name, sensor, logical in CAMS:
        T = np.asarray(cal[sensor]["T_sensor_rig"], dtype=np.float64)
        rot_v, proj_train = V.virtual_camera_geometry(T[:3, :3], T[:3, 3], K, 0.0, None)
        img_train = v._render_virtual(sources, 0.0, rot_v, K, "clip", order.index(name))
        proto = build_proto(sensorsim_pb2, logical.lower(), cal[sensor])
        img = frames[sensor]
        rect = D.build_ftheta_rectifier_for_resolution(
            camera_proto=proto, target_cfg=D._rectification_target_config(),
            source_resolution_hw=(int(img.shape[0]), int(img.shape[1])))
        img_drv = np.asarray(rect.rectify(img), dtype=np.uint8)
        proj_drv = D._build_virtual_lidar2img(proto).astype(np.float64)
        both = (img_train.sum(-1) > 0) & (img_drv.sum(-1) > 0)
        diff = np.abs(img_train.astype(np.float64) - img_drv.astype(np.float64))[both]
        mse = float((diff ** 2).mean()) if diff.size else float("nan")
        rel = np.abs(proj_train - proj_drv).max() / np.abs(proj_train).max()
        report["cameras"][name] = {
            "source_hw": list(img.shape[:2]), "shape_train": list(img_train.shape),
            "shape_driver": list(img_drv.shape), "valid_both_frac": float(both.mean()),
            "valid_train_only": int(((img_train.sum(-1) > 0) & ~both).sum()),
            "valid_driver_only": int(((img_drv.sum(-1) > 0) & ~both).sum()),
            "mae": float(diff.mean()) if diff.size else None,
            "psnr_db": None if mse == 0 else 10 * np.log10(255.0 ** 2 / mse),
            "pixels_off_by_gt2": float((diff.max(axis=-1) > 2).mean()) if diff.size else None,
            "projection_max_rel_diff": float(rel),
        }
        cv2.imwrite(os.path.join(a.out, f"{a.profile}_{name}_train.png"), cv2.cvtColor(img_train, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(a.out, f"{a.profile}_{name}_driver.png"), cv2.cvtColor(img_drv, cv2.COLOR_RGB2BGR))
    path = os.path.join(a.out, f"raw_fisheye_{a.profile}.json")
    open(path, "w").write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
