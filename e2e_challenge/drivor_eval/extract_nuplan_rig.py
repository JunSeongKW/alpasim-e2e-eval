# SPDX-License-Identifier: Apache-2.0
"""Canonical nuPlan camera rig that DrivoR was trained on, from NAVSIM logs.

    python extract_nuplan_rig.py --logs /home/kaist5/Dataset/navsim/download/navsim_logs/trainval \
        --out nuplan_rig.json

Reads the first frame of every log (every vehicle and calibration in trainval),
and per camera takes the element-wise median yaw / pitch / roll of the optical
axis in the ego (rear-axle) frame plus the median translation. Intrinsics and
distortion take a few per-vehicle values across nuPlan logs; the most frequent
(per camera-image count) is used and the full histogram is recorded.
The adapter re-projects PAI renders into these orientations; the network does
not read cam_K / world_2_cam (configs/models/INPUT_CONTRACT.json), so only the
image appearance has to match.
"""

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np

CAMS = ["CAM_F0", "CAM_B0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2"]
# OpenCV camera axes (x right, y down, z forward) expressed in the ego frame
# (x forward, y left, z up) for a camera looking straight ahead.
BASE = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def rot(axis, a):
    c, s = np.cos(a), np.sin(a)
    if axis == "z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def to_ypr(R):
    """cam->ego rotation = Rz(yaw) @ Ry(pitch) @ Rx(roll) @ BASE."""
    M = R @ BASE.T
    yaw = np.arctan2(M[1, 0], M[0, 0])
    pitch = np.arcsin(-M[2, 0])
    roll = np.arctan2(M[2, 1], M[2, 2])
    return np.array([yaw, pitch, roll])


def from_ypr(ypr):
    yaw, pitch, roll = ypr
    return rot("z", yaw) @ rot("y", pitch) @ rot("x", roll) @ BASE


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ypr = {c: [] for c in CAMS}
    trans = {c: [] for c in CAMS}
    calib, logs = Counter(), sorted(a.logs.glob("*.pkl"))
    for path in logs:
        frame = pickle.load(path.open("rb"))[0]
        assert np.allclose(frame["lidar2ego_translation"], 0) and \
            np.allclose(frame["lidar2ego_rotation"], [1, 0, 0, 0]), path
        for c in CAMS:
            cam = frame["cams"][c]
            R = np.asarray(cam["sensor2lidar_rotation"], dtype=np.float64)
            assert np.allclose(from_ypr(to_ypr(R)), R, atol=1e-6), (path, c)
            ypr[c].append(to_ypr(R))
            trans[c].append(np.asarray(cam["sensor2lidar_translation"], dtype=np.float64))
            calib[(tuple(np.round(np.asarray(cam["cam_intrinsic"]).ravel(), 4)),
                   tuple(np.round(np.asarray(cam["distortion"]).ravel(), 6)))] += 1
    (K, dist), _ = calib.most_common(1)[0]
    total = sum(calib.values())
    out = {"source": str(a.logs), "n_logs": len(logs), "resolution_wh": [1920, 1080],
           "K": np.array(K).reshape(3, 3).tolist(), "distortion_k1k2p1p2k3": list(dist),
           "calibration_histogram": [{"fx_fy_cx_cy": [k[0], k[4], k[2], k[5]], "distortion": list(d),
                                      "share": round(n / total, 4)} for (k, d), n in calib.most_common()],
           "cameras": {}}
    for row in out["calibration_histogram"]:
        print("calib", row)
    for c in CAMS:
        y = np.array(ypr[c])
        med = np.median(y, axis=0)
        out["cameras"][c] = {
            "yaw_pitch_roll_deg": np.degrees(med).round(4).tolist(),
            "yaw_pitch_roll_deg_p05_p95": np.degrees(np.percentile(y, [5, 95], axis=0)).round(3).tolist(),
            "cam_to_ego_rotation": from_ypr(med).tolist(),
            "translation_m": np.median(np.array(trans[c]), axis=0).round(4).tolist(),
        }
        print(c, "ypr", out["cameras"][c]["yaw_pitch_roll_deg"], "t", out["cameras"][c]["translation_m"])
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
