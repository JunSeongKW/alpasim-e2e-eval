# SPDX-License-Identifier: Apache-2.0
"""Unit checks of the PAI -> nuPlan mapping and the route command (drivor-eval env).

    $ENV/bin/python -m pytest e2e_challenge/drivor_eval/test_pai_views.py -q
"""

import math
from types import SimpleNamespace

import numpy as np
import pytest

import pai_views as pv
from drivor_driver import COMMAND_THRESHOLD_M, LEFT, RIGHT, STRAIGHT, UNKNOWN, command_from_route


def ftheta_spec(name, cam_to_rig, position=(1.5, 0.0, 1.5), hw=(1080, 1920)):
    """A plain equidistant f-theta camera (r = 946 * theta), pose given as a matrix."""
    R = np.asarray(cam_to_rig)
    w = math.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    if w > 1e-6:
        q = (w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w))
    else:  # 180 deg rotations
        x = math.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
        q = (0.0, x, (R[0, 1] + R[1, 0]) / (4 * x), (R[0, 2] + R[2, 0]) / (4 * x))
    return {"logical_id": name,
            "intrinsics": {"resolution_h": hw[0], "resolution_w": hw[1],
                           "ftheta_param": {"principal_point_x": 960.0, "principal_point_y": 540.0,
                                            "reference_poly": "ANGLE_TO_PIXELDIST",
                                            "angle_to_pixeldist_poly": [0.0, 946.0],
                                            "pixeldist_to_angle_poly": [0.0, 1 / 946.0],
                                            "max_angle": 1.35}},
            "rig_to_camera": {"vec": dict(zip("xyz", position)), "quat": dict(zip("wxyz", q))}}


def test_front_quaternion_is_forward_opencv_base():
    # camera_front_wide_120fov of clipgt-6e6cbe67 (rig_to_camera quaternion, w x y z)
    R = pv.quat_wxyz_to_matrix(-0.4962693, 0.5017115, -0.49946183, 0.50253385)
    assert np.allclose(R, [[0, 0, 1], [-1, 0, 0], [0, -1, 0]], atol=0.02)


def test_rotation_only_maps_principal_point_to_principal_point():
    rig = pv.NuplanRig()
    rays = rig.pixel_rays()
    # a source looking exactly where each target looks: the target principal
    # point (undistorted ray (0, 0, 1)) must land on the source principal point
    specs = {}
    for target, names in pv.SOURCES.items():
        for name in names:
            specs[name] = ftheta_spec(name, rig.rotation(target), position=rig.position(target))
    sources = {c: pv.FthetaSource.from_spec(specs[c], (1080, 1920)) for c in pv.NEEDED_CAMERAS}
    for target in ("cam_f0", "cam_l0", "cam_r0"):
        ray = rays[560, 960] @ rig.rotation(target).T
        src = sources[pv.SOURCES[target][0]]
        pix, valid, _, _ = src.project((ray @ src.cam_to_rig)[None])
        assert valid[0] and np.allclose(pix[0], [960, 540], atol=0.5), (target, pix)


def test_rendered_size_scaling_matches_official_rule():
    spec = ftheta_spec("camera_front_wide_120fov", np.eye(3))
    half = pv.FthetaSource.from_spec(spec, (540, 960))
    assert np.allclose(half.principal_point, [480, 270])
    assert np.allclose(half.angle_to_pixeldist, [0.0, 473.0])


def test_pixel_rays_undistort_round_trip():
    rig = pv.NuplanRig()
    rays = rig.pixel_rays()
    assert rays.shape == (1080, 1920, 3)
    assert np.allclose(np.linalg.norm(rays, axis=-1), 1.0)
    # barrel distortion: the corner ray is wider than the pinhole 63.7 deg HFOV implies
    assert math.degrees(math.atan2(-rays[540, 0, 0], rays[540, 0, 2])) > 33.0


@pytest.mark.parametrize("ys,expected", [
    ([], UNKNOWN), ([0.0], STRAIGHT), ([COMMAND_THRESHOLD_M + 0.1], LEFT),
    ([-(COMMAND_THRESHOLD_M + 0.1)], RIGHT), ([COMMAND_THRESHOLD_M], STRAIGHT)])
def test_command_rule(ys, expected):
    wps = [SimpleNamespace(x=42.1, y=y) for y in ys]
    assert command_from_route(wps) == expected


def test_command_rule_skips_near_waypoints():
    wps = [SimpleNamespace(x=5.0, y=10.0), SimpleNamespace(x=42.1, y=0.0)]
    assert command_from_route(wps) == STRAIGHT
