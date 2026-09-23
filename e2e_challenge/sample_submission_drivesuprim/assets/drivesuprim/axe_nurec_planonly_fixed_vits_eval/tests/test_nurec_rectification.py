"""The vendored rectifier must stay the challenge driver's rectifier.

Training images are rectified here; evaluation rectifies them again inside the
AlpaSim driver. If the two ever drift, the model is trained on one geometry and
scored on another -- and the difference would be invisible in any per-image
check, because both outputs look like plausible pinhole images. So this pins the
copy to its source: the file must differ only in its import, and the two must
agree pixel for pixel on a real camera model.
"""

import importlib.util
import os
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from navsim.planning.data.nurec_rectify import ftheta_proto
from navsim.planning.data.nurec_rectify.rectification import (
    build_ftheta_rectifier_for_resolution,
)
from navsim.planning.data.nurec_rectify.schema import RectificationTargetConfig

UPSTREAM = Path(os.environ.get(
    "NUREC_CHALLENGE_RECTIFICATION",
    "/rhome/junseong/alpasim/e2e_challenge/sample_submission_drivesuprim"
    "/drivesuprim_challenge/rectification.py",
))
VENDORED = Path("navsim/planning/data/nurec_rectify/rectification.py")

# The target the challenge driver hardcodes: DriveSuprim's BEV input resolution,
# reached by rectifying at 1920x1080 and resizing the way its feature builder does.
TARGET = RectificationTargetConfig(
    focal_length=(412.0, 366.22222222222223),
    principal_point=(256.0, 132.74074074074073),
    resolution_hw=(256, 512),
    radial=(0.0,) * 6,
    tangential=(0.0,) * 2,
    thin_prism=(0.0,) * 4,
)

# A real f-theta model, copied out of one clip's USDZ so the test needs no dataset.
CAMERA_MODEL = {
    "type": "ftheta",
    "parameters": {
        "resolution": [1920, 1080],
        "principal_point": [961.3438110351562, 744.7864990234375],
        "pixeldist_to_angle_poly": [
            0.0, 0.0010651813354343176, 4.1703902553535954e-09,
            -9.677027408561134e-12, 4.7017894270898544e-14, -1.3423182578045114e-17,
        ],
        "angle_to_pixeldist_poly": [
            0.0, 938.5548706054688, -1.3443444967269897,
            2.3548173904418945, -30.547151565551758, 10.295219421386719,
        ],
        "max_angle": 1.352455637370406,
        "linear_cde": [1.0, 0.0, 0.0],
    },
}


def _strip_comments(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def _load_upstream():
    """Import the original module with our stand-in registered as its proto package."""
    package = types.ModuleType("alpasim_grpc")
    v0 = types.ModuleType("alpasim_grpc.v0")
    v0.sensorsim_pb2 = ftheta_proto
    package.v0 = v0
    schema = types.ModuleType("upstream_schema")
    schema.RectificationTargetConfig = RectificationTargetConfig
    saved = {k: sys.modules.get(k) for k in ("alpasim_grpc", "alpasim_grpc.v0")}
    sys.modules["alpasim_grpc"] = package
    sys.modules["alpasim_grpc.v0"] = v0
    try:
        source = UPSTREAM.read_text().replace(
            "from .schema import RectificationTargetConfig",
            "from navsim.planning.data.nurec_rectify.schema import RectificationTargetConfig",
        )
        module = types.ModuleType("upstream_rectification")
        exec(compile(source, str(UPSTREAM), "exec"), module.__dict__)
        return module
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


upstream_available = pytest.mark.skipif(
    not UPSTREAM.is_file(), reason="the challenge submission is not readable here"
)


def _camera():
    return ftheta_proto.available_camera_from_usdz("camera_front_wide_120fov", CAMERA_MODEL)


def _rectifier(builder):
    camera = _camera()
    return builder(
        camera_proto=camera,
        target_cfg=TARGET,
        source_resolution_hw=(camera.intrinsics.resolution_h, camera.intrinsics.resolution_w),
    )


def _test_image(seed: int = 0) -> np.ndarray:
    """Structured noise -- flat colour would hide a remap that is subtly off."""
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, size=(1080, 1920, 3), dtype=np.uint8)
    image[::37, :, :] = 255
    image[:, ::53, :] = 0
    return image


@upstream_available
def test_the_copy_differs_from_its_source_only_in_the_import():
    ours = _strip_comments(VENDORED.read_text())
    theirs = _strip_comments(UPSTREAM.read_text())
    ours = ours.replace(
        "from navsim.planning.data.nurec_rectify import ftheta_proto as sensorsim_pb2\n"
        "from navsim.planning.data.nurec_rectify.schema import RectificationTargetConfig",
        "from alpasim_grpc.v0 import sensorsim_pb2\n\nfrom .schema import RectificationTargetConfig",
    )
    normalise = lambda text: re.sub(r"\n{2,}", "\n\n", text).strip()
    assert normalise(ours) == normalise(theirs), (
        "the vendored rectifier has drifted from the challenge driver's copy")


@upstream_available
def test_rectified_pixels_match_the_challenge_driver():
    theirs = _rectifier(_load_upstream().build_ftheta_rectifier_for_resolution)
    ours = _rectifier(build_ftheta_rectifier_for_resolution)
    image = _test_image()
    np.testing.assert_array_equal(
        np.asarray(ours.rectify(image)), np.asarray(theirs.rectify(image)),
        err_msg="vendored rectification differs from the challenge driver's")


def test_the_target_is_the_resolution_the_model_reads():
    out = np.asarray(_rectifier(build_ftheta_rectifier_for_resolution).rectify(_test_image()))
    assert out.shape == (256, 512, 3)
    assert out.dtype == np.uint8


def test_the_whole_target_view_is_covered():
    """A target wider than the lens would leave unfilled borders and go unnoticed."""
    out = np.asarray(_rectifier(build_ftheta_rectifier_for_resolution).rectify(
        np.full((1080, 1920, 3), 200, dtype=np.uint8)))
    unfilled = (out.sum(axis=2) == 0).mean()
    assert unfilled == 0.0, f"{unfilled:.1%} of the rectified view has no source pixels"


def test_absent_linear_cde_is_reported_absent():
    """protobuf presence semantics: reading a submessage must not create it."""
    param = ftheta_proto.FthetaCameraParam()
    assert not param.HasField("linear_cde")
    assert param.linear_cde.linear_c == 1.0
    assert not param.HasField("linear_cde")
    param.linear_cde.linear_c = 2.0
    assert param.HasField("linear_cde")

    copy = ftheta_proto.FthetaCameraParam()
    copy.CopyFrom(param)
    assert copy.HasField("linear_cde") and copy.linear_cde.linear_c == 2.0
    assert not ftheta_proto.FthetaCameraParam().HasField("linear_cde")
