"""The calibration guard: a pinhole's principal point is inside its own image.

A prepared root can carry logs whose cam_intrinsic predates rectification
(nuPlan's, cx 960 cy 560 for 1920x1080) while the files beside it are the
rectified 512x256 views. Every BEV reference point then projects outside the
image, SpatialCrossAttention returns its residual, and the encoder trains on
zero image content without a single warning -- measured bev_mask 0.000% across
all 3,136 queries on every frame of a 30-epoch run.
"""

import numpy as np
import pytest

from navsim.agents.drivesuprim.drivesuprim_config import DriveSuprimConfig
from navsim.agents.drivesuprim.drivesuprim_features import DriveSuprimFeatureBuilder


class _Cam:
    def __init__(self, K):
        self.intrinsics = np.asarray(K, dtype=np.float64)
        self.sensor2lidar_rotation = np.eye(3)
        self.sensor2lidar_translation = np.array([1.7, 0.0, 1.5])
        self.camera_path = "CAM_F0/x.jpg"


def _builder():
    return DriveSuprimFeatureBuilder(DriveSuprimConfig(training=False))


def _K(fx, cx, cy):
    return [[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]]


def test_rectified_intrinsics_pass():
    """TARGET_K: fx 377, principal point at the centre of 512x256."""
    b = _builder()
    out = b._build_lidar2img(_Cam(_K(377.0, 256.0, 128.0)), orig_h=256, orig_w=512)
    assert out.shape == (4, 4)


def test_the_nuplan_pinhole_against_rectified_files_is_refused():
    """The exact pair that cost the 30-epoch run."""
    b = _builder()
    with pytest.raises(ValueError, match="does not belong to this image"):
        b._build_lidar2img(_Cam(_K(1545.0, 960.0, 560.0)), orig_h=256, orig_w=512)


def test_the_message_names_the_two_env_vars_to_check():
    b = _builder()
    with pytest.raises(ValueError) as e:
        b._build_lidar2img(_Cam(_K(1545.0, 960.0, 560.0)), orig_h=256, orig_w=512)
    assert "NUREC_PREPARED_ROOT" in str(e.value) and "NUREC_SENSOR_ROOT" in str(e.value)


def test_intrinsics_for_the_pre_resize_image_still_pass_after_scaling():
    """A log may legitimately hold the full-res pinhole; the resize fixes it."""
    b = _builder()
    cfg = b._config
    # 1920x1080 nuPlan pinhole, with the files stored at that resolution
    out = b._build_lidar2img(_Cam(_K(1545.0, 960.0, 560.0)), orig_h=1080, orig_w=1920)
    assert out.shape == (4, 4)
    scaled_cx = 960.0 * cfg.bev_img_width / 1920.0
    assert 0.0 < scaled_cx < cfg.bev_img_width


def test_a_principal_point_just_outside_is_refused():
    b = _builder()
    w = b._config.bev_img_width
    with pytest.raises(ValueError):
        b._build_lidar2img(_Cam(_K(377.0, float(w) + 1.0, 128.0)), orig_h=256, orig_w=512)


def test_crop_top_is_applied_before_the_check():
    """bev_crop_top_bottom shifts cy; the guard must see the shifted value."""
    b = DriveSuprimFeatureBuilder(DriveSuprimConfig(training=False))
    h = b._config.bev_img_height
    # cy sits inside only because the crop pulls it back in
    cam = _Cam(_K(377.0, 256.0, float(h) + 20.0))
    with pytest.raises(ValueError):
        b._build_lidar2img(cam, orig_h=256, orig_w=512, crop_top=0)
    out = b._build_lidar2img(cam, orig_h=256, orig_w=512, crop_top=40)
    assert out.shape == (4, 4)


def test_it_is_its_own_exception_type():
    """DatasetSSL's retry wrapper keys on the type to fail fast."""
    from navsim.agents.drivesuprim.drivesuprim_features import CalibrationMismatch
    b = _builder()
    with pytest.raises(CalibrationMismatch):
        b._build_lidar2img(_Cam(_K(1545.0, 960.0, 560.0)), orig_h=256, orig_w=512)
    assert issubclass(CalibrationMismatch, ValueError)


def test_the_retry_wrapper_does_not_swallow_it():
    """Eight retries on a systematic error only bury the message."""
    import navsim.planning.training.dataset_ssl as ds_mod
    from navsim.agents.drivesuprim.drivesuprim_features import CalibrationMismatch

    class _Loader:
        tokens = ["a", "b", "c"]

    calls = []

    class _DS(ds_mod.DatasetSSL):
        def __init__(self):
            self._scene_loader = _Loader()

        def _getitem_impl(self, idx):
            calls.append(idx)
            raise CalibrationMismatch("principal point outside the view")

    with pytest.raises(CalibrationMismatch):
        _DS().__getitem__(0)
    assert calls == [0], f"retried a systematic failure: {calls}"
