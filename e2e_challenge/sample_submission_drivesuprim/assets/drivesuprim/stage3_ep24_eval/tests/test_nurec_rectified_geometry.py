"""The geometry the rectified logs assert, checked against the data itself.

Three claims carry the rectified tree, and none of them is self-evident:

1. ``T_sensor_rig`` in the USDZ is *sensor-to-rig*.  Read the other way round it
   still yields a valid rotation and a plausible-looking projection, so nothing
   downstream would complain -- the boxes would simply be wrong.
2. The NAVSIM ego frame *is* the NuRec rig frame, so the USDZ's sensor-to-rig
   needs no rebasing to become the logs' ``sensor2lidar``.
3. The rectified images carry ``TARGET_K``, which is the resolution
   DriveSuprim's BEV path reads, so ``_build_lidar2img``'s rescale is the
   identity and training and evaluation project through the same matrix.

Each is tested against the real dataset where the dataset is readable, and the
rewriting itself is tested on a synthetic log so it runs anywhere.
"""

import json
import os
import pickle
import re
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from navsim.planning.data.nurec_attach_images import (
    CAMERA_DIRECTORIES,
    NuRecImageMatchError,
    camera_indices_for,
    match_frames_to_images,
)
from navsim.planning.data.nurec_rectify.calibration import (
    CAMERA_SENSORS,
    ClipCalibration,
    read_usdz_calibration,
    usdz_path_for,
)
from navsim.planning.data.nurec_rectify.logs import NO_RENDER_PREFIX, rewrite_frames
from navsim.planning.data.nurec_rectify.target import (
    TARGET_DISTORTION,
    TARGET_HEIGHT,
    TARGET_K,
    TARGET_WIDTH,
    rectification_target_config,
)

# Resolved through ``setup_nurec.sh`` so the tests read the same data the
# pipeline does; the literals are the fallback for a bare shell.  Each of these
# is outside the tree -- the sample set and the challenge submission are not
# ours -- so every test that needs one skips when it is not readable.
_REPO = Path(__file__).resolve().parents[1]
USDZ_ROOT = Path(os.environ.get("NUREC_USDZ_ROOT", _REPO / "data" / "usdz"))
# The prepared root, not the source root: prepared_source now holds only maps
# and routes -- its logs carried the pre-rectification calibration and were
# archived, which is the whole point these tests exist to protect.
LOG_ROOT = Path(
    os.environ.get("NUREC_PREPARED_ROOT", _REPO / "data" / "prepared")
) / "navsim_logs" / "trainval"
DRIVER = Path(os.environ.get(
    "NUREC_CHALLENGE_DRIVER",
    "/rhome/junseong/alpasim/e2e_challenge/sample_submission_drivesuprim"
    "/drivesuprim_challenge/driver.py",
))
CLIP = "00040136-e651-4abd-991d-0655ccda9430"

needs_dataset = pytest.mark.skipif(
    not usdz_path_for(USDZ_ROOT, CLIP).is_file(), reason="the NuRec sample set is not readable here"
)
needs_logs = pytest.mark.skipif(
    not (LOG_ROOT / f"nurec-{CLIP}.pkl").is_file(), reason="the prepared logs are not readable here"
)
needs_driver = pytest.mark.skipif(
    not DRIVER.is_file(), reason="the challenge submission is not readable here"
)


@pytest.fixture(scope="module")
def calibration() -> ClipCalibration:
    return read_usdz_calibration(usdz_path_for(USDZ_ROOT, CLIP), CLIP)


# --------------------------------------------------------------------------
# 1. T_sensor_rig is sensor-to-rig
# --------------------------------------------------------------------------


@needs_dataset
@pytest.mark.parametrize(
    "camera, forward, lateral",
    [
        # The rig frame is x forward, y left, z up; the optical frame is
        # x right, y down, z forward.  Read as sensor-to-rig, the third column
        # is where the camera looks, in rig coordinates.
        ("CAM_F0", 0.99, 0.0),   # straight ahead
        ("CAM_L0", 0.2, 0.8),    # forward and to the left
        ("CAM_R0", 0.2, -0.8),   # forward and to the right
    ],
)
def test_sensor_to_rig_points_the_camera_where_its_name_says(calibration, camera, forward, lateral):
    """Transposed, every one of these flips sign or swaps left for right."""
    optical_axis = calibration[camera].rotation[:, 2]
    assert optical_axis[0] > forward, f"{camera} does not look forward: {optical_axis}"
    if lateral > 0:
        assert optical_axis[1] > lateral, f"{camera} does not look left: {optical_axis}"
    elif lateral < 0:
        assert optical_axis[1] < lateral, f"{camera} does not look right: {optical_axis}"
    else:
        assert abs(optical_axis[1]) < 0.1, f"{camera} is not centred: {optical_axis}"
    # The image's y axis points down, so in a level rig it is -z.
    assert calibration[camera].rotation[2, 1] < -0.95


@needs_dataset
def test_translation_is_the_camera_position_in_the_rig(calibration):
    """A camera on a car: ahead of the rig origin, above the road, roughly level."""
    for camera in CAMERA_SENSORS:
        x, y, z = calibration[camera].translation
        assert 1.0 < x < 4.0, f"{camera} sits at x={x}, which is not on the car"
        assert 0.5 < z < 2.5, f"{camera} sits at z={z}, which is not a camera height"
        assert abs(y) < 1.5
    # The cross cameras straddle the centreline; the wide camera is on it.
    assert calibration["CAM_L0"].translation[1] > 0.5
    assert calibration["CAM_R0"].translation[1] < -0.5
    assert abs(calibration["CAM_F0"].translation[1]) < 0.3


# --------------------------------------------------------------------------
# 2. The NAVSIM ego frame is the NuRec rig frame
# --------------------------------------------------------------------------


@needs_dataset
@needs_logs
def test_the_log_ego_pose_is_the_usdz_rig_pose():
    """``sensor2lidar`` needs no rebasing only because these are the same frame.

    The logs sample the trajectory at their own timestamps, so this compares at
    the two timestamps the USDZ hits exactly rather than interpolating.
    """
    with (LOG_ROOT / f"nurec-{CLIP}.pkl").open("rb") as stream:
        frames = pickle.load(stream)
    with zipfile.ZipFile(usdz_path_for(USDZ_ROOT, CLIP)) as archive:
        trajectory = json.loads(archive.read("rig_trajectories.json"))["rig_trajectories"][0]
    stamps = np.asarray(trajectory["T_rig_world_timestamps_us"])
    poses = np.asarray(trajectory["T_rig_worlds"])

    compared = 0
    for frame in frames:
        matches = np.flatnonzero(stamps == int(frame["timestamp"]))
        if not len(matches):
            continue
        compared += 1
        np.testing.assert_allclose(
            np.asarray(frame["ego2global_translation"], dtype=np.float64),
            poses[matches[0]][:3, 3],
            atol=1e-3,
            err_msg="the log's ego origin is not the USDZ's rig origin",
        )
    assert compared >= 2, "no log timestamp landed on a stored rig pose"


@needs_dataset
@needs_logs
def test_the_prepared_logs_put_the_lidar_at_the_ego_origin():
    """Only because ``lidar2ego`` is identity is sensor-to-ego also sensor-to-lidar."""
    with (LOG_ROOT / f"nurec-{CLIP}.pkl").open("rb") as stream:
        frames = pickle.load(stream)
    for frame in frames:
        np.testing.assert_allclose(np.asarray(frame["lidar2ego"], dtype=np.float64), np.eye(4), atol=1e-9)


# --------------------------------------------------------------------------
# 3. The target is the driver's, and training projects through the same matrix
# --------------------------------------------------------------------------


@needs_driver
def test_the_target_pinhole_is_the_drivers():
    """A drifted target would train on one camera and be scored on another.

    This fails while the driver still holds the old 412 x 366.22 pinhole.  That
    is the intended signal, not a broken test: training has moved to the
    isotropic 377 target and evaluation has not, which is exactly the state that
    must not reach a scored run.  Applying
    ``docs/nurec/DRIVER_K_PATCH.md`` clears it.
    """
    source = DRIVER.read_text()

    def constant(name: str) -> float:
        match = re.search(rf"^{name} = ([0-9.]+)$", source, re.MULTILINE)
        assert match, f"{name} is no longer a module constant in the driver"
        return float(match.group(1))

    config = rectification_target_config()
    hint = ("the driver and target.py disagree; apply "
            "docs/nurec/DRIVER_K_PATCH.md and rebuild the "
            "submission image")
    assert config.focal_length == (constant("_VIRTUAL_FX"), constant("_VIRTUAL_FY")), hint
    assert config.principal_point == (constant("_VIRTUAL_CX"), constant("_VIRTUAL_CY")), hint
    assert config.resolution_hw == (
        int(constant("_VIRTUAL_PINHOLE_HEIGHT")),
        int(constant("_VIRTUAL_PINHOLE_WIDTH")),
    )
    # The zero-valued tuples select the overscan path; empty ones would not.
    assert config.radial == (0.0,) * 6 and config.tangential == (0.0,) * 2
    assert config.thin_prism == (0.0,) * 4
    assert config.safety_margin_px == 4


def test_the_feature_builders_rescale_is_the_identity():
    """At 512x256 the model must project through exactly ``TARGET_K``."""
    from navsim.agents.drivesuprim.drivesuprim_config import DriveSuprimConfig
    from navsim.agents.drivesuprim.drivesuprim_features import DriveSuprimFeatureBuilder

    rotation = np.eye(3)
    translation = np.array([2.0, 0.1, 1.5])
    camera = SimpleNamespace(
        intrinsics=TARGET_K.copy(),
        sensor2lidar_rotation=rotation,
        sensor2lidar_translation=translation,
    )
    # A real builder, not a stand-in self: _build_lidar2img now checks that the
    # principal point lands inside the view it is scaling to, and that check is
    # part of what this test should exercise.
    builder = DriveSuprimFeatureBuilder(
        DriveSuprimConfig(training=False, bev_img_width=TARGET_WIDTH,
                          bev_img_height=TARGET_HEIGHT))
    built = builder._build_lidar2img(camera, TARGET_HEIGHT, TARGET_WIDTH, crop_top=0)

    sensor_to_ego = np.eye(4)
    sensor_to_ego[:3, :3] = rotation
    sensor_to_ego[:3, 3] = translation
    viewpad = np.eye(4)
    viewpad[:3, :3] = TARGET_K
    np.testing.assert_allclose(built, (viewpad @ np.linalg.inv(sensor_to_ego)).astype(np.float32),
                               rtol=1e-6, atol=1e-4)


@needs_dataset
def test_the_optical_axis_lands_on_the_principal_point(calibration):
    """The challenge driver's own session-start assertion, on our matrices.

    ``cnx_bev_bridge.build_camera_geometry`` refuses to start a session unless a
    point 10 m down the optical axis projects within 0.01 px of the principal
    point.  Ours have to clear the same bar.
    """
    for name in CAMERA_SENSORS:
        entry = calibration[name]
        sensor_to_ego = np.eye(4)
        sensor_to_ego[:3, :3] = entry.rotation
        sensor_to_ego[:3, 3] = entry.translation
        viewpad = np.eye(4)
        viewpad[:3, :3] = TARGET_K
        projection = viewpad @ np.linalg.inv(sensor_to_ego)

        point = np.append(entry.translation + 10.0 * entry.rotation[:, 2], 1.0)
        projected = projection @ point
        assert projected[2] > 0.0, f"{name}: the sanity point is behind the camera"
        error = np.linalg.norm(projected[:2] / projected[2] - TARGET_K[:2, 2])
        assert error < 1e-2, f"{name}: optical axis lands {error:.4f}px off the principal point"


# --------------------------------------------------------------------------
# The rewrite itself
# --------------------------------------------------------------------------

ALL_CAMERAS = ("CAM_F0", "CAM_L0", "CAM_L1", "CAM_L2", "CAM_R0", "CAM_R1", "CAM_R2", "CAM_B0")


def _synthetic_calibration() -> ClipCalibration:
    from navsim.planning.data.nurec_rectify.calibration import CameraCalibration

    cameras = {}
    for index, (name, sensor) in enumerate(CAMERA_SENSORS.items()):
        transform = np.eye(4)
        transform[:3, 3] = [2.0 + index, 0.5 * index, 1.5]
        cameras[name] = CameraCalibration(
            sensor_name=sensor, sensor_to_rig=transform, camera_model={"type": "ftheta"}
        )
    return ClipCalibration(clip_id="clip", cameras=cameras)


def _synthetic_log(frame_count: int, timestamps):
    return [
        {
            "log_name": "nurec-clip",
            "timestamp": timestamp,
            "anns": {"gt_boxes": []},
            "cams": {
                name: {
                    "data_path": f"old/{name}.jpg",
                    "cam_intrinsic": np.full((3, 3), 1545.0),
                    "distortion": np.full(5, 0.7),
                    "sensor2lidar_rotation": np.zeros((3, 3)),
                    "sensor2lidar_translation": np.zeros(3),
                }
                for name in ALL_CAMERAS
            },
        }
        for timestamp in timestamps
    ][:frame_count]


def _render_tree(root: Path, timestamps) -> Path:
    clip_root = root / "clip"
    (clip_root).mkdir(parents=True)
    (clip_root / "_complete").touch()
    for directory in CAMERA_DIRECTORIES.values():
        camera_root = clip_root / directory
        camera_root.mkdir()
        for timestamp in timestamps:
            (camera_root / f"{timestamp}.png").touch()
    return root


def test_rewrite_replaces_the_rendered_cameras_and_leaves_the_others(tmp_path):
    timestamps = [i * 500_000 for i in range(6)]
    image_root = _render_tree(tmp_path / "rectified", timestamps)
    frames = _synthetic_log(6, timestamps)

    stats = rewrite_frames(frames, _synthetic_calibration(), image_root, "clip",
                           trailing_frames_without_images=0)
    assert stats["frames_with_images"] == 6 and stats["images"] == 18

    calibration = _synthetic_calibration()
    for frame in frames:
        for name in CAMERA_SENSORS:
            camera = frame["cams"][name]
            assert camera["data_path"].endswith(f"{frame['timestamp']}.png")
            assert camera["data_path"].startswith("clip/")
            np.testing.assert_array_equal(camera["cam_intrinsic"], TARGET_K)
            np.testing.assert_array_equal(camera["distortion"], TARGET_DISTORTION)
            np.testing.assert_array_equal(camera["sensor2lidar_rotation"], calibration[name].rotation)
            np.testing.assert_array_equal(
                camera["sensor2lidar_translation"], calibration[name].translation
            )
        for name in ("CAM_L1", "CAM_L2", "CAM_R1", "CAM_R2", "CAM_B0"):
            untouched = frame["cams"][name]
            assert untouched["data_path"] == f"old/{name}.jpg"
            assert untouched["cam_intrinsic"][0, 0] == 1545.0


def test_a_frame_past_the_last_render_gets_a_sentinel_not_a_stale_path(tmp_path):
    """Frame 40 of a 41-frame log has no render; nothing may pretend it does."""
    timestamps = [i * 500_000 for i in range(6)]
    image_root = _render_tree(tmp_path / "rectified", timestamps[:5])
    frames = _synthetic_log(6, timestamps)

    stats = rewrite_frames(frames, _synthetic_calibration(), image_root, "clip",
                           trailing_frames_without_images=1)
    assert stats["frames_with_images"] == 5
    for name in CAMERA_SENSORS:
        assert frames[-1]["cams"][name]["data_path"].startswith(NO_RENDER_PREFIX)
        # The calibration is still correct even where the image is missing.
        np.testing.assert_array_equal(frames[-1]["cams"][name]["cam_intrinsic"], TARGET_K)


# --------------------------------------------------------------------------
# Which frames are allowed to have no image
# --------------------------------------------------------------------------


def _indices(tmp_path, timestamps):
    root = _render_tree(tmp_path, timestamps)
    return camera_indices_for("clip", root / "clip")


def test_a_missing_image_at_the_tail_is_tolerated(tmp_path):
    timestamps = [i * 500_000 for i in range(6)]
    indices = _indices(tmp_path, timestamps[:5])
    matches, _ = match_frames_to_images(
        _synthetic_log(6, timestamps), indices, "clip", 60_000, trailing_frames_without_images=1
    )
    assert [entry is None for entry in matches] == [False] * 5 + [True]


def test_a_missing_image_before_the_tail_still_rejects_the_clip(tmp_path):
    """NAVSIM would read this one, so it has to fail loudly."""
    timestamps = [i * 500_000 for i in range(6)]
    indices = _indices(tmp_path, timestamps[:1] + timestamps[3:])  # frames 1 and 2 unrendered
    with pytest.raises(NuRecImageMatchError, match="frame 1 of 6"):
        match_frames_to_images(
            _synthetic_log(6, timestamps), indices, "clip", 60_000, trailing_frames_without_images=1
        )


def test_the_tail_allowance_does_not_excuse_a_frame_outside_it(tmp_path):
    timestamps = [i * 500_000 for i in range(6)]
    indices = _indices(tmp_path, timestamps[:3])
    with pytest.raises(NuRecImageMatchError, match="frame 3 of 6"):
        match_frames_to_images(
            _synthetic_log(6, timestamps), indices, "clip", 60_000, trailing_frames_without_images=2
        )


# --------------------------------------------------------------------------
# The output is a prepared root, not just logs
# --------------------------------------------------------------------------


def _prepared_root(tmp_path: Path, timestamps) -> Path:
    """A miniature of what NUREC_PREPARED_ROOT names."""
    root = tmp_path / "prepared"
    logs = root / "navsim_logs" / "trainval"
    logs.mkdir(parents=True)
    for name in ("maps", "routes", "metric_cache"):
        (root / name).mkdir()
    (root / "maps" / "clip.pkl").touch()
    for clip in ("clip", "unrendered"):
        with (logs / f"nurec-{clip}.pkl").open("wb") as stream:
            frames = _synthetic_log(len(timestamps), timestamps)
            for frame in frames:
                frame["log_name"] = f"nurec-{clip}"
            pickle.dump(frames, stream)
    (root / "index.json").write_text(json.dumps({
        "format_version": 2,
        "logs": {"train": ["nurec-clip", "nurec-unrendered"], "val": ["nurec-unrendered"]},
        "num_logs": 2,
        "sensor_root": "/somewhere/unrectified",
        "map_root": str(root / "maps"),
    }))
    return root


def test_write_logs_produces_a_prepared_root_training_can_point_at(tmp_path, monkeypatch):
    """train.sh reads navsim_log_path and original_sensor_path out of index.json."""
    from navsim.planning.data.nurec_rectify import logs as logs_module

    timestamps = [i * 500_000 for i in range(6)]
    source = _prepared_root(tmp_path, timestamps)
    images = _render_tree(tmp_path / "rectified", timestamps)
    output = tmp_path / "prepared_rectified"

    calibration = _synthetic_calibration()
    monkeypatch.setattr(logs_module, "load_cache", lambda _: {"clip": calibration})

    summary = logs_module.write_logs(
        source, output, tmp_path / "calibration.json", images,
        trailing_frames_without_images=0,
    )
    assert summary["written_logs"] == 1  # 'unrendered' has no calibration

    index = json.loads((output / "index.json").read_text())
    assert index["sensor_root"] == str(images.resolve()), "training would read the un-rectified tree"
    assert index["rectified"] is True
    # A log that could not be rewritten must leave the splits, or training loads
    # a pickle that is not there.
    assert index["logs"] == {"train": ["nurec-clip"], "val": []}
    assert index["num_logs"] == 1

    for name in ("maps", "routes", "metric_cache"):
        link = output / name
        assert link.is_symlink() and link.resolve() == (source / name).resolve()

    assert (output / "navsim_logs" / "trainval" / "nurec-clip.pkl").is_file()
    assert not (output / "navsim_logs" / "trainval" / "nurec-unrendered.pkl").exists()


def test_write_logs_refuses_to_overwrite_the_source_root(tmp_path):
    from navsim.planning.data.nurec_rectify import logs as logs_module

    source = _prepared_root(tmp_path, [i * 500_000 for i in range(6)])
    with pytest.raises(ValueError, match="in place"):
        logs_module.write_logs(source, source, tmp_path / "c.json", tmp_path / "img")
