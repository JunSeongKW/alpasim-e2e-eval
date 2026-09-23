from __future__ import annotations

import numpy as np

from drivesuprim_challenge import driver


def test_driver_uses_k377_training_geometry() -> None:
    target = driver._rectification_target_config()

    assert target.resolution_hw == (256, 512)
    assert target.focal_length == (377.0, 377.0)
    assert target.principal_point == (256.0, 128.0)
    np.testing.assert_array_equal(
        driver._VIRTUAL_K,
        np.array(
            [
                [377.0, 0.0, 256.0],
                [0.0, 377.0, 128.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )


def test_rectifier_is_built_from_decoded_frame_resolution(
    monkeypatch,
) -> None:
    calls = []

    def fake_build(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(
        driver,
        "build_ftheta_rectifier_for_resolution",
        fake_build,
    )

    service = driver.DriveSuprimChallengeDriver(
        policy_handle=object(),
        inference_interval_us=100_000,
    )
    session = driver.SessionState(
        camera_specs={"CAM_F0": object()},
        rectifiers={"CAM_F0": None},
    )

    first = service._rectifier_for_image(
        session,
        "CAM_F0",
        source_resolution_hw=(1080, 1920),
    )
    cached = service._rectifier_for_image(
        session,
        "CAM_F0",
        source_resolution_hw=(1080, 1920),
    )

    assert first is cached
    assert len(calls) == 1
    assert calls[0]["source_resolution_hw"] == (1080, 1920)

    rebuilt = service._rectifier_for_image(
        session,
        "CAM_F0",
        source_resolution_hw=(1080, 1900),
    )

    assert rebuilt is not first
    assert len(calls) == 2
    assert calls[1]["source_resolution_hw"] == (1080, 1900)

