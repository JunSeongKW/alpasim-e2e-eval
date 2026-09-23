"""Deployment stand-in for the training tree's rig-pose helper.

`drivesuprim_features.py` imports `tilt_for` at module scope, but only calls it
from `_get_bev_camera_feature` and only when `bev_pitch_correct` is set -- the
BEV path's correction for the log pose being gravity-aligned while
`sensor2lidar_*` tilts with the car.

The flat ViT variant has no BEV projection, so that path is never entered, and
the challenge driver builds its own feature tensors rather than going through
DriveSuprimFeatureBuilder at all. The import still has to resolve, hence this
file.

It raises rather than returning identity on purpose. Returning "no tilt" would
let a BEV model deploy with the correction silently disabled and score like a
miscalibrated camera; raising says plainly that the real module -- which reads
per-log rig poses that ship with the training data -- was not part of the
handoff bundle.
"""

from __future__ import annotations


def tilt_for(log_name: str, timestamp: int):
    """Rotation reconciling the gravity-aligned log pose with the rig frame."""

    raise NotImplementedError(
        "navsim.planning.data.nurec_rig_pose.tilt_for is a deployment stand-in: "
        "the real implementation was not included in the model bundle. It is "
        "only needed by the BEV pitch-correction path (bev_pitch_correct), "
        f"which the flat ViT variant does not use. Called with log_name="
        f"{log_name!r}, timestamp={timestamp!r}."
    )
