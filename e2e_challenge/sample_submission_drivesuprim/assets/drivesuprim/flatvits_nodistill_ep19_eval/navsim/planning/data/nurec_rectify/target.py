"""The pinhole both training and the challenge driver rectify to.

``drivesuprim_agent_bevformer_*`` resolves to ``backbone_type: bevformer_m``,
which reads a 512x256 per-camera view with a ``lidar2img`` per camera.  Training
and the challenge driver must rectify to the *same* pinhole, or the model is
trained on one camera and scored on another -- so these four numbers exist in
two places and ``tests/test_nurec_rectified_geometry.py`` checks they agree.

The target is isotropic: ``fx == fy``, principal point at the image centre.
That is a deliberate change from the values this file used to hold, which were
nuPlan's pinhole (fx=fy=1545, cx=960, cy=560 at 1920x1080) carried through
DriveSuprim's 512x256 resize -- 412 x 366.22.  Two things were wrong with them,
both measured on this dataset:

* **Anisotropy.**  512x256 is 2:1 while 1920x1080 is 1.78:1, so the resize
  squashed the image vertically by 12.5%: a circle in the world came out an
  ellipse of ratio 1.125.  On NAVSIM this was unavoidable -- the camera is real
  and fx=fy=1545 is fixed, and SafeDrive's ``bev_crop_top_bottom: 28`` was the
  only lever, taking the squash from 12.5% to 6.7%.  Rectification removes the
  constraint entirely: fx and fy are free, so the squash goes to zero.  fy=377
  keeps SafeDrive's vertical framing (37.5 deg against its 36.7 deg).
* **A blind wedge between the cameras.**  The cross cameras sit ~67 deg off
  forward (measured across all 1,607 clips: p50 67.0, p99 68.4, max 70.2), and
  fx=412 gives each view only 63.7 deg -- leaving 3.4 deg that no camera sees,
  2.9 m wide at 40 m.  fx=377 gives 68.4 deg and closes it for 99% of clips.

Lowering fx costs 8.5% of horizontal object size against the old target; fy
*rises* 2.9%, so this is un-squashing rather than a net loss.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from navsim.planning.data.nurec_rectify.schema import RectificationTargetConfig

TARGET_WIDTH = 512
TARGET_HEIGHT = 256

TARGET_FX = 377.0
TARGET_FY = 377.0
TARGET_CX = 256.0
TARGET_CY = 128.0

TARGET_RESOLUTION_WH: Tuple[int, int] = (TARGET_WIDTH, TARGET_HEIGHT)

#: ``cam_intrinsic`` for a rectified view.
TARGET_K = np.array(
    [[TARGET_FX, 0.0, TARGET_CX], [0.0, TARGET_FY, TARGET_CY], [0.0, 0.0, 1.0]],
    dtype=np.float64,
)

#: ``distortion`` for a rectified view.  The lens is already gone; the five
#: zeros keep the field the shape NAVSIM's ``Camera`` and ``bev_undistort``
#: expect rather than dropping it.
TARGET_DISTORTION = np.zeros(5, dtype=np.float64)


def rectification_target_config() -> RectificationTargetConfig:
    """The driver's target, tuple-for-tuple.

    The zero-valued distortion tuples are not decoration: ``_has_distortion``
    treats a present-but-zero tuple as distortion, which turns on the overscan
    path and makes the rectifier build a 528x264 canvas and crop it back.  With
    zero coefficients that crop is exact, but dropping the tuples would change
    which code path runs -- and the driver runs this one.
    """
    return RectificationTargetConfig(
        focal_length=(TARGET_FX, TARGET_FY),
        principal_point=(TARGET_CX, TARGET_CY),
        resolution_hw=(TARGET_HEIGHT, TARGET_WIDTH),
        radial=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        tangential=(0.0, 0.0),
        thin_prism=(0.0, 0.0, 0.0, 0.0),
        max_overscan_scale=2.0,
        safety_margin_px=4,
    )
