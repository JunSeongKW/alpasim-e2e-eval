"""The rectification target, lifted from the challenge submission's schema.

Only this one dataclass is needed; the rest of that module pulls in the whole
driver configuration.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RectificationTargetConfig:
    """Target pinhole parameters for rectifying a rendered camera."""

    focal_length: tuple[float, float]
    principal_point: tuple[float, float]
    resolution_hw: tuple[int, int]
    radial: tuple[float, ...] = ()
    tangential: tuple[float, ...] = ()
    thin_prism: tuple[float, ...] = ()

    # We rectify a larger canvas and only crop in the end to allow for
    # margin when applying the distortion of the pinhole camera.
    max_overscan_scale: float = 2.0
    safety_margin_px: int = 10
