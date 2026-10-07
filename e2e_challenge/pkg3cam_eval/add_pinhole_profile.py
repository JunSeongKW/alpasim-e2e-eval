# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 NVIDIA Corporation
"""Add a higher-resolution virtual pinhole to a copied driver.py.

    python add_pinhole_profile.py <driver.py> K565P5_768X384

The img768 bundle trains with ``bev_ori_image_mode: raw_fisheye``: every view is
rendered from the raw f-theta image as a pinhole that keeps the camera's own
pose and uses the 512x256 rectification target K (fx=fy=377, cx=256, cy=128)
scaled to bev_img_* -- drivesuprim_features._scaled_bev_intrinsics, then
NuRecRawFisheyeVirtualizer.render_native_views. At 768x384 that is
K = (565.5, 565.5, 384, 192).

The driver already rectifies f-theta -> pinhole with one target and builds each
camera's lidar2img from the same K, so the profile only has to carry the K and
the output size. Without it the policy would upscale a 512x256 view while
keeping the 512x256 lidar2img, which is neither the trained pixels nor a
consistent projection.
"""

import sys
from pathlib import Path

PROFILES = {
    # name: (fx, fy, cx, cy, width, height)
    "K565P5_768X384": (565.5, 565.5, 384.0, 192.0, 768, 384),
}

path, name = Path(sys.argv[1]), sys.argv[2]
fx, fy, cx, cy, width, height = PROFILES[name]
text = path.read_text()

anchor = '    "K377_512X256": (377.0, 377.0, 256.0, 128.0),\n'
assert text.count(anchor) == 1, "profile table not found"
text = text.replace(anchor, anchor + f'    "{name}": ({fx}, {fy}, {cx}, {cy}),\n')

anchor = ") = _PINHOLE_PROFILES[_PINHOLE_PROFILE]\n"
assert text.count(anchor) == 1, "profile unpack not found"
text = text.replace(
    anchor,
    anchor
    + "\n# Output size follows the profile (added for the img768 bundle).\n"
    + f'_PINHOLE_SIZES = {{"{name}": ({width}, {height})}}\n'
    + "_VIRTUAL_PINHOLE_WIDTH, _VIRTUAL_PINHOLE_HEIGHT = _PINHOLE_SIZES.get(\n"
    + "    _PINHOLE_PROFILE, (_VIRTUAL_PINHOLE_WIDTH, _VIRTUAL_PINHOLE_HEIGHT)\n"
    + ")\n",
)
path.write_text(text)
print(f"added {name}: K=({fx}, {fy}, {cx}, {cy}) size={width}x{height} -> {path}")
