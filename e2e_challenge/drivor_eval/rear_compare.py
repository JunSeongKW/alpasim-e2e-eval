# SPDX-License-Identifier: Apache-2.0
"""cam_b0 built two ways next to the raw rear pair (drivor-eval env).

    python rear_compare.py CASE_DIR OUT.png

CASE_DIR holds session_request.json and step_t<us>/<logical_id>.jpg folders
(frames pulled from a rollout.asl). The rotation-only stitch puts both rear
cameras at one centre; the proxy stitch is what pai_views uses. On frames
before a rear-end collision (clipgt-0026708c, 5cam run 2026-10-07) the
follower is ghosted by the former and solid in the latter.
"""
import sys, json
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pai_views as pv
case = Path(sys.argv[1]); out = sys.argv[2]
req = json.loads((case / "session_request.json").read_text())
specs = {c["logical_id"]: c for c in req["rollout_spec"]["vehicle"]["available_cameras"]}
rig = pv.NuplanRig(); rays = rig.pixel_rays()
src = {c: pv.FthetaSource.from_spec(specs[c], (1080, 1920)) for c in pv.SOURCES["cam_b0"]}
def rotation_only(images):
    rays_ego = rays @ rig.rotation("cam_b0").T
    projs = [src[n].project(rays_ego @ src[n].cam_to_rig) for n in pv.SOURCES["cam_b0"]]
    acc = np.zeros((1080, 1920, 3), np.float32); wsum = np.zeros((1080, 1920), np.float32)
    import cv2
    for n, (pix, valid, edge, th) in zip(pv.SOURCES["cam_b0"], projs):
        px = np.where(valid[..., None], pix, -1).astype(np.float32)
        img = cv2.remap(images[n], px[..., 0], px[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        w = pv._border_weight(pix, valid, (1080, 1920)).astype(np.float32)
        acc += img * w[..., None]; wsum += w
    return np.clip(acc / np.maximum(wsum, 1e-6)[..., None], 0, 255).astype(np.uint8)
steps = sorted(case.glob("step_t*"))
fig, axes = plt.subplots(len(steps), 4, figsize=(22, 3.3 * len(steps)))
for r, st in enumerate(steps):
    images = {c: np.asarray(Image.open(st / f"{c}.jpg").convert("RGB")) for c in pv.NEEDED_CAMERAS}
    views = pv.build_view_maps({c: pv.FthetaSource.from_spec(specs[c], (1080, 1920)) for c in pv.NEEDED_CAMERAS}, rig, rays)
    proxy = pv.render(views["cam_b0"], images)
    panels = [(images["camera_rear_right_70fov"], "rear_right (raw)"), (images["camera_rear_left_70fov"], "rear_left (raw)"),
              (rotation_only(images), "B0 rotation-only stitch"), (proxy, f"B0 proxy stitch (ground + {pv.PROXY_RADIUS_M:.0f} m cylinder)")]
    for c, (img, title) in enumerate(panels):
        ax = axes[r, c]; ax.imshow(img); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=11)
        if c == 0: ax.set_ylabel(st.name.replace("step_t", "t="), fontsize=9)
fig.tight_layout(); fig.savefig(out, dpi=65); print("wrote", out)
