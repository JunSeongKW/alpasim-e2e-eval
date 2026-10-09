# SPDX-License-Identifier: Apache-2.0
"""PAI f-theta renders -> the nuPlan pinhole views DrivoR was trained on.

DrivoR reads four 1920x1080 nuPlan images [cam_f0, cam_b0, cam_l0, cam_r0]
(distorted, fx=fy=1545, pp=(960, 560), OpenCV k1 k2 p1 p2 k3) and ignores the
cam_K / world_2_cam it is given (configs/models/INPUT_CONTRACT.json): the view
identity lives in per-camera learned registers, so the adapter has to deliver
images that look like those cameras. This module builds, once per session, a
cv2.remap table per target view from the session's own calibration
(available_cameras), so every clip uses its vehicle's real extrinsics.

  cam_f0 <- camera_front_wide_120fov   rotation-only re-projection
  cam_l0 <- camera_cross_left_120fov   rotation-only re-projection
  cam_r0 <- camera_cross_right_120fov  rotation-only re-projection
  cam_b0 <- camera_rear_left_70fov + camera_rear_right_70fov   proxy stitch

Rotation-only is what the official driver's FthetaToPinholeRectifier does for
VaVAM (src/driver/src/alpasim_driver/rectification.py, configs/driver/vavam.yaml),
extended by the rotation between the PAI camera and the nuPlan camera. It is
exact for distant content; the camera position difference (PAI mounts are
lower) is not modelled.

No PAI camera faces straight back. rear_left/right sit ~1 m either side of the
car looking back-out at about +-152 deg, so their views meet behind the car
from 2 m apart. A rotation-only stitch puts both at one centre, and an object
straight behind then lands on the far side of the seam in each camera: a car
following in-lane vanishes from the stitched image at every distance, because
the 1 m camera offset exceeds the car's half width. cam_b0 is therefore built
on proxy geometry around the nuPlan B0 position: the ground plane (z = 0 in
the rear-axle rig frame) out to PROXY_RADIUS_M, beyond that a vertical
cylinder of that radius. Content near the proxy (a follower ~15 m back, the
road surface) stitches without a seam; other depths shift by
baseline x (1/d - 1/R). Where both cameras see a pixel the two are blended
with border-distance weights. Pixels no camera sees are filled from the
nearest source edge (see build_view_maps), in every view.

Frames: rig / ego = rear axle, x forward, y left, z up (both datasets).
Camera = OpenCV (x right, y down, z forward). PAI rig_to_camera quaternions
(w, x, y, z) are the camera->rig rotation (front_wide gives the forward
OpenCV base exactly), nuPlan sensor2lidar_rotation is camera->ego.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
# A priori proxy distance for the rear stitch (not tuned on any scored scene):
# a follower at 1.5 s headway and 36 km/h.
PROXY_RADIUS_M = 15.0
FEATHER_PX = 48.0

SOURCES = {
    "cam_f0": ("camera_front_wide_120fov",),
    "cam_l0": ("camera_cross_left_120fov",),
    "cam_r0": ("camera_cross_right_120fov",),
    "cam_b0": ("camera_rear_left_70fov", "camera_rear_right_70fov"),
}
TARGET_ORDER = ("cam_f0", "cam_b0", "cam_l0", "cam_r0")
NEEDED_CAMERAS = tuple(sorted({s for v in SOURCES.values() for s in v}))


def quat_wxyz_to_matrix(w: float, x: float, y: float, z: float) -> np.ndarray:
    q = np.array([w, x, y, z], dtype=np.float64)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


@dataclass
class FthetaSource:
    """One PAI camera: f-theta intrinsics at the rendered size + pose in the rig."""

    name: str
    angle_to_pixeldist: np.ndarray
    principal_point: np.ndarray
    linear: np.ndarray
    max_angle: Optional[float]
    resolution_hw: Tuple[int, int]
    cam_to_rig: np.ndarray
    position: np.ndarray

    @classmethod
    def from_spec(cls, spec: dict, rendered_hw: Tuple[int, int]) -> "FthetaSource":
        """`spec` is an AvailableCamera as a dict (MessageToDict, field names kept).

        The intrinsics are given at the native resolution; scale them to the
        rendered one exactly as rectification._scale_ftheta_intrinsics_to_resolution
        does (radial terms by the y scale, the x/y ratio through linear C/D).
        """
        intr = spec["intrinsics"]
        ft = intr["ftheta_param"]
        # Like the official rectifier, project with the forward (angle ->
        # pixel distance) polynomial whatever reference_poly says; some clips
        # mark the backward one as reference but every clip carries both, so
        # check that they agree instead.
        forward = np.asarray(ft.get("angle_to_pixeldist_poly", []), dtype=np.float64)
        backward = np.asarray(ft.get("pixeldist_to_angle_poly", []), dtype=np.float64)
        if forward.size < 2:
            raise ValueError(f"{spec['logical_id']}: no angle_to_pixeldist polynomial")
        if backward.size >= 2 and float(ft.get("max_angle", 0.0)) > 0:
            theta = np.linspace(0.0, float(ft["max_angle"]), 64)
            back = np.polynomial.polynomial.polyval(np.polynomial.polynomial.polyval(theta, forward), backward)
            if np.abs(back - theta).max() > np.radians(0.5):
                raise ValueError(f"{spec['logical_id']}: f-theta polynomials disagree by "
                                 f"{np.degrees(np.abs(back - theta).max()):.2f} deg")
        native_h, native_w = int(intr["resolution_h"]), int(intr["resolution_w"])
        h, w = rendered_hw
        sx, sy = w / native_w, h / native_h
        cde = ft.get("linear_cde", {})
        c, d, e = cde.get("linear_c", 1.0), cde.get("linear_d", 0.0), cde.get("linear_e", 0.0)
        ratio = sx / sy
        pose = spec["rig_to_camera"]
        q = pose["quat"]
        v = pose["vec"]
        max_angle = float(ft.get("max_angle", 0.0))
        return cls(
            name=spec["logical_id"],
            angle_to_pixeldist=forward * sy,
            principal_point=np.array([ft["principal_point_x"] * sx, ft["principal_point_y"] * sy]),
            linear=np.array([[c * ratio, d * ratio], [e, 1.0]]),
            max_angle=max_angle if max_angle > 0 else None,
            resolution_hw=(h, w),
            cam_to_rig=quat_wxyz_to_matrix(q.get("w", 0.0), q.get("x", 0.0), q.get("y", 0.0), q.get("z", 0.0)),
            position=np.array([v.get("x", 0.0), v.get("y", 0.0), v.get("z", 0.0)]),
        )

    def project(self, rays_cam: np.ndarray):
        """Rays (..., 3) in this camera's frame -> pixels, valid, edge pixels, theta.

        pixels/valid follow rectification._FthetaCamera.ray_to_pixel. The edge
        pixel is the in-image pixel nearest along the ray's image direction
        (theta capped at max_angle, then clipped to the frame); it fills target
        pixels no camera sees. theta is the angle from the optical axis.
        """
        flat = rays_cam.reshape(-1, 3)
        xy, z = flat[:, :2], flat[:, 2]
        r_xy = np.linalg.norm(xy, axis=1)
        theta = np.arctan2(r_xy, z)
        front = z > 0
        within = np.ones_like(front) if self.max_angle is None else theta <= self.max_angle + 1e-6
        unit = np.divide(xy, r_xy[:, None], out=np.zeros_like(xy), where=r_xy[:, None] > 1e-12)
        radius = np.polynomial.polynomial.polyval(theta, self.angle_to_pixeldist)
        pix = (unit * radius[:, None]) @ self.linear.T + self.principal_point
        h, w = self.resolution_hw
        inside = (pix[:, 0] >= -0.5) & (pix[:, 0] <= w - 0.5) & (pix[:, 1] >= -0.5) & (pix[:, 1] <= h - 0.5)
        valid = front & within & inside
        cap = np.minimum(theta, self.max_angle if self.max_angle is not None else np.pi / 2 - 1e-3)
        edge = (unit * np.polynomial.polynomial.polyval(cap, self.angle_to_pixeldist)[:, None]) @ self.linear.T \
            + self.principal_point
        edge = np.stack([np.clip(edge[:, 0], 0, w - 1), np.clip(edge[:, 1], 0, h - 1)], 1)
        shape = rays_cam.shape[:-1]
        return pix.reshape(shape + (2,)), valid.reshape(shape), edge.reshape(shape + (2,)), theta.reshape(shape)


class NuplanRig:
    """Canonical nuPlan target cameras (nuplan_rig.json, from NAVSIM trainval logs)."""

    def __init__(self, path: Path = HERE / "nuplan_rig.json"):
        d = json.loads(Path(path).read_text())
        self.K = np.array(d["K"], dtype=np.float64)
        self.dist = np.array(d["distortion_k1k2p1p2k3"], dtype=np.float64)
        self.wh = tuple(d["resolution_wh"])
        self.cams = {k.lower(): v for k, v in d["cameras"].items()}

    def rotation(self, cam: str) -> np.ndarray:
        return np.array(self.cams[cam]["cam_to_ego_rotation"], dtype=np.float64)

    def position(self, cam: str) -> np.ndarray:
        return np.array(self.cams[cam]["translation_m"], dtype=np.float64)

    def pixel_rays(self) -> np.ndarray:
        """Unit rays (H, W, 3) in the camera frame for every *distorted* pixel.

        cv2.undistortPointsIter with a tight criterion: nuPlan's k1 = -0.356 is
        strong enough that the default 5 iterations leave corner residuals.
        """
        w, h = self.wh
        u, v = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
        pts = np.stack([u, v], -1).reshape(-1, 1, 2)
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 200, 1e-12)
        und = cv2.undistortPointsIter(pts, self.K, self.dist, None, None, crit).reshape(-1, 2)
        # Round-trip check: re-distort and compare with the pixel grid.
        back, _ = cv2.projectPoints(np.concatenate([und, np.ones((len(und), 1))], 1),
                                    np.zeros(3), np.zeros(3), self.K, self.dist)
        err = np.abs(back.reshape(-1, 2) - pts.reshape(-1, 2)).max()
        if err > 0.05:
            raise RuntimeError(f"nuPlan undistortion did not converge (max {err:.3f} px)")
        rays = np.concatenate([und, np.ones((len(und), 1))], 1)
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        return rays.reshape(h, w, 3)


@dataclass
class ViewMap:
    target: str
    sources: List[str]
    maps: List[Tuple[np.ndarray, np.ndarray]]   # per source cv2.remap tables
    weights: Optional[List[np.ndarray]]          # per source (H, W, 1) float32, None = single source
    coverage: float                              # share of target pixels some source really sees
    edge_filled: float                           # share filled from the nearest source edge


def _border_weight(pix: np.ndarray, valid: np.ndarray, hw: Tuple[int, int]) -> np.ndarray:
    h, w = hw
    d = np.minimum.reduce([pix[..., 0] + 0.5, w - 0.5 - pix[..., 0], pix[..., 1] + 0.5, h - 0.5 - pix[..., 1]])
    return np.where(valid, np.clip(d / FEATHER_PX, 1e-3, 1.0), 0.0)


def build_view_maps(sources: Dict[str, FthetaSource], rig: NuplanRig,
                    rays: Optional[np.ndarray] = None) -> Dict[str, ViewMap]:
    """Per target view: remap tables (and blend weights for the rear stitch).

    Target pixels no source sees (B0's wedge right behind the bumper and top
    corners, the bottom rows of F0 on some rigs) take the nearest in-frame
    pixel of the source whose optical axis is closest, i.e. edge replication,
    rather than black: a black wedge is a hard synthetic edge the network never
    saw in training. Their share is reported as edge_filled.
    """
    rays = rig.pixel_rays() if rays is None else rays
    out = {}
    for target in TARGET_ORDER:
        names = SOURCES[target]
        rays_ego = rays @ rig.rotation(target).T
        if len(names) == 1:
            # rotation-only: rays_ego @ R applies R^T row-wise (rig -> camera)
            cams = [rays_ego @ sources[names[0]].cam_to_rig]
        else:
            # Proxy geometry around the nuPlan camera centre: ground, then cylinder.
            centre = rig.position(target)
            r_xy = np.linalg.norm(rays_ego[..., :2], axis=-1)
            t = PROXY_RADIUS_M / np.maximum(r_xy, 1e-9)
            down = rays_ego[..., 2] < -1e-9
            t_ground = np.where(down, -centre[2] / np.where(down, rays_ego[..., 2], -1.0), np.inf)
            points = centre + rays_ego * np.minimum(t, t_ground)[..., None]
            cams = [(points - sources[n].position) @ sources[n].cam_to_rig for n in names]
        proj = [sources[n].project(c) for n, c in zip(names, cams)]
        valids = np.stack([p[1] for p in proj])
        seen = valids.any(0)
        nearest = np.argmin(np.stack([p[3] for p in proj]), axis=0)
        maps, weights = [], []
        for i, (name, (pix, valid, edge, _)) in enumerate(zip(names, proj)):
            fill = ~seen & (nearest == i)
            px = np.where(valid[..., None], pix, np.where(fill[..., None], edge, -1.0))
            maps.append(cv2.convertMaps(px[..., 0].astype(np.float32), px[..., 1].astype(np.float32),
                                        cv2.CV_16SC2))
            weights.append(np.where(fill, 1.0, _border_weight(pix, valid, sources[name].resolution_hw)))
        norm = None
        if len(names) > 1:
            total = np.sum(weights, axis=0)
            norm = [np.divide(w, total, out=np.zeros_like(w), where=total > 0)[..., None].astype(np.float32)
                    for w in weights]
        out[target] = ViewMap(target, list(names), maps, norm, float(seen.mean()), float((~seen).mean()))
    return out


def render(view: ViewMap, images: Dict[str, np.ndarray]) -> np.ndarray:
    """Images (H, W, 3) uint8 RGB keyed by PAI logical id -> nuPlan view uint8 RGB."""
    outs = []
    for name, (m1, m2) in zip(view.sources, view.maps):
        img = images[name]
        outs.append(cv2.remap(img, m1, m2, interpolation=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=0))
    if view.weights is None:
        return outs[0]
    acc = np.zeros(outs[0].shape, dtype=np.float32)
    for o, w in zip(outs, view.weights):
        acc += o.astype(np.float32) * w
    return np.clip(acc + 0.5, 0, 255).astype(np.uint8)


class SessionViews:
    """Remap tables for one session, built lazily once the rendered size is known."""

    def __init__(self, camera_specs: Dict[str, dict], rig: NuplanRig, rays: np.ndarray):
        missing = [c for c in NEEDED_CAMERAS if c not in camera_specs]
        if missing:
            raise ValueError(f"session lacks cameras {missing}; available {sorted(camera_specs)}")
        self.specs = camera_specs
        self.rig = rig
        self.rays = rays
        self.views: Optional[Dict[str, ViewMap]] = None
        self.rendered_hw: Optional[Tuple[int, int]] = None

    def render_all(self, images: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        hw = images[NEEDED_CAMERAS[0]].shape[:2]
        if any(images[c].shape[:2] != hw for c in NEEDED_CAMERAS):
            raise ValueError("PAI cameras rendered at different sizes: "
                             + str({c: images[c].shape[:2] for c in NEEDED_CAMERAS}))
        if self.views is None or self.rendered_hw != hw:
            sources = {c: FthetaSource.from_spec(self.specs[c], hw) for c in NEEDED_CAMERAS}
            self.views = build_view_maps(sources, self.rig, self.rays)
            self.rendered_hw = hw
        return {t: render(self.views[t], images) for t in TARGET_ORDER}
