# NuRec image geometry — what is done and what is left

The rendered NuRec images and the calibration in the NAVSIM logs describe two
different cameras. Training resized a raw f-theta image and projected it with a
fixed NuPlan pinhole; AlpaSim evaluation rectifies the same image to a pinhole
and projects it with the clip's own extrinsics. This file records what has been
verified, what is built, and what remains.

Nothing here touches the EPDMS run: it needs no GPU, reads from `/mnt/nfs`, and
writes to a copy of the logs rather than the ones being scored.

## What the mismatch actually is

Measured on this dataset, not taken from the write-up:

| | NAVSIM logs (all 1,603 clips) | USDZ, per clip |
|---|---|---|
| model | pinhole, zero distortion | `ftheta` |
| intrinsics | fx=fy=1545, cx=960, cy=560 | polynomial, principal point ~(961, 745) |
| extrinsics | one fixed set for every clip | `T_sensor_rig`, per clip |

The principal point alone is 185 px out vertically. The intrinsics are as wrong
as the extrinsics, so fixing only the poses would not have helped.

## The resolution is not a choice; the pinhole inside it was

`drivesuprim_agent_bevformer_vov_v2_vits*` resolves to `backbone_type:
bevformer_m`, which takes the per-camera BEV path — `bev_img_width/height` of
512x256 with a `lidar2img` per camera — and never the 2048x512 `ori` panorama.
So 512x256 is fixed. What goes *in* those pixels is a free parameter once the
images are rectified, and the value it started at was inherited rather than
chosen:

```
was   fx 412.0   fy 366.22222222222223   cx 256.0   cy 132.74074074074073
now   fx 377.0   fy 377.0                cx 256.0   cy 128.0
```

The old numbers are nuPlan's pinhole — fx=fy=1545, cx=960, cy=560 at 1920x1080 —
carried through DriveSuprim's 512x256 resize. Two things were wrong with them,
both measured here (`target.py` carries the same argument next to the constants):

* **A 12.5% vertical squash.** 512x256 is 2:1 and 1920x1080 is 1.78:1, so the
  resize compressed vertically: a circle in the world projected to an ellipse of
  ratio 1.125. On NAVSIM this was unavoidable — the camera is real and
  fx=fy=1545 is fixed — and `bev_crop_top_bottom: 28`, the SafeDrive recipe
  validated on NAVSIM, was the only lever, taking the squash to 6.7%. It could
  not reach zero. Rectification makes fx and fy free, so it does: fx=fy → 0%,
  and fy=377 holds SafeDrive's vertical framing to within 2% (37.5° vs 36.7°).
* **A wedge no camera saw.** The cross cameras sit ~67° off forward and the
  separation varies per clip — across all 1,607 clips, p50 67.02°, p99 68.35°,
  max 70.17°. At fx=412 each view is 63.7° wide, leaving 3.4° between adjacent
  cameras: 2.9 m of blind width at 40 m, enough to hide a car, and in the clip
  checked it hid one. fx=377 gives 68.4° and closes it on 99% of clips. Closing
  it on all of them needs fx ≤ 364.5.

Cost: objects are 8.5% smaller horizontally than at fx=412, while fy *rises*
2.9% — this un-squashes rather than shrinking.

Evidence: `data/validation/target_k_comparison.png` (four scenes, three
candidate targets, with a 1 m world circle drawn — 69x61 px at the old target,
63x63 at 377) and `data/validation/blind_gap.png` (the three views laid out by
true bearing, wedges marked).

**Training and the driver must hold the same four numbers.** Training is at 377;
the driver is not patched yet. `docs/nurec/DRIVER_K_PATCH.md` is the change, and
`tests/test_nurec_rectified_geometry.py::test_the_target_pinhole_is_the_drivers`
parses the driver's source and fails until it lands.

With rectified images, `bev_undistort` and `bev_crop_top_bottom` must both be
off: the distortion is already gone, and cropping 28 px from a 256-line image
moves it off the target the driver rectifies to. Its *effect* is in the target K
instead. Every BEV yaml already leaves both at `false`/`0`; only
`..._safedrive_resize.yaml` sets them, and that one must not be used here.

## Built and verified

### The rectifier

`navsim/planning/data/nurec_rectify/`

- `rectification.py` — the challenge driver's rectifier, copied verbatim; the
  only edit is the import.
- `ftheta_proto.py` — stand-ins for the six protobuf fields it reads, plus
  `available_camera_from_usdz` to build one from a USDZ camera model.
- `schema.py` — `RectificationTargetConfig`, lifted from the same submission.
- `target.py` — the driver's target pinhole, stated once for the whole
  pipeline: `TARGET_K`, `TARGET_DISTORTION`, `rectification_target_config()`.

`tests/test_nurec_rectification.py` (5 tests) pins the copy to its source: the
file must differ only in its import, and both must produce identical pixels from
a real f-theta model.

Measured: rectifier build 0.05-0.11 s per camera, 2-10 ms per image.

### The pipeline

- `calibration.py` — reads `rig_trajectories.json` out of each clip's USDZ
  (0.06-0.08 s per clip; the 1.9 GB volume beside it is never touched) and
  caches every clip's `T_sensor_rig` and f-theta model in one JSON.
- `images.py` — rectifies a render tree into a 512x256 mirror of itself, one
  rectifier per (clip, camera), resumable through a per-clip `_complete` marker.
- `logs.py` — writes a **copy** of the prepared root whose CAM_L0/F0/R0 carry
  the rectified `data_path`, `TARGET_K`, zero `distortion`, and the clip's own
  `sensor2lidar_rotation`/`_translation`. Out comes a whole prepared root, not
  just logs: `train.sh` reads `navsim_log_path` and `original_sensor_path` out of
  `$NUREC_PREPARED_ROOT/index.json`, so an `index.json` naming the rectified
  sensor root is written beside the logs and `maps`, `routes` and `metric_cache`
  — all keyed by log name, none of them touched by rectification — are symlinked
  back. Logs that could not be rewritten are dropped from the splits, so training
  never asks for a pickle that is not there.

`scripts/nurec/verify_rectified_projection.py` rebuilds `lidar2img` from the
rewritten log alone and projects the annotated boxes into the rectified images.
`nurec_validate.py` and `nurec_attach_images.py` share one rule about which
frames are allowed to have no image.

### The frames reconcile, and that was the part most likely to be wrong

Deriving `sensor2lidar` from `T_sensor_rig` turned out to need no rebasing at
all, for two reasons that are now checked rather than assumed
(`tests/test_nurec_rectified_geometry.py`, 14 tests):

- **`T_sensor_rig` is sensor-to-rig.** Its columns are the camera's optical axes
  in rig coordinates and its translation is the camera origin in the rig frame —
  which is NAVSIM's `sensor2lidar` convention exactly, and the same reading
  `cnx_bev_bridge._camera_to_rig_matrix` uses at evaluation time. Read the other
  way round it is still a valid rotation, so the test checks that each camera
  ends up looking where its name says: CAM_L0 forward-and-left, CAM_R0
  forward-and-right, image-y down.
- **The NAVSIM ego frame is the NuRec rig frame.** The prepared logs'
  `ego2global_translation` reproduces `rig_trajectories.json`'s `T_rig_worlds`
  to under a millimetre at timestamps the two share, and `lidar2ego` is the
  identity — so sensor-to-rig is already sensor-to-lidar.

The projection is then the driver's: `TARGET_K @ inv(T_sensor_rig)`. It clears
the same sanity check the driver refuses to start a session without — a point
10 m down the optical axis lands within 0.01 px of the principal point — and
`DriveSuprimFeatureBuilder._build_lidar2img` reproduces it exactly at 512x256.

### It goes through the real feature builder

Not a stand-in: `DriveSuprimFeatureBuilder._get_bev_camera_feature`, driven by an
`AgentInput` built from a rewritten log with
`drivesuprim_agent_bevformer_vov_v2_vits_planonly_scratch`.

```
backbone_type=bevformer_m  bev=512x256  num_cameras=3  seq_len=3
undistort=False  crop=0                     <- already the yaml's defaults
bev_imgs      (3, 3, 3, 256, 512)  finite   <- the resize is the identity
lidar2img     (3, 3, 4, 4)         finite
cam_l0/f0/r0: |lidar2img - TARGET_K @ inv(T_sensor_rig)| <= 2.8e-05
              probe10m uv = (256.000, 128.000)
```

So the matrix the model trains against is the challenge driver's
`_build_virtual_lidar2img` to within float32 rounding. `bev_undistort` and
`bev_crop_top_bottom` need no change — they are already `false`/`0` everywhere
except `..._safedrive_resize.yaml`, which must not be used with this tree.

The preflight `train.sh` runs before launching DDP passes on the rectified root:

```
logs 617   frames 25271   frames_with_route 25271   checked_images 74028
frames_without_images 595   max_frame_interval_error_us 0
```

(617 rather than 1,603 only because the rectification run is still going.)

### Verified on the images

Over the first five rewritten logs (4,951 annotations, 205 frames):

```
framed_corner_inside_ratio   0.979   corners of framed boxes inside the frame
framed_whole_ratio           0.938   framed boxes entirely inside
ground_below_horizon_ratio   1.000   visible ground ahead below the principal row
nan_projections                  0
```

Numbers alone could not have caught a projection that is consistently wrong with
itself, so the overlays are the check that decides: at close range the boxes
wrap the vehicles, footprint on the wheels and roofline on the roof, and the
ego's own future path traces the lane it drives down. `--out-dir` writes a
contact sheet.

One thing worth knowing before reading the report: the rectified views are about
64° wide against the 120° lens they came from, and the cross cameras sit ~67°
off forward, so the three views barely overlap and most annotations in a clip
are outside every view. `boxes_framed` is a small fraction of `annotations` even
when the calibration is perfect.

## How to run it

```bash
source setup_nurec.sh      # every path below comes from here

# 1. per-clip calibration out of the USDZ archives (~2 min for 1,607 clips)
python -m navsim.planning.data.nurec_rectify.calibration \
  --usdz-root $NUREC_USDZ_ROOT --output $NUREC_CALIBRATION --workers 8

# 2. rectify (resumable; skips clips that already have _complete)
python -m navsim.planning.data.nurec_rectify.images \
  --calibration $NUREC_CALIBRATION --render-root $NUREC_RENDER_ROOT \
  --output-root $NUREC_SENSOR_ROOT --workers 12

# 3. the rectified prepared root -- source logs are not touched
python -m navsim.planning.data.nurec_rectify.logs \
  --source-prepared-root $NUREC_SOURCE_ROOT --output-prepared-root $NUREC_PREPARED_ROOT \
  --calibration $NUREC_CALIBRATION --image-root $NUREC_SENSOR_ROOT

# 4. verify
python scripts/nurec/verify_rectified_projection.py \
  --log-root $NUREC_PREPARED_ROOT/navsim_logs/trainval \
  --image-root $NUREC_SENSOR_ROOT --out-dir $NUREC_VALIDATION_ROOT/images_iso377 --limit 20
```

Steps 2 and 3 are resumable and the raw render is still being produced, so
re-running them picks up only what appeared since. Step 2 is I/O bound on the
NFS renders: about 4-8 clips a minute at 12 workers on a shared box, ~14 MB per
clip, ~22 GB for the full set.

## It runs through the training pipeline

The real `DriveSuprimFeatureBuilder` was given a rewritten log and produced the
features the BEV backbone reads:

```
backbone_type=bevformer_m  bev=512x256  num_cameras=3  seq_len=3
undistort=False  crop=0
bev_imgs          (3, 3, 3, 256, 512)  float32  finite=True
lidar2img         (3, 3, 4, 4)         float32  finite=True
bev_ego_pose      (3, 3)               float32  finite=True

cam_l0: |lidar2img - TARGET_K @ inv(T_sensor_rig)| = 2.8e-05  probe10m uv=(256.000, 128.000)
cam_f0: |lidar2img - TARGET_K @ inv(T_sensor_rig)| = 1.4e-05  probe10m uv=(256.000, 128.000)
cam_r0: |lidar2img - TARGET_K @ inv(T_sensor_rig)| = 1.4e-05  probe10m uv=(256.000, 128.000)
```

So the matrix the model trains against is the one the challenge driver builds,
to float32 precision, and the driver's own session-start probe passes through
the feature builder unchanged.

Two things this settled that were assumptions before:

- **The agent config already needs no edit.**
  `drivesuprim_agent_bevformer_vov_v2_vits_planonly_scratch` (and every BEV yaml
  except `..._safedrive_resize`) leaves `bev_undistort: false` and
  `bev_crop_top_bottom: 0` at their defaults, which is what rectified images
  need. Only `..._safedrive_resize.yaml` sets `bev_undistort: true` /
  `bev_crop_top_bottom: 28`, and that one must not be used against this tree.
- **`nurec_validate` had the same over-strict rule** as `nurec_attach_images`,
  and it is the preflight `scripts/nurec/train.sh` runs before launching DDP, so
  it would have rejected every clip over the missing 41st frame. It now takes
  the same `trailing_frames_without_images`.

## Left to do

1. **The driver's four constants.** Training is at 377 and the challenge driver
   is not — `docs/nurec/DRIVER_K_PATCH.md`, plus a submission image rebuild.
   Agreed with the render owner (`docs/nurec/junseong_exchange/04`); pending.
2. **The checkpoint has to match the driver.** Existing checkpoints were trained
   at 412/366, so patching the driver without a 377-trained checkpoint recreates
   this same mismatch with the sign flipped. Order: train at 377, then switch
   the driver.
3. **The rest of the render.** `data/renders_raw` was 1,055 of 1,603 clips; steps
   2 and 3 pick up the remainder when it lands.
4. **The five cameras NuRec does not render** — CAM_L1/L2/R1/R2/B0 — are left
   untouched in the rewritten logs: still nuPlan's calibration, still pointing at
   images that do not exist. `bev_num_cameras: 3` never reads them, but a
   five-camera config would, and this tree cannot serve one.

## The 41st frame, and the 60 ms limit

`nurec_attach_images` attached 41 of 1,603 logs. Two separate causes, and only
one of them is a bug. Both were measured across all 1,603 logs; the numbers are
in `data/validation/timestamp_offsets.txt`.

**Fixed — the last frame.** Each log has 41 frames; the 2 Hz render produced 40.
NAVSIM only ever loads sensors for the first `num_history_frames` of a scene
window (`AgentInput.from_scene_dict_list`), and with `num_history_frames: 4` /
`num_future_frames: 8` the last frame of a log can only ever be a future frame,
which needs no image. Requiring an image for every frame therefore rejected a
whole clip over a frame no training window would have opened. Missing images are
now tolerated as a *suffix* — `trailing_frames_without_images`, defaulting to the
scene filter's 8 — while a gap anywhere earlier still rejects the clip, because
that one *would* be read. Frames past the last render get an explicit
`__no_render__/...` `data_path` rather than a stale path into the un-rectified
tree.

That alone takes the attach rate from 41 to **1,247 of 1,603**.

**A threshold decision — the remaining 356.** The offset is not the constant
57.9 ms this file used to record. The renderer picked the nearest real capture
time to each requested timestamp, and those are systematically *late*: over a
200-clip sample the signed offset runs 0–64 ms with a median of 43–47 ms, and
the only negatives are each log's missing last frame. The three cameras do not
fire together either, so a clip whose front camera lands at 33 ms has its two
cross cameras at 63 ms — past `DEFAULT_MAX_TIMESTAMP_ERROR_US` of 60,000 on
*every* frame, which the trailing-frame rule cannot help.

```
tolerance= 60ms  trailing=0:   41/1603    trailing=8: 1247/1603
tolerance= 70ms  trailing=0:   44/1603    trailing=8: 1601/1603
tolerance= 80ms  trailing=0:   44/1603    trailing=8: 1602/1603
tolerance=120ms  trailing=0:   44/1603    trailing=8: 1603/1603
```

70 ms takes it to 1,601. But the limit is not really what is wrong here: at the
*median* offset of 45 ms the ego has moved 0.9 m at 20 m/s between the pose the
log records and the image paired with it, and raising the tolerance accepts more
of that rather than less. This is a call about how much stale-image error the
training is willing to carry, so it is left as a flag rather than changed.

**Superseded, probably.** `junseong` is re-rendering the whole set at the exact
NAVSIM timestamps —
`26.04_release_navsim_exact_2hz_challenge_raw_1900x1080_3cam_png`, 41 frames per
clip, `timestamp_error_us=0` — which removes both causes and the 0.9 m with them.
It was 704 of 1,603 clips done as of 2026-08-25. Note it renders at
**1900x1080**, not 1920x1080; `images.py` reads the resolution off the PNGs and
scales the f-theta model to it the way `build_ftheta_rectifier_for_resolution`
does, so the same command works against either tree — but the two rectified
outputs are not interchangeable, and the resolution the eval session reports
should be confirmed to match whichever one training uses.

## Data locations

Everything resolves from this tree; `setup_nurec.sh` exports the variables and
`./data/*` are symlinks to where the bytes actually live.

```
data/usdz          -> .../NuRec/sample_set/26.04_release/{clip}/{clip}.usdz
data/renders_raw   -> .../26.04_release_navsim_exact_2hz_challenge_raw_1900x1080_3cam_png
                      the challenge-matched render: 41 frames, timestamp error 0,
                      ego hood present.  Still being produced.
data/prepared_source -> nurec_axe/           source logs, maps, routes, metric_cache
                      navsim_logs/trainval/nurec-{clip}.pkl   (never edited in place)
data/images        -> .../navsim_exact_2hz_512x256_iso377_3cam_png/{clip}/{camera}/{ts}.png
data/prepared      -> nurec_axe_iso377/      NUREC_PREPARED_ROOT
                      navsim_logs/trainval/nurec-{clip}.pkl
                      index.json                       (sensor_root -> data/images)
                      {maps,routes,metric_cache}       -> prepared_source
data/epdms         -> nurec_epdms/           per-token PDM scores
data/validation    -> nurec_validation/      figures, reports
assets/nurec/calibration.json                per-clip T_sensor_rig + f-theta model

reference code  /rhome/junseong/alpasim/e2e_challenge/sample_submission_drivesuprim/
camera names    camera_cross_left_120fov / camera_front_wide_120fov / camera_cross_right_120fov
                -> CAM_L0 / CAM_F0 / CAM_R0
```

Superseded and kept only as a fallback: `data/rectified_all/26.04_release_2hz_512x256_3cam_png`
(the 412/366 rectification of the older 1920x1080 render, 1,607 clips, 24 GB) and
`nurec_axe_rectified/` (617 logs pointing at it). Both have the older render's
defects — 40 frames, +45 ms timestamps, no ego hood — on top of the old target K.
