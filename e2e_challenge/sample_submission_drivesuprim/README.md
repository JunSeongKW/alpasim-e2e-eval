# DriveSuprim AlpaSim E2E Evaluation

This is a DriveSuprim-specific contestant driver for the AlpaSim E2E challenge.
It uses the VAVAM sample only for the already-tested gRPC session, camera
delivery, route classification, and trajectory-cache plumbing. DriveSuprim
model construction, input features, checkpoint loading, and inference are
implemented separately in `drivesuprim_challenge`.

## Prepare assets

From the AlpaSim repository root:

```bash
bash e2e_challenge/sample_submission_drivesuprim/scripts/prepare_assets.sh \
  /rhome/junhyeok/DriveSuprim
```

## Build and start

```bash
bash e2e_challenge/sample_submission_drivesuprim/scripts/build_image.sh
e2e_challenge/sample_submission_drivesuprim/run_local_container.sh
```

The host must have NVIDIA Container Toolkit configured for Docker. The launch
script uses Docker's standard `--gpus` interface, so it does not bind
`/dev/nvidia*` or host driver libraries manually. Select a GPU with
`ALPASIM_GPU_INDEX` (default: `0`).

The first startup loads the released checkpoint and can take some time.
Following the NVIDIA VAVAM sample's configurable readiness pattern,
`GetVersion` responds immediately by default so the managed evaluator can
identify every replica while policies load concurrently. Set
`DRIVESUPRIM_REQUIRE_POLICY_FOR_VERSION=1` to make local version probes wait for
policy readiness. `StartSession` still waits for the policy before accepting a
rollout, so the evaluator cannot begin against the fallback trajectory while
the model is loading. Its default wait limit is 600 seconds and can be changed
with `DRIVESUPRIM_SESSION_STARTUP_TIMEOUT_S`.

Each managed replica receives two concurrent rollouts. DriveSuprim replans on
every `Drive` RPC without an additional time-based throttle and combines
inference requests that arrive within 5 ms into one model forward with batch
size up to two. The released model architecture and weights are unchanged.
`DRIVESUPRIM_MAX_BATCH_SIZE` and `DRIVESUPRIM_BATCH_WAIT_MS` control this
micro-batching behavior. `DRIVESUPRIM_INFERENCE_INTERVAL_US` controls the
optional minimum replanning interval and defaults to `0` (disabled).

## Run the local PAI evaluation

In another terminal:

```bash
source setup_local_env.sh
ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=6789 \
uv run alpasim_wizard +e2e_challenge=dev \
  wizard.log_dir=./runs/e2e_challenge_drivesuprim
```

The aggregate result is written under:

```text
runs/e2e_challenge_drivesuprim/aggregate/
```

## Input adaptation

- The NuPlan track uses the native `CAM_L0`, `CAM_F0`, and `CAM_R0` streams.
- The PAI track maps `camera_cross_left_120fov`,
  `camera_front_wide_120fov`, and `camera_cross_right_120fov` to those roles.
- DriveSuprim's original feature builder performs its released
  crop/stitch/resize pipeline and produces the 1024 x 256 ViT input.
- AlpaSim route waypoints are converted to NAVSIM left/straight/right one-hot
  commands.
- AlpaSim linear velocity and acceleration supply the remaining status inputs.
- DriveSuprim returns 40 ego-relative poses at 10 Hz over four seconds.

When `DRIVESUPRIM_BACKBONE_TYPE=bevformer_m`, the driver switches to the
optional BEV adapter instead of the stitched-camera path. It keeps three
synchronized multi-camera frames per session, rectifies f-theta inputs to a
120-degree pinhole view, constructs `lidar2img` from the advertised camera
intrinsics/extrinsics, and supplies ego poses relative to the newest frame.
The first two inference calls pad unavailable history with the oldest frame.
Other backbone types continue to use the original crop/stitch/resize adapter.

The NuPlan camera names and resolutions match DriveSuprim's NAVSIM input
contract most closely. The PAI mapping preserves the same preprocessing shape,
but its camera calibration and visual domain differ from the training data.
