# AXE local evaluation on the official curated_val split

Evaluates the AXE DriveSuprim driver against the AlpaSim PAI-AV contract at
upstream commit `cd713e0` -- the commit the competition server runs.

## What this environment is

A pristine `NVlabs/alpasim` checkout of the `e2e_challenge` branch at
`cd713e0`. The scorer is upstream's, unmodified -- including the
`left_corridor_laterally` hard gate, which NVIDIA added in `f012862` (PR #163)
on this same branch and which the organizer-published reference runs in
`../local_evaluation/data/pai/` score against. Nothing in this directory
patches the simulator; the three scripts only start the driver containers and
hand the wizard a GPU placement.

| | |
|---|---|
| AlpaSim | `cd713e0` (e2e_challenge branch, 5 commits ahead of the fork's `f012862`) |
| Driver image | `alpasim-e2e-drivesuprim-stage3:ep24-sweep` |
| Checkpoint | `nurec_stage3_ep24/checkpoint/epoch=24-step=18150.ckpt` |
| Checkpoint sha256 | `1d9fa8d7b8bb4f6d5f732cdf203a3a44c603780c1086726be841eb050d1212dc` |
| Scene suite | `nurec_curated_val` (441 scenes) |
| Artifacts | `/home/kaist5/Dataset/alpasim/data/nre-artifacts` (symlinked in as `data/nre-artifacts`) |

## The contract comes from the presets

`+e2e_challenge=dev` and `+nurec_scenes=curated_val` are what pin the scoring
contract, and they are exactly what produced the organizer-published reference
runs in `../local_evaluation/data/pai/`. Verified in the composed config:

```
cameras            6  (front_wide, front_tele, cross_left, cross_right,
                       rear_left, rear_right) @ 10 Hz, 1080 px
n_sim_steps        200
force_gt_duration  1.7 s
route_start_offset 40.0 m
send_recording_gt  false
controller         nonlinear MPC
harmonizer         enabled
scene_score        progress_saturation 0.8, min_gt_distance 5.0 m
```

Note the harmonizer. `dev.yaml` leaves the base renderer command alone, so the
harmonizer is on -- and the published references were generated the same way.
Only `ec2.yaml` (the leaderboard preset) overrides the renderer command to drop
it. A local score therefore reproduces the reference bundle, not the
leaderboard.

## Deviations from the official topology

Only placement, never scoring. Rollouts are synchronous request/response, so
where a service lands changes wall-clock and nothing else.

| | official `8gpu_32rollouts` | here |
|---|---|---|
| renderer GPUs | 0-3 | 4-7 |
| driver GPUs | 4-7, 16 containers | 4-7, configurable |
| concurrent rollouts | 32 | `ROLLOUT_WORKERS` (default 8) |

GPUs 0-3 belong to other jobs on this box, so the renderer shares 4-7 with the
drivers. Check `nvidia-smi` before starting -- each NuRec scene the renderer
caches stays resident in VRAM, and a driver replica needs roughly 14 GB.

Controller gains are a submitted artifact, not part of the contract. The run
script defaults to the tuned AXE values (`long 0.5`, `lat 6.0`, `idx 3`), all
inside the published allowed ranges. Set `MPC_OVERRIDES=""` for stock defaults.

## Running

```bash
cd /home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim

# 1. start the driver replicas; prints the hydra address list
DRIVERS=$(e2e_challenge/axe_local_eval/start_drivers.sh | tail -1)

# 2. evaluate (441 scenes x 3 rollouts)
DRIVER_ADDRESSES="$DRIVERS" e2e_challenge/axe_local_eval/run_curated_val.sh

# 3. rank against the 8 published reference subjects
uv run --extra local-evaluation \
  python e2e_challenge/local_evaluation/evaluate.py \
  --track pai \
  --run axe-ep24=./runs/axe-ep24-curatedval \
  --output-dir ./runs/axe-ep24-curatedval/local-evaluation

# 4. tear down only our containers
e2e_challenge/axe_local_eval/stop_drivers.sh
```

Smoke first: `SCENE_LIMIT=2 N_ROLLOUTS=1 ROLLOUT_WORKERS=2` with
`REPLICAS_PER_GPU=1 GPU_INDICES_CSV=4,5`.

`stop_drivers.sh` filters by container name on purpose. Other people run
containers from the same AlpaSim images, so an `--filter ancestor=` sweep would
take theirs down too.

## Startup time

A 16-renderer / 16-controller stack on this machine used to take 30-50 minutes
to bind its first port, and often never did: the runtime's version probe gives
up after `SERVICE_STARTUP_TIMEOUT_SEC`. Two mechanisms, both measured here on
2026-09-23, and both fixed by default in `run_curated_val.sh` (`FAST_STARTUP=1`):

- **`.pyc` write storm.** Every fresh container compiles Python bytecode on
  import and writes it into its own overlay layer. `docker diff` eight minutes
  after `compose up` showed 745 new `.pyc` in one renderer and 236 in the
  controller, with all 37 python processes in uninterruptible disk wait and the
  shared root disk pinned at 100%. `PYTHONDONTWRITEBYTECODE=1` is now passed to
  renderer, physics, controller and runtime. It changes nothing about what they
  compute.
- **Renderer downloads.** Each renderer fetches ~2.8 GB of harmonizer and
  huggingface weights into `$HOME/.cache` (`HOME=/home` in `nre-ga:26.04`).
  `run_curated_val.sh` mounts `.cache/renderer-shared/` (override with
  `RENDER_CACHE`) as every renderer's `/home/.cache`, and puts the inductor and
  triton compile caches there too. The directory must be seeded *before*
  sixteen replicas start, or they race on the download rename;
  `warm_renderer_cache.sh` does that with a single renderer and is called
  automatically when the cache is empty (`RENDER_CACHE_WARM=0` skips the check).

Two smaller items ride along: `UV_OFFLINE=1` (a github.com outage once stopped
runs that needed nothing from the network), and `REUSE_DRIVERS=1` for
`start_drivers.sh`, which prints the addresses of an already-running, ready
driver set instead of loading sixteen checkpoints again -- useful when a retry
is about the simulator side. Reuse is keyed on container name only, so change
`CONTAINER_PREFIX` when the image or driver environment changes.

Startup is still bounded by the shared disk: check `iostat -xz 1 5` and
`vmstat 1 5` first, and expect several minutes even when it is quiet.

## Artifacts

`curated_val` draws from three HF revisions (`26.04`, `26.01`, `25.05`); the
CSVs carry the per-scene revision and the wizard downloads what is missing into
`all-usdzs/<artifact-uuid>.usdz`. Note that name: the older hand-built symlinks
in that directory are named by scene-id suffix instead, and the wizard does not
recognise those, so it re-downloads a scene that looks present.

## Local score is not the leaderboard score

The PCS scale is anchored to two reference subjects chosen for this bundle
(`alternative_2` = 600, `alpamayo1` = 2000). It ranks models against each other;
it does not predict the official number. The leaderboard also runs a private
scene set with the harmonizer off.
