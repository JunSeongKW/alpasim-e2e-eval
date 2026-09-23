# Handoff: epoch 29 / step 30330 checkpoint evaluation

## Objective

Evaluate the checkpoint below on all 441 `curated_val` clips using the same
local evaluation contract and parallelism as the AXE-v9 submission evaluation.

- Checkpoint: `/home/kaist5/data/junseong/AlpaSim-E2E-Challenge/models/260923_epoch=29-step=30330.ckpt`
- SHA-256: `7e0e59be63146c72e917989d30ce778ce807e7971386d61c66e5d6f925db289b`
- GPUs: 4, 5, 6, 7
- Preset: `dev`
- Scenes: 441 `curated_val` clips
- Rollouts per scene: 1
- Driver replicas: 16 (4 per GPU)
- Runtime workers: 16
- Renderer replicas: 16 (4 per GPU)
- Physics replicas: 4, with concurrency 4 each
- Controller replicas: 16
- Video: disabled
- MPC gains: longitudinal position weight 0.25, lateral position weight 1.0,
  start-index penalty 3

No valid rollout completed in the attempts described below. There is no score
to report and no `aggregate/results-summary.json`.

## Prepared artifacts

The submission image was reproduced by replacing only the checkpoint in the
AXE-v9 image.

- Base image: `696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v9`
- Derived image: `alpasim-e2e-axe-v9:260923-ep29-step30330`
- Derived image ID: `sha256:dc5d8f006b2e42a753ff0669ca00953fe3a49efc5e3c4a5abffdddcdef500273`
- Overlay Dockerfile: `e2e_challenge/axe_local_eval/Dockerfile.checkpoint_overlay`
- Evaluation script: `e2e_challenge/axe_local_eval/run_260923_ep29_step30330.sh`
- Failed run directory: `runs/leaderboard-260923-ep29-step30330`
- Wizard log: `runs/leaderboard-260923-ep29-step30330.wizard.log`
- Launcher log: `e2e_challenge/axe_local_eval/260923_ep29_step30330.log`
- Provenance file: `runs/leaderboard-260923-ep29-step30330.provenance.json`

The derived image has the same environment, command, entrypoint, working
directory, and user as AXE-v9. The checkpoint inside the image has the expected
SHA-256.

A single-driver smoke test on GPU 4 succeeded before the full run:

```text
[DriveSuprim] exact checkpoint load: missing=0 unexpected=0
[DriveSuprim] policy load complete
```

There was no checkpoint error and no CUDA OOM.

## First full attempt

Timeline on 2026-09-23 KST:

- 14:24: launcher started and waited for GPUs 4-7.
- 14:29: GPUs were free; 16 AXE-v9 drivers started.
- 14:36: all 16 drivers were ready; the 441-clip wizard started.
- About 15:14: renderer initialization eventually completed after very slow
  container and model-cache startup.
- 16:20: wizard exited with status 1.

The driver services all loaded the new checkpoint and answered version probes.
The wizard generated and started the renderer, physics, and controller
containers, but `runtime-0` never acquired a host process. There were 21
service shims (16 renderer + 4 physics + 1 controller) instead of the expected
22. No `alpasim_runtime.simulate` process existed, no worker log was created,
and the completed count remained 0/441.

Docker inspection and removal calls frequently took 15-300 seconds or timed
out. After the wizard exited, `docker compose down` also stalled. The services
were eventually torn down, so a host-launched runtime could connect to the 16
drivers but correctly failed to connect to the simulation services that had
already been removed. That host-runtime attempt was stopped and produced no
rollout.

## Controlled startup attempt

A second startup reused the generated compose configuration but created
services in smaller groups to avoid launching 22 containers at once.

- Physics became ready: 4/4.
- The first renderer group became ready: 4/4.
- Two more renderer containers started under an alternate Compose project.
- Controller remained at 0/16 after roughly 28 minutes. All 16 controller
  Python processes were in uninterruptible `D` state and had not opened ports
  6020-6035.
- Several renderer batch-create calls timed out, but Docker later exposed
  partially created containers in `Created` state. Container-name reservations
  remained after client timeouts.

This attempt was stopped at the user's request before starting the runtime.
It also produced 0/441 completed rollouts.

## Failure cause

The failure is an infrastructure startup failure caused by severe shared root
disk and Docker overlay-storage contention. It is not a model, checkpoint, GPU
memory, or scoring-code failure.

Observed evidence:

- `/`, `/home`, Docker storage, repository data, and the other active jobs all
  reside on `/dev/vda1`.
- `iostat -xz 1` repeatedly showed `vda` at 97.4-99.6% utilization.
- `vmstat` showed 5-16 blocked processes during startup.
- A later snapshot had 36 processes in `D` state plus multiple
  `flush-253:0` kernel workers in `D` state.
- GPUs 4-7 were mostly at 0% utilization while container creation and Python
  imports were stalled.
- Even one Docker container could take 1-3 minutes to reach `Started`.
- Docker API calls such as `inspect`, `rm`, and `system df` sometimes timed out.

The AXE-v9 startup pattern amplifies this bottleneck:

- The controller container launches 16 `uv run python` processes
  simultaneously, causing concurrent environment and module reads.
- Each renderer has an independent writable Docker layer and initializes its
  Harmonizer cache under `/tmp/nre-cache-dir`.
- Sixteen renderer containers initializing together generate heavy overlay
  metadata and cache I/O.

There were also unrelated active Ray/training jobs under
`/home/kaist5/data/hanbin/VLA/Asyncronous_DrivoR_VLM`. Multiple Ray sessions,
training workers, and raylets from those jobs were in `D` state on the same
root disk. Do not stop those jobs without their owner's authorization. Their
presence confirms this was system-wide shared-storage contention rather than
an isolated AlpaSim process problem.

## Cleanup and retained state

Cleanup was completed and verified at handoff: no matching evaluation
container or process remained, and none of ports 6000-6035 or 6900-6915 was
listening. GPUs 4-7 subsequently showed about 932 MiB each from unrelated
`/home/kaist5/data/junhyeok/...` Python processes; those were not modified.
Verify the state again before retrying:

```bash
docker ps -a --format '{{.Names}}' \
  | grep -E 'leaderboard-260923-ep29-step30330|e29r2|axe-lb-260923-e29s30330'
ss -ltn | grep -E ':(600[0-9]|601[0-9]|602[0-9]|603[0-5]|690[0-9]|691[0-5])\\b'
nvidia-smi -i 4,5,6,7
```

The checkpoint, derived image, Dockerfile, evaluation script, failed-run logs,
and this handoff document are intentionally retained. SafeDrive files and jobs
were not modified.

## Recommended retry

Wait until the shared disk is quiet before retrying. A reasonable gate is:

```bash
iostat -xz 1 5
vmstat 1 5
```

Proceed only when `vda` is no longer pinned near 100% utilization, the blocked
process count is close to zero, and basic Docker create/remove operations
complete promptly. Confirm GPUs 4-7 are free as well.

The launcher refuses to overwrite an existing run directory. Archive or remove
the failed evidence directory first, then run:

```bash
cd /home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim
mv runs/leaderboard-260923-ep29-step30330 \
   runs/leaderboard-260923-ep29-step30330.failed-20260923
mv runs/leaderboard-260923-ep29-step30330.provenance.json \
   runs/leaderboard-260923-ep29-step30330.failed-20260923.provenance.json
./e2e_challenge/axe_local_eval/run_260923_ep29_step30330.sh
```

For the cleanest comparison, retry the unchanged launcher under low disk load.
If startup must be made robust while the server remains busy, start physics,
controller, and renderer containers in small groups and launch
`alpasim_runtime.simulate` only after every gRPC version probe succeeds. That
changes startup orchestration only; do not change the 16-driver, 16-worker,
16-renderer evaluation contract.
