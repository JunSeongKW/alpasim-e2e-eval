# Runbook — from a finished render to a running training

State as of 2026-08-26 01:20. Three inputs are in flight; nothing else blocks.

| input | who | when |
|---|---|---|
| `challenge_jpeg95_1920x1080` render | render owner | early morning |
| rectification to 512x256 K377 | `rectify_follow.sh`, running | tracks the render |
| per-token PDM scores | scoring run on another host | ~20:00 |
| **`driver.py` K377** | **render owner — not applied yet** | **before evaluation** |

## Morning — when the render and rectification finish

```bash
source setup_nurec.sh
bash scripts/nurec/finalize_data.sh
```

Rewrites the logs against the rectified images, runs the preflight, projects the
annotated boxes, and re-runs the geometry tests. Expect roughly

```
frames_without_images        0
max_frame_interval_error_us  0
framed_corner_inside_ratio   ~0.97
ground_below_horizon_ratio   ~0.99
nan_projections              0
```

and one deliberate test failure — `test_the_target_pinhole_is_the_drivers` —
until the driver moves to 377.

## Between morning and the scores landing

Two things worth measuring while the GPUs are free, neither of which needs the
PDM scores:

**1. Dataloader throughput.** There is no feature cache (see below), so sample
production happens in the dataloader workers and sets the training rate. One
worker produces ~1.25 samples/s. At `batch_size=8` per rank that is ~0.8 s of
features per step with `num_workers=8`. Measure against the actual step time and
raise `NUM_WORKERS` until the GPU stops waiting.

**2. A short real run.** Everything verified so far is `fast_dev_run` — exactly
one step. Not yet observed: the loss going down, the LR schedule, checkpoint
writing, the teacher EMA over many steps. A few hundred steps on the tokens that
are already scored would cover all of it:

```bash
source setup_nurec.sh
NUM_GPUS=3 BATCH_SIZE=8 bash scripts/nurec/train_planonly.sh \
  drivesuprim_agent_bevformer_vov_v2_vits_planonly_scratch \
  ++trainer.params.max_steps=300 trainer.params.limit_val_batches=4 \
  "train_test_split.scene_filter.tokens=[...scored tokens...]"
```

## Why there is no feature cache

`cache_path` defaults to `null`. Not because the cache is large (it would be
~600 GB: 12.5 MB a sample x 48,038) but because photometric augmentation is
applied *inside* `compute_features`:

```python
frame_teacher.append(self.bev_teacher_augmentation(resized))   # clean
frame_student.append(self.bev_student_augmentation(resized))   # jitter + blur
```

Caching stores the augmented tensors, so one jitter draw would be frozen into
every epoch and the student/teacher photometric consistency the SSL objective
rests on would see the same pair forever. Set `NUREC_CACHE_ROOT` only if that is
what you want.

## Night — the real run

```bash
source setup_nurec.sh
NUM_GPUS=4 BATCH_SIZE=8 bash scripts/nurec/train_planonly.sh
```

The launcher refuses to start while any training token lacks a PDM score, and
names how many are missing. Defaults to
`drivesuprim_agent_bevformer_vov_v2_vits_planonly_scratch`: ViT-Small/DINOv3,
planning objective only, no perception heads, no feasibility gate, no stage-1
checkpoint.

## Verified, and not

| | |
|---|---|
| data → features → planning loss → backward → step | yes |
| DDP, 3 GPUs, `static_graph=True` | yes |
| no perception loss, no gate | yes — `pdm_tl_loss=0`, no aux heads |
| loss decreasing over 300 steps | yes — 89.7 → ~73, val/loss-ori 18.0 → 16.7 |
| checkpoint write | yes — `save_top_k` on `val/loss-ori`, 711 MB each |
| dataloader keeping up | yes — GPU at 84-100% with `num_workers=8` |
| `batch_size=8` per rank | yes, but **18.8 GB of 24 GB** in a real run |
| training K == driver K | **no — driver still 412** |

`num_workers` measured without a GPU: 0 → 5.6 samples/s, 4 → 14.5, 8 → 98.5,
16 → 38.7. Eight is the knee; sixteen contends. The 98.5 is a warm-page-cache
figure and optimistic, but eight was enough to hold the GPU at 84-100% in the
300-step run, which is the number that matters.

The 18.8 GB is the one to watch. The earlier 6.2 GB came from `fast_dev_run` —
a single step, before the optimizer state and activations settle. A real run
leaves ~5 GB of headroom on a 24 GB card, so a second tenant on the same GPU
will OOM it. Check `nvidia-smi` before launching.

`max_steps` is not in the trainer config, so it needs `++` to append:
`++trainer.params.max_steps=N`. A plain assignment raises a Hydra struct error.

Training at 377 while the driver stays at 412 recreates the mismatch this whole
effort removed, with the sign flipped. The checkpoint and the driver have to
move together.
