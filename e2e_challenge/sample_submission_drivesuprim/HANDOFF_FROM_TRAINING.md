# What the evaluation environment needs from a training run

Written 2026-09-01. Companion to `EVAL_INPUT_SPEC.md` (the model input contract)
and `AXE_NUREC_RANKING_ABLATION_30CLIPS.md` (what we measured with it).

**A checkpoint alone is not enough.** The checkpoint carries no
`hyper_parameters` -- its top-level keys are `epoch`, `global_step`,
`pytorch-lightning_version`, `loops`, `callbacks`, `optimizer_states`,
`lr_schedulers` -- so the architecture and the ranking composition cannot be
recovered from it. They have to arrive alongside.

---

## 1. Always send these

| Item | Why the build needs it |
|---|---|
| `<name>.ckpt` | The weights. ~687 MB for ViT-S plan-only. |
| `drivesuprim_agent_bevformer_vov_v2_vits_planonly_scratch_nurec.yaml` | The deployment config is composed from this yaml with hydra. Without it we are guessing the head set, the BEV geometry and the ranking weights. |
| `navsim/` (the tree you trained with) | Two separate reasons: the config generator reads `PACKAGE_ROOT/navsim/planning/script/config/common/agent/`, and the build copies `PACKAGE_ROOT/navsim` into the image. Model code and eval code must be the same code. |
| `assets/nurec/vocab/nurec_train_kmeans_4096x40x3.npy` | `vocab_size` is read from this file's shape at config time, and the file is baked into the image. |
| `README.md` in the package root | The build script checks for it before starting. |
| Training step / epoch, in writing | Maturity decides whether an evaluation is worth running at all -- see section 5. |

---

## 2. Send these too when the corresponding thing changed

| If you changed | Also send | What breaks otherwise |
|---|---|---|
| Only the weights (same code, same vocab) | Nothing else -- a commit hash confirming "no code or vocab change" is enough | -- |
| Model or ranking code | The full `navsim/` tree | A config key the eval code does not define is dropped silently or rejected. `pdm_rank_imi_normalize` was exactly this case: the yaml set it, our tree had no such field, and the ranking ran without the normalisation. |
| The vocabulary | The new `.npy` | The checkpoint's own `vocab` buffer and the shipped file disagree; `vocab_size` is read from the file, weights from the checkpoint. |
| `use_aux_heads`, `pdm_heads`, or any head set | The yaml reflecting it | Exact-checkpoint load fails at startup (by design, see section 4). |
| `sigma`, `soft_label_*`, or other label-side knobs | A note; they do not affect deployment | Nothing breaks, but we cannot interpret results without knowing. |

---

## 3. Do NOT send

```
ori_vocab_pdm_score / *.pkl     per-token 4096-candidate labels -- training only
prepared log pkl                training only
rectified 512x256 images        training only; the renderer produces frames live
exp/                            checkpoints aside, nothing there is used
```

Closed-loop evaluation renders its own observations from the NuRec USDZ scenes,
so none of the training data is required.

---

## 4. Constraints that must hold

### Input contract

These are fixed by `EVAL_INPUT_SPEC.md`. Changing any of them means changing the
evaluation environment as well, so tell us before you do.

```
rectified pinhole   K377, 512x256, distortion tuple present and all zero
cameras             CAM_L0 / CAM_F0 / CAM_R0, in that order, 3 only
history             2 Hz, t-1.0s / t-0.5s / t, 3 frames, 9 images per sample
route               use_route = true, 20 slots, 80 m horizon, 40 m near cutoff
vocabulary          (4096, 40, 3)
output              (40, 3), 0.1 s interval, 4 s horizon, rig (rear axle) frame
inference           model = teacher, use_first_stage_traj_in_infer = false
```

### Structural

- `pdm_heads` in the yaml must match the head set in the checkpoint. The driver
  runs with `DRIVESUPRIM_REQUIRE_EXACT_CHECKPOINT=1`, so a mismatch raises at
  startup instead of driving with silently missing weights.
- `state_dict` keys must keep the `agent.model.{teacher,student}.model...`
  layout.
- The vocabulary file and the checkpoint's `vocab` tensor must be the same
  clustering. Millimetre-level float differences are fine -- `load_state_dict`
  overwrites the buffer with the checkpoint's copy anyway -- but a different
  clustering is not.

---

## 5. Send a mature checkpoint, or say that it is not one

This is the single most expensive lesson from the last round. Two checkpoints
from the same data, same vocabulary, same labels and same ranking:

| | ep09 (step 4,840) | step43590 |
|---|---|---|
| Frames planning a stop (endpoint < 1 m) | 0 | 37 |
| Median planned endpoint | 38.87 m | 6.41 m |
| Behaviour at a standstill | drives off at 5-7 m/s | comes to a stop |

Every ranking-weight experiment run against ep09 was measuring under-training,
not ranking. If you send an early checkpoint for a smoke test, label it as one
and we will not draw conclusions from the score.

---

## 6. Suggested delivery layout

```
handoff/
├── checkpoint/<name>.ckpt
├── navsim/                                       the tree you trained with
├── assets/nurec/vocab/nurec_train_kmeans_4096x40x3.npy
├── README.md
└── TRAIN_INFO.txt      steps/epochs, what changed, commit hash,
                        and whether the ranking composition changed
```

Point `PACKAGE_ROOT` at that directory and it builds as-is. The previous
`axe_nurec_training_handoff` bundle had exactly this shape (checkpoint delivered
separately) and worked without modification.

---

## 7. What we check on arrival

```bash
# build
CHECKPOINT=<ckpt> CONFIG=<generated json> PACKAGE_ROOT=<handoff> IMAGE=<tag> \
  ./build_axe_nurec_planonly_ep09_driver.sh

# two lines at driver startup
[DriveSuprim] exact checkpoint load: missing=0 unexpected=0
[DriveSuprim] RESOLVED CONFIG: ... bev_num_cameras=3 bev_img=512x256 \
              vocab=4096 inference_model=teacher

# three lines while driving
[bevformer] BEV cells reached by any camera: 98.x%     below 95% means the
                                                       intrinsics and the images
                                                       are not a pair
route_valid=10                                         "route=off" means the
                                                       route never arrived
prediction: key=final_traj shape=(40, 3)
```

All five passing means the run is comparable with the earlier experiments.

A sixth check is worth running once per new architecture:
`python3 analyze_rank_contributions.py <driver_log>` prints, per ranking term,
its spread share, the driven candidate's rank under that term alone, and how
often removing the term changes the choice. That is how we established that
`imi` decides 90% of frames and that `gt_compliance` is never the deciding term.
