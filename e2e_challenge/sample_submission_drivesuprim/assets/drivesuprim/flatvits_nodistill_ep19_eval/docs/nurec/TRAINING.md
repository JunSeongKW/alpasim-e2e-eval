# NuRec training pipeline

The production path now consumes the ClipGT map embedded in each original
NuRec USDZ.  A clip-local `AbstractMap` adapter supplies lane polygons,
lane/roadblock topology, intersections, crosswalks, and the expert route to
AXE's unmodified metric cache and EPDMS scorer.

```text
original USDZ ──→ compact map bundle ──────────────┐
                                                  ├─→ map-aware NAVSIM logs
existing NuRec NAVSIM logs ───────────────────────┘          │
                                                             ├─→ metric cache
NuRec trajectory vocabulary ─────────────────────────────────┘       │
                                                                     └─→ EPDMS → training
```

## 1. Production USDZ path

The raw reconstruction and image assets are not copied.  Only the small
`clipgt/*.parquet` map payload is read from each USDZ and written as one map
bundle per clip.

```bash
export NUREC_USDZ_ROOT=/storage/nurec/26.04_release
export NUREC_INPUT_LOG_ROOT=/storage/nurec_navsim_all/navsim_logs/nurec
export NUREC_SENSOR_ROOT=/storage/rendered_nurec_images
export NUREC_PREPARED_ROOT=/fast-local/nurec_axe

# Defaults to two concurrent USDZ readers to protect shared storage.
NUREC_MAP_JOBS=2 bash scripts/nurec/prepare_usdz.sh
```

This writes:

- `maps/<clip-id>.pkl`: compact ClipGT vector map and matched expert route
- `navsim_logs/trainval/*.pkl`: source logs with `map_location=nurec:<clip-id>`
  and per-frame future `roadblock_ids`
- `index.json`: all converted logs under `train`, a reproducibly sampled `val`
  subset, and the sensor/map roots

Preparation fails if any source log has no corresponding map bundle.  It never
overwrites the existing converted logs.

NuRec에는 별도 hold-out partition을 만들지 않습니다. 변환에 성공한 모든
log가 학습에 사용되고, 기본적으로 그중 10%를 validation에도 중복
사용합니다. 샘플은 seed로 재현 가능합니다.

```bash
NUREC_VAL_FRACTION=0.10 NUREC_SPLIT_SEED=2026 \
  bash scripts/nurec/prepare_usdz.sh
```

The primary geometry comes from `lane.parquet` and `association.parquet`.
`ROAD_SEGMENT_SIBLING_LANE` creates roadblocks, intersection-linked lanes create
roadblock connectors, and `NEXT_LANE` creates the graph.  Ego poses from
`egomotion_estimate.parquet` are map-matched to produce the ordered route.

The training contract is fixed at 0.5 seconds (`500,000us`). Preparation checks
timestamp spacing, map bundles, and routes. `train.sh` additionally verifies
that every selected timestamp has rendered `CAM_L0/F0/R0` images before
starting DDP. Image checks can be run independently while rendering progresses:

```bash
python -m navsim.planning.data.nurec_validate \
  --log-root $NUREC_PREPARED_ROOT/navsim_logs/trainval \
  --sensor-root $NUREC_SENSOR_ROOT \
  --map-root $NUREC_PREPARED_ROOT/maps
```

The NuRec setup is a strict three-camera configuration. Pixel-space rotation
augmentation is disabled (`only_ori_input=true`) because L0/F0/R0 do not cover
the extra side FoV needed to crop a yaw-rotated panorama. Enabling it fails
early instead of silently requiring L1/R1 images.

The renderer writes 10 Hz PNGs with a small timestamp offset from the NAVSIM
log. Sync selects the nearest rendered image for each 0.5-second frame (maximum
allowed error: 60 ms) and maps camera names as follows:

```text
camera_cross_left_120fov  -> CAM_L0
camera_front_wide_120fov  -> CAM_F0
camera_cross_right_120fov -> CAM_R0
```

Run this repeatedly while rendering progresses; completed clips are attached
and incomplete clips are reported without failing:

```bash
bash scripts/nurec/sync_images.sh
```

## 2. Generate EPDMS labels

Use the DriveSuprim-range-filtered vocabulary first:

```bash
export NUREC_VOCAB_PATH=/home1/irteam/hyeonseo_workspace/workspaces/satyam/nurec_vocab/nurec_train_kmeans_8192x40x3_drivesuprim_range.npy
export NAVSIM_TRAJPDM_ROOT=/storage/nurec/traj_pdm

NUREC_PDM_THREADS=32 bash scripts/nurec/generate_epdms.sh
```

The script first builds the standard AXE metric cache, then scores all 40-step
vocabulary trajectories.  The result is:

```text
$NAVSIM_TRAJPDM_ROOT/ori/vocab_score_8192_nurec/nurec.pkl
```

Set that path as `NUREC_ORI_PDM_SCORE` for training.

## 3. Train

### ViT-S BEVFormer full Stage 1 -> 2 -> 3 curriculum

Use `train_staged.sh` for the full gated model. The stage transition is a
**weights-only** load through `CKPT`; optimizer, epoch, and scheduler state start
fresh in every new stage. `RESUME_CKPT` is only for resuming an interrupted run
of the same stage.

```bash
source setup_nurec.sh

# Stage 1: 20 epochs, perception pre-training (planning loss/gate off)
NUM_GPUS=4 BATCH_SIZE=8 NUM_WORKERS=8 \
  bash scripts/nurec/train_staged.sh stage1

# Pick the Stage-1 checkpoint by validation loss / perception metrics.
CKPT="$NAVSIM_EXP_ROOT/drivesuprim_bevformer_vov_v2_vits_stage1_ckpt/epoch=XX-step=YYYY.ckpt" \
NUM_GPUS=4 BATCH_SIZE=8 NUM_WORKERS=8 \
  bash scripts/nurec/train_staged.sh stage2

# Pick the Stage-2 checkpoint, then fine-tune end-to-end for 30 epochs.
CKPT="$NAVSIM_EXP_ROOT/drivesuprim_bevformer_vov_v2_vits_stage2_ckpt/epoch=XX-step=YYYY.ckpt" \
NUM_GPUS=4 BATCH_SIZE=8 NUM_WORKERS=8 \
  bash scripts/nurec/train_staged.sh stage3
```

The defaults reproduce the repository's staged recipe: `20 / 5 / 30` epochs,
`lr=1e-4`, cosine schedule length 30 in every stage, AdamW, gradient clipping,
and `bf16-mixed`. Stage 2 freezes the shared image/BEV perception trunk while
the auxiliary heads and planner train; Stage 3 unfreezes the trunk. The
feasibility gate is off only in Stage 1.

For a command/preflight check without launching GPUs:

```bash
NUREC_DRY_RUN=1 bash scripts/nurec/train_staged.sh stage1
```

To continue an interrupted stage, do not also set `CKPT`:

```bash
RESUME_CKPT=/path/to/same-stage.ckpt \
  bash scripts/nurec/train_staged.sh stage2
```

The launcher rejects Torch `<2.1` and fp16 because both are known-bad on the
H200 path, verifies every training token has its paired PDM score, and checks
that Stage 1 sees real supported object boxes before it can train an accidental
all-background detector. The current NuRec labels contain vehicle, pedestrian,
bicycle, and generic-object boxes; the seven-class head remains checkpoint- and
architecture-compatible, with absent classes simply receiving no positive
examples.

The NuRec map geometry limitation documented below also affects Stage-1
drivable-area segmentation. It is not repaired by the staged curriculum.

### Generic/legacy launcher

```bash
export NUREC_ORI_PDM_SCORE=$NAVSIM_TRAJPDM_ROOT/ori/vocab_score_8192_nurec/nurec.pkl
NUM_GPUS=8 BATCH_SIZE=8 bash scripts/nurec/train.sh drivesuprim_agent_r34
```

NuRec does not expose a reliable traffic-light phase for every frame.  The
generated score files therefore retain `traffic_light_compliance=1.0` as the
neutral multiplicative value, while `scripts/nurec/train.sh` disables both its
BCE loss and its trajectory-ranking contribution.  The prediction head remains
in the model so existing DriveSuprim checkpoints stay load-compatible.

`train.sh` reads both sensor and map roots from `index.json`; manually exporting
`NUREC_MAP_ROOT` is not necessary.

## Known map limitations

- Static traffic-light geometry exists, but a reliable time-varying
  red/yellow/green state has not been identified.  It remains empty rather than
  being fabricated.
- `drivable_space.parquet` is used when present.  The current PDM implementation
  primarily scores the union of roadblock/lane and intersection polygons, so
  clips without it still have a valid drivable map.
- Roadblock grouping is derived from ClipGT associations and should be audited
  against `map.xodr` on the full release before treating labels as final.

## Legacy portable-manifest path

AXE consumes NAVSIM scene objects, not bare images. This adapter keeps the AXE
model, feature builders, targets, and trainer unchanged and converts a portable
NuRec JSONL export to that scene contract. Sensor files remain on the storage
server and are not duplicated.

### Export contract

Write one JSON object per line (one line is one continuous log):

```json
{"log_name":"run_001","split":"train","map_location":"us-nv-las-vegas-strip","frames":[{"timestamp_us":0,"ego_translation":[0,0,0],"ego_yaw":0,"ego_velocity":[0,0],"ego_acceleration":[0,0],"driving_command":[1,0,0,0],"roadblock_ids":[],"cameras":{"cam_f0":{"image_path":"run_001/f0/000.jpg"},"cam_l0":{"image_path":"run_001/l0/000.jpg"},"cam_l1":{"image_path":"run_001/l1/000.jpg"},"cam_l2":{"image_path":"run_001/l2/000.jpg"},"cam_r0":{"image_path":"run_001/r0/000.jpg"},"cam_r1":{"image_path":"run_001/r1/000.jpg"},"cam_r2":{"image_path":"run_001/r2/000.jpg"},"cam_b0":{"image_path":"run_001/b0/000.jpg"}},"annotations":{"boxes":[],"names":[]}}]}
```

Required facts are `log_name`, `split`, `map_location`, ordered frames, ego pose,
and all eight camera paths. Paths are relative to `NUREC_SENSOR_ROOT`. Rotation
is either `ego_yaw` in radians or `ego2global_rotation` as `[w,x,y,z]`.
Timestamps must be increasing; training expects 0.5 s spacing. Each log must have
at least 12 frames because the supplied split uses 4 history + 8 future frames.
The manifest must include both train and validation logs.

In the legacy mode, `map_location` must name an installed nuPlan map. For actual
NuRec local maps, use the production USDZ path above; do not silently substitute
an unrelated city map.

### Prepare and train

```bash
export NUREC_MANIFEST=/storage/nurec/axe_manifest.jsonl
export NUREC_SENSOR_ROOT=/storage/nurec/sensors
export NUREC_PREPARED_ROOT=/fast-local/nurec_axe
export NUREC_ORI_PDM_SCORE=/storage/nurec/artifacts/ori_vocab_pdm_scores.pkl

bash scripts/nurec/prepare.sh
NUM_GPUS=8 BATCH_SIZE=8 bash scripts/nurec/train.sh drivesuprim_agent_r34
```

The prepare step validates paths and writes `navsim_logs/trainval/*.pkl` plus an
`index.json` containing the split. Re-running it updates logs deterministically.
Use `--skip-file-check` only to inspect metadata before the storage mount exists.
Normal Hydra overrides can follow the agent argument, for example:

```bash
bash scripts/nurec/train.sh drivesuprim_agent_r34 trainer.params.fast_dev_run=true
```

DriveSuprim always needs an original, token-aligned vocabulary PDM-score pickle;
the launcher deliberately fails early if `NUREC_ORI_PDM_SCORE` is absent. Its
shape is the same as the existing NAVSIM artifact, but keys must be the tokens
generated for this NuRec export. Generate it through the existing metric/PDM
scoring pipeline after the NuRec map adapter is available. The launcher starts
with `only_ori_input=true`. Rotation-ensemble training additionally needs the
offline augmentation JSON and rotated per-token PDM scores; regenerate both for
the converted tokens before overriding `only_ori_input=false`.

---

## driving_command 4 슬롯을 왜 0으로 남겨두는가 (2026-08-26 결정: 유지)

`use_route=True` 면 `drivesuprim_features.py:136-138` 이 driving_command 를
`zeros_like` 로 덮어쓴다. 그래서 `_status_encoding = nn.Linear(8, tf_d_model)`
의 앞 4 열은 항상 0 을 곱하는 죽은 가중치가 된다. **의도된 것이고 그대로 둔다.**

### 악영향이 없다는 근거 (전수 확인함)

- **forward** — `num_ego_status=1`, 레이아웃은 `[cmd(4) | vel(2) | acc(2)]` 연속.
  출력 = `W[:,4:8] @ [vel,acc] + b`. 앞 4 열 기여 0.
- **backward** — `∂L/∂W[:,0:4] = grad_out ⊗ 0 = 0`. loss 로 업데이트되지 않는다.
- **출력 이후** — `status_encoding` 은 정규화 없이 `drivesuprim_model.py:1074`
  의 `tr_out + status_encoding.unsqueeze(1)` 로 직행. 죽은 열이 스케일에
  끼어들 자리가 없다.
- **소비자** — `driving_command` 를 읽는 곳은 `drivesuprim_features.py:131`
  단 한 곳. `status_feature[:, 0:4]` 를 읽는 코드는 없다.
- **weight_decay 0.01** 이 걸려 있어 AdamW 가 gradient 와 무관하게 그 4 열을
  매 스텝 0 으로 감쇠시킨다. NuRec 학습을 돌리면 물리적으로도 0 에 수렴한다.
  삭제하든 안 하든 도달점이 같다.

### 유지하는 이유

`_status_encoding` 이 `[256,8]` 로 남아 있어야 NAVSIM 시대 체크포인트 6 종이
그대로 로드된다. 백본 비교표가 성립하는 이유가 정확히 이것이다.

### 굳이 제거한다면 — 세 곳이 한 세트다

하나라도 빠지면 조용히 틀어진다.

1. `drivesuprim_model.py:388` — shape 를 `use_route` 조건부로.
   NAVSIM 평가 6 종은 전부 `use_route=false` 라 cmd 를 정상 사용 중이다.
   무조건 줄이면 그 평가들이 깨진다.
2. `drivesuprim_features.py:136` — cmd 슬롯을 아예 만들지 않는다.
3. `drivesuprim_agent.py:143-147` 로더 — **이게 핵심.** 지금 로더는 shape
   불일치 텐서를 *통째로 버린다*. shape 만 줄이면 살아 있는 vel/acc 가중치까지
   같이 날아가 fresh init 이 된다. 슬라이스가 반드시 같이 가야 한다:

   ```python
   k = "_status_encoding.weight"
   if k in state_dict and state_dict[k].shape[1] == 8 and own[k].shape[1] == 4:
       state_dict[k] = state_dict[k][:, 4:]   # 레이아웃이 연속이라 정확히 등가
   ```

### 남아 있는 함정

지금 상태에서 누가 `use_route=false` 로 되돌리면 **학습된 적 없는 cmd 가중치가
조용히 부활한다.** NuRec 학습분을 `use_route=false` 로 평가하지 말 것.
