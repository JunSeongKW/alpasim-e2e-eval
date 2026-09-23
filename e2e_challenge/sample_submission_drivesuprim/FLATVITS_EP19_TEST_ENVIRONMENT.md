# flat ViT-S (nodistill ep19) — 테스트 환경 세팅

작성/검증일: 2026-09-03 (Asia/Seoul)

`nurec_flatvits_nodistill_ep19` 번들을 평가 환경에 통합한 기록입니다. 이전
체크포인트들과 달리 **아키텍처가 바뀌어** 드라이버 코드까지 손대야 했습니다.
결과는 [`FLATVITS_EP19_10CLIP_RESULTS.md`](FLATVITS_EP19_10CLIP_RESULTS.md).

---

## 1. 이 모델이 무엇이 다른가

DriveSuprim 원래 front-end 입니다. BEV 투영이 없고, 세 카메라를 파노라마로
이어 붙여 ViT-S 가 직접 읽습니다.

```
CAM_L0 / F0 / R0   각 512x256, K377 정류
  -> 스티칭         front 통째 + 좌우 view 에서 128px 씩 제거 = 1024x256 (정확히 4:1)
  -> ViT-S patch16  16 x 64 = 1024 토큰
  -> planner        이미지 1024 토큰 + route 20 토큰
  -> 4,096 후보를 네 헤드가 채점
```

체크포인트 실측:

```
epoch 19 / global_step 14520,  535 MB,  728 파라미터
헤드    NC / DAC / EP / imi / pdm_score   (gt_compliance 없음)
BEV 관련 파라미터 0개
백본 키 _backbone.image_encoder.vit....   (BEV 판은 .image_encoder.image_backbone.vit....)
route 파라미터 28개,  vocab (4096, 40, 3)
sha256  9f43a00619edfec15e5f44b5aa8942c312e382e5cd0fce8f756b270e44a22ef2
```

---

## 2. 패키지 조립

`assets/drivesuprim/flatvits_nodistill_ep19_eval/` 를 ep09 패키지에서 복제한 뒤:

```
번들 code/ 8개 파일로 덮어씀
  drivesuprim_{backbone_pe,features,config,model,loss_fn,agent}.py
  ssl_meta_arch.py
  vits_dinov3_backbone.py   -> navsim/agents/backbones/bevformer/
config/drivesuprim_agent_flatvits_planonly_nurec.yaml  -> navsim/.../common/agent/
checkpoint/epoch=19-step=14520.ckpt
```

번들 코드는 우리 트리보다 상위 버전이라 `_rank_product` / `pdm_rank_imi_normalize`
가 이미 들어 있었습니다.

---

## 3. 고쳐야 했던 것 다섯 가지

전부 조용히 틀릴 수 있는 것들입니다.

### 3.1 config — 상속 chain 이 아직 구 라벨

`drivesuprim_agent_flatvits_planonly_nurec.yaml` 은 front-end 만 바꾸고
채점은 base yaml 을 상속합니다. 그런데 그 base 는 아직 **4헤드(gt_compliance 포함)**
입니다. 그대로 두면 없는 헤드가 생겨 로드가 실패합니다.

번들 `docs/SCORING_AND_HEADS.md` 3~4장의 v6 채점 블록을 생성기에서 명시적으로
덮어썼습니다.

```yaml
pdm_heads: [no_at_fault_collisions, drivable_area_compliance, ego_progress]
pdm_rank_product: true
pdm_rank_product_terms: [same three]
pdm_rank_imi_normalize: true
pdm_imi_rank_weight: 1.0
pdm_aggregate_head: true          # 학습은 하되
pdm_aggregate_rank_weight: 0.0    # 랭킹엔 안 들어감
```

### 3.2 스티칭 — 드라이버 상수가 nuPlan 기준 (가장 위험했던 것)

드라이버의 레거시 경로는 nuPlan 1920x1080 을 전제로 세로 28행, 측면 416열을
잘랐습니다. NuRec 512x256 에 그대로 적용하면:

```
드라이버 기존   256 -> 200 (세로 28행씩), 측면 크롭 0
                -> 1536 x 200 을 1024x256 으로 리사이즈       화각도 종횡비도 다름
학습 코드       세로 0, 측면 (3W - 4H)/4 = 128
                -> 1024 x 256                                리사이즈 자체가 없음
```

학습의 프레임 유도 규칙을 그대로 옮긴 `_stitch_three_cameras()` 로 교체했습니다.
두 데이터셋 모두 검증했습니다.

```
NuRec  512x256   -> (256, 1024, 3)
nuPlan 1920x1080 -> (1024, 4096, 3)
```

이걸 놓치면 예외 없이 "그럴듯한 파노라마"가 만들어지고 점수만 조용히 떨어집니다.

### 3.3 route 입력 누락

레거시(비-BEV) 경로에는 route feature 를 넣는 코드가 아예 없었습니다. 이 모델은
`use_route: true` 라 `_route_inputs` 가 `KeyError` 를 던집니다.

BEV 경로와 같은 처리를 추가했습니다.

- `route_feature` / `route_mask` — 20슬롯, 미터 원값을 `/80` 정규화, gap 3.5~4.5 m 재검증
- status 벡터의 **command 4칸 제로화** — 학습(`_get_status_feature`)이 route 사용 시
  0으로 채우므로, 살아있는 command 를 넣으면 `_status_encoding` 이 본 적 없는 입력을 받음
- 진단 로그 `[DriveSuprim] FLAT INPUT: imgs=... route_valid=N`

### 3.4 백본 판별

`_detect_backbone` 이 체크포인트 키로 아키텍처를 추론하는데, flat 판은 ViT 가
`image_backbone` 없이 한 단계 위에 있어 어느 분기에도 걸리지 않았습니다.

```python
# .image_encoder.vit.patch_embed.proj.weight        -> vits_flat   (신규)
# .image_encoder.image_backbone.vit.patch_embed...  -> bevformer_m (기존)
```

두 문자열은 동시에 매칭될 수 없습니다.

### 3.5 이미지에 박힌 백본 타입

`Dockerfile.axe_nurec_vits` 가 `ENV DRIVESUPRIM_BACKBONE_TYPE=bevformer_m` 을 굽습니다.
판별 결과와 어긋나면 `checkpoint/backbone mismatch` 로 로드가 거부됩니다.
드라이버 실행 스크립트에 전달 통로를 넣었습니다.

```bash
DRIVESUPRIM_BACKBONE_TYPE=vits_flat  ./run_..._4gpu_drivers_foreground.sh
```

---

## 4. 번들에 빠진 모듈 — 대체본으로 처리

`drivesuprim_features.py` 가 `navsim.planning.data.nurec_rig_pose.tilt_for` 를 모듈
수준에서 import 하는데, 그 파일이 번들에도 이 서버 어디에도 없습니다.

호출 지점은 `_get_bev_camera_feature` 한 곳이고 `bev_pitch_correct` 가 켜졌을 때뿐
입니다. flat 모델은 BEV 경로를 타지 않고, 챌린지 드라이버는 애초에
`DriveSuprimFeatureBuilder` 를 쓰지 않습니다. import 만 해결되면 됩니다.

**identity 를 반환하지 않고 예외를 던지도록** 만들었습니다. "기울기 없음"을 돌려주면
BEV 모델이 피치 보정이 꺼진 채로 조용히 배포될 수 있고, 그건 캘리브레이션이 어긋난
카메라로 주행하는 것과 같습니다.

학습 담당자에게 이 파일을 요청하면 BEV 계열도 이 트리에서 돌릴 수 있습니다.

---

## 5. 검증 (기동 로그)

```
[DriveSuprim] RESOLVED CONFIG: backbone=vits_flat image_backbone=vits ...
              vocab=4096 feasibility=False inference_model=teacher topks=256
[DriveSuprim] exact checkpoint load: missing=0 unexpected=0
[DriveSuprim] FLAT INPUT: imgs=(3, 256, 1024) x2 status=2 route_valid=10
```

`missing=0 unexpected=0` 이 3.1 의 config 가 체크포인트와 맞는다는 증거입니다.
`gt_compliance` 를 남겼다면 여기서 예외로 죽습니다.

---

## 6. 실행 중 만난 인프라 문제

### 6.1 harmonizer 다운로드로 첫 실행이 타임아웃

첫 시도는 `DEADLINE_EXCEEDED` 로 죽었습니다. 렌더러 10개가 각자
`nvidia/DiffusionHarmonizer` (`harmonizer_nontemporal.pt`) 를 HuggingFace 에서
받느라 기동에 11분 넘게 걸려 `SERVICE_STARTUP_TIMEOUT_SEC=900` 을 넘겼습니다.

이 모델은 NuRec 의 Gaussian Splat 렌더에서 나오는 아티팩트를 지우는 후처리이고,
`base_config.yaml:141` 의 `--enable-harmonizer` 로 **모든 평가에서 기본 활성**입니다.
공식 프리셋도 같으므로 끄면 입력 분포가 달라져 점수를 비교할 수 없습니다.

대응: `SERVICE_STARTUP_TIMEOUT_SEC=2400` 으로 재실행 → 정상 완료.

캐시가 컨테이너 내부(`/home/.cache/nre/harmonizer/`)라 컨테이너마다 반복됩니다.
`services.renderer.volumes` 에 `~/.cache/nre:/home/.cache/nre` 를 추가하면 매 실행
10분가량 절약됩니다. (미적용 — 렌더러 볼륨은 평가 조건에 속해 임의로 바꾸지 않았습니다.)

### 6.2 동시 실행을 위한 포트 분리

`network_mode: host` 라 두 스택을 같이 돌리면 6000번대가 충돌합니다.
`WIZARD_BASEPORT` 통로를 추가해 스택마다 100씩 띄웁니다.

또 드라이버 컨테이너 이름이 `<prefix>-gpu<N>` 이라 GPU 를 반복 지정하면 이름이
겹칩니다. 접두사를 나눠(`ds-fa` / `ds-fb` / `ds-fc`) 10개를 띄웠습니다.

---

## 7. 재현

```bash
cd /home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim
N=e2e_challenge/sample_submission_drivesuprim
P=$N/assets/drivesuprim/flatvits_nodistill_ep19_eval

# 빌드
PACKAGE_ROOT=$P CHECKPOINT=$P/checkpoint/epoch=19-step=14520.ckpt \
CONFIG=$P/flatvits_config.json \
IMAGE=alpasim-e2e-drivesuprim-flatvits-nodistill:ep19 \
  $N/build_axe_nurec_vits_driver.sh

# 드라이버 (백본 타입 지정이 필수)
IMAGE=alpasim-e2e-drivesuprim-flatvits-nodistill:ep19 \
EXPECTED_CHECKPOINT_SHA256=9f43a006... DRIVESUPRIM_BACKBONE_TYPE=vits_flat \
GPU_INDICES_CSV=0,1,2,3 DRIVER_PORTS_CSV=6860,6861,6862,6863 CONTAINER_PREFIX=ds-fa \
  $N/run_axe_nurec_planonly_ep09_4gpu_drivers_foreground.sh

# 평가
IMAGE=<위와 동일> EXPECTED_CHECKPOINT_SHA256=<위와 동일> RUN_DIR=$PWD/runs/... \
SCENE_IDS_CSV=$(paste -sd, $N/flatvits_10clips_seed20260903.txt) \
DRIVER_PORTS_CSV=... RENDER_GPUS_CSV=... ROLLOUT_WORKERS=10 \
NRE_CACHE_SIZE=1 RENDER_VIDEO=true WIZARD_BASEPORT=6800 \
SERVICE_STARTUP_TIMEOUT_SEC=2400 \
  $N/run_axe_nurec_planonly_ep09_debug_video_eval.sh
```
