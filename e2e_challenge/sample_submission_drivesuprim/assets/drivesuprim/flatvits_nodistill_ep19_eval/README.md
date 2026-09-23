# NuRec plan-only 모델 인계 문서

캘리브레이션이 정상화된 ViT-S BEVFormer plan-only 모델을 다른 환경에서 돌리기
위한 코드와 설명입니다. 체크포인트는 별도로 전달받으십시오.

이 문서는 **먼저 읽어야 하는 순서대로** 쓰여 있습니다. 2장은 건너뛰지
마십시오 — 이 프로젝트에서 30에폭짜리 학습 하나를 통째로 버리게 만든 함정이고,
데이터를 직접 준비하는 경우 똑같이 재현됩니다.

---

## 1. 이 모델이 하는 일

카메라 3대와 자차 상태, 경로를 받아 **4,096개 후보 궤적 중 하나를 고릅니다.**
궤적을 회귀로 생성하지 않고 고정된 vocabulary에서 순위를 매겨 선택하는
구조입니다.

점수는 AlpaSim scene score입니다:

```
score = NC × DAC × GT × EP
```

| 항목 | 뜻 |
|---|---|
| `NC` | no-at-fault collision — 자기 과실 충돌이 없으면 1 |
| `DAC` | drivable area compliance — 주행 가능 영역을 벗어나지 않으면 1 |
| `GT` | gt compliance — 기록된 사람 주행에서 횡방향 4 m 이상 벗어나면 0 |
| `EP` | ego progress — 사람 대비 전진량 비율 |

NAVSIM의 EPDMS 8개 항목이 아닙니다. 이 모델은 위 4개 헤드만 가지고 있고,
나머지 4개(lane keeping, driving direction, history comfort, TTC)와
traffic light는 **아예 파라미터가 존재하지 않습니다.** NuRec 데이터에서 그
항목들이 측정이 아니라 상수이거나 맵 결함을 재는 값이었기 때문입니다
(자세한 근거는 agent YAML 상단 주석에 있습니다).

랭킹 점수는 다음과 같이 결합됩니다:

```
0.3·log softmax(imi) + 8.0·log s(NC) + 8.0·log s(DAC) + 8.0·log s(GT) + 1.0·log s(EP)
```

게이트 항목에 전진보다 8배 큰 가중치가 붙습니다. 이 계수들은 **선택에만**
영향을 주고 최종 점수 계산에는 관여하지 않습니다.

---

## 2. 가장 중요한 것 — 캘리브레이션 계약

### 무엇이 문제였나

이 모델은 BEVFormer로 이미지를 BEV 격자에 투영합니다. 그 투영은
`lidar2img = K @ inv(cam2lidar)` 로 만들어지고, `K`는 로그의
`cam_intrinsic`에서 옵니다.

로그에 담긴 `cam_intrinsic`이 **실제 이미지 파일과 다른 해상도의 것**이면,
투영은 조용히 틀립니다. 예외도 나지 않고, 로스도 정상적으로 내려가고,
점수도 그럴듯하게 나옵니다.

이 프로젝트에서 실제로 벌어진 일:

* 로그에는 nuPlan의 핀홀 (fx=fy=1545, cx=960, cy=560, 1920×1080)이 들어 있었고
* 이미지 파일은 정류된 512×256이었습니다
* 주점(cx=960)이 512픽셀 폭 이미지 바깥에 있으니
* **BEV 참조점의 0.89%만 이미지 안에 떨어졌고**
* `SpatialCrossAttention`은 마스크가 전부 False면 residual만 돌려주므로
* **BEV 피쳐에 이미지 정보가 문자 그대로 0이 되었습니다**

30에폭을 학습하고 0.9259라는 그럴듯한 점수까지 받은 뒤에야 발견했습니다.
검증 방법: 그 체크포인트에 실제 이미지를 넣은 결과와 **순수한 검은 화면**을
넣은 결과가 비트 단위로 동일했습니다.

### 계약

**로그의 `cam_intrinsic`은 그 로그가 가리키는 이미지 파일의 것이어야 합니다.**

이 저장소의 정류 목표는 `navsim/planning/data/nurec_rectify/target.py`에
있습니다:

```
TARGET_WIDTH  = 512     TARGET_FX = 377.0     TARGET_CX = 256.0
TARGET_HEIGHT = 256     TARGET_FY = 377.0     TARGET_CY = 128.0
```

등방(fx == fy)이고 주점이 이미지 중앙입니다. 이 값을 고른 이유
(비등방 왜곡 제거, 카메라 사이 사각지대 폐쇄)는 그 파일 docstring에 측정치와
함께 적혀 있습니다.

### 방어 장치

`navsim/agents/drivesuprim/drivesuprim_features.py`의
`_assert_intrinsics_match_image()`가 매 샘플마다 주점이 이미지 범위 안에
있는지 확인하고, 아니면 `CalibrationMismatch`를 던집니다. 이 예외는
데이터로더의 재시도 래퍼에서 **재시도하지 않고 그대로 올라옵니다** — 스플릿
전체가 같은 캘리브레이션을 쓰므로 재시도는 메시지를 묻을 뿐입니다.

가드가 있어도 **직접 확인하십시오.** 가드는 주점이 이미지 밖에 있는 경우만
잡습니다. 주점이 안에 있으면서 초점거리만 틀린 경우는 통과합니다.

### 확인 방법

학습이나 채점을 시작하기 전에 이 한 줄을 보십시오:

```
[bevformer] BEV cells reached by any camera: 98.438%  (cell-camera-anchor mean 32.812%)
```

* **앞 숫자(any camera)** 가 격자 커버리지입니다. 이 장비 구성에서 **98% 근처**가
  정상입니다. 도달하지 못하는 셀은 차 앞 5 m 이내뿐이고, 정류 이미지의 수직
  화각 37.5도에서 나오는 물리적 사각지대입니다.
* **뒤 숫자(mean)** 는 (셀 × 카메라 × z앵커) 삼중항 비율이라 구조적으로 낮습니다.
  카메라 3대 중 보통 1대가 한 셀을 보므로 약 ⅓입니다. **커버리지로 읽지
  마십시오.**
* **어느 쪽이든 0에 가까우면 캘리브레이션과 이미지가 짝이 맞지 않는 것입니다.**

더 강한 검사는 5장의 체크리스트에 있습니다.

---

## 3. 환경

### 검증된 조합

```
python              3.9.25
torch               2.1.2+cu118
pytorch-lightning   2.2.1
numpy               1.23.4
hydra-core          1.2.0
opencv-python       4.9.0
shapely             2.0.7
pandas              2.3.3
nuplan-devkit       v1.2 (pip)
```

`requirements.txt`는 `torch==2.0.1`로 핀되어 있지만 **실제로 검증된 건
2.1.2+cu118**입니다. H200에서는 2.0.1이 아닌 이 조합을 쓰십시오.

### 설치

```bash
tar xzf axe-nurec-planonly.tar.gz
cd axe-nurec-planonly
pip install -r requirements.txt        # nuplan-devkit 포함
```

`navsim`은 패키지로 설치하지 않고 `PYTHONPATH`로 잡습니다.
`setup_nurec.sh`가 처리합니다.

설치가 끝나면 **데이터 없이** 코드가 온전한지부터 확인하십시오:

```bash
python scripts/nurec/verify_bundle.py
```

정상이면 이렇게 나옵니다:

```
navsim      .../axe-nurec-planonly/navsim/__init__.py
vocabulary  (4096, 40, 3)  nurec_train_kmeans_4096x40x3.npy
rectify     fx=377 fy=377 cx=256 cy=128  @ (512, 256)
model       43.5M parameters
geometry    bev 56x56   cameras 3   frames 3   image 512x256
heads       ['no_at_fault_collisions', 'drivable_area_compliance', 'gt_compliance', 'ego_progress']
imports     CalibrationMismatch guard, NuRecSimulator (MPC)

OK -- the bundle is intact.
```

### osqp 는 필요 없습니다

채점이 쓰는 MPC(`NuRecSimulator` → `batch_linear_mpc.py` → `batch_qp.py`)는
**순수 torch**입니다. osqp는 `nurec_controller/reference/linear_mpc.py`
하나만 쓰고, 그건 비교용 레퍼런스 구현이라 채점 경로에 들어오지 않습니다.
레퍼런스 컨트롤러를 직접 돌릴 때만 osqp **1.x**가 필요합니다 (0.6.x는
`Workspace already setup!`으로 깨집니다).

### GPU

H200에서는 backward 중 CUDA illegal instruction이 나는 조합이 있어
`drivesuprim_flash` 계열 환경이 필요했습니다. forward만 돌려보면 드러나지
않으니, 재학습할 계획이면 **backward까지 포함한 1스텝 스모크런**을 먼저
돌리십시오.

혼합정밀도는 **`bf16-mixed`** 를 쓰십시오. `fp16`은 이 파이프라인에서 재현
가능하게 NaN이 납니다 (bf16에는 GradScaler가 없습니다).

---

## 4. 데이터

### 레이아웃

```
$NUREC_PREPARED_ROOT/          기본값 data/prepared
  index.json                   클립 목록, train/val 분할, num_frames
  navsim_logs/trainval/*.pkl   클립당 로그 하나 (프레임 리스트)
  metric_cache/                채점이 읽는 캐시
  rectified_logs.json

$NUREC_SENSOR_ROOT/            기본값 data/images
  <clip-uuid>/CAM_F0/*.jpg     정류된 512×256
  <clip-uuid>/CAM_L0/*.jpg
  <clip-uuid>/CAM_R0/*.jpg
```

**`NUREC_PREPARED_ROOT`와 `NUREC_SENSOR_ROOT`는 반드시 짝이 맞는 쌍이어야
합니다.** 캐시를 다른 곳에서 쓰려고 루트를 옮기는 순간 2장의 사고가 그대로
재현됩니다. 캐시만 옮기고 싶으면 `NUREC_METRIC_CACHE_ROOT`를 쓰십시오.

### 카메라는 3대뿐입니다

`bev_num_cameras: 3` → `[CAM_L0, CAM_F0, CAM_R0]`.

같은 로그에 다른 5대(CAM_L1/L2/R1/R2/B0)의 항목이 남아 있을 수 있지만
**그것들은 정류되지 않았고 여전히 fx 1545 / cx 960 / cy 560을 들고
있습니다.** 절대 켜지 마십시오.

### 좌표계

**뒷축(rear axle)이 기준입니다.** vocabulary 궤적, GT 경로, 예측 궤적이
전부 뒷축 기준이고, 박스 중심은 그보다 1.47 m 앞에 있습니다. 다른 좌표계에서
온 궤적을 그대로 넣으면 GT가 조용히 나빠집니다.

### 맵 결함 (알고 있어야 함)

NuRec 맵에는 `DRIVABLE_AREA` 레이어가 **0개**입니다. 주행 가능 영역은
`[LANE, LANE_CONNECTOR]` 폴리곤 합집합으로 만듭니다
(`aux_bev_drivable_layers`). 이 폴리곤들은 서로 완전히 맞물리지 않아
격자의 0.309%에 구멍이 생기고, 프레임의 25.8%가 구멍을 하나 이상 포함합니다.
구멍은 가장 가까운 폴리곤에서 최대 1.264 m 거리입니다.

`INTERSECTION` 레이어를 합치면 그 구멍의 0.19%를 메우는 대신 1.317%를 새로
칠하게 되어(7배 손해) **쓰지 않기로 했습니다.**

### 검증 분할이 학습과 겹칩니다

`index.json`의 `validation_sampling.overlaps_train: true`입니다. 기본 val
분할은 train 클립에서 30%를 뽑은 **부분집합**입니다.

**따라서 기본 설정으로 나오는 점수는 일반화 수치가 아닙니다.** 같은 조건의
두 체크포인트를 비교하는 데는 유효하지만, 외부에 보고할 숫자라면 별도의
홀드아웃 로그 목록을 `--log-names`로 지정하십시오.

---

## 5. 모델 실행

### 채점

```bash
source setup_nurec.sh
bash scripts/nurec/score_epdms.sh /path/to/checkpoint.ckpt
```

읽어야 할 컬럼은 **`score`** 입니다. CSV에 있는 `pdms` / `pdms_v1`은 옛 가중합이고
이 모델이 학습한 대상이 아닙니다. 점수에 안 들어가는 항목들이 1.0으로 고정된
채 가중평균에 들어가므로 **훨씬 높게 나옵니다.** 인용하지 마십시오.

주요 환경변수:

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `SCORE_SPLIT` | `val` | `val` \| `train` |
| `SCORE_THREADS` | 24 | Ray 시뮬레이션 스레드 |
| `SCORE_BATCH` | 8 | 예측 단계 배치 |
| `SCORE_WORKERS` | 8 | 데이터로더 워커 |
| `SCORE_SAVE_PICKLE` | 0 | 1이면 예측 궤적을 `<ckpt-dir>/<ckpt>.pkl`로 저장 (비디오에 필요) |
| `NUREC_USE_ROUTE` | `true` | **학습과 반드시 일치시킬 것** |
| `NUREC_USE_MPC` | 1 | 1=MPC, 0=LQR |

### 컨트롤러는 MPC 여야 합니다

hydra 기본값은 LQR입니다. LQR로 돌리면 **기록된 사람 주행조차 자기 경로를
못 따라가서** GT가 92%까지 떨어집니다. `score_epdms.sh`가
`NUREC_USE_MPC=1`일 때 `NuRecSimulator`로 바꿔줍니다. 직접
`run_pdm_score_*`를 호출한다면 이 override를 손수 넣어야 합니다.

### route 설정을 맞추십시오

`use_route=True`면 `_status_encoding`의 앞 네 컬럼이 route에서 오고,
`False`면 실제 driving command가 그 자리에 들어갑니다. 학습과 채점이
어긋나면 조용히 다른 입력을 먹입니다. 이 모델은 **`use_route=true`로
학습되었습니다.**

### 비디오

```bash
source setup_nurec.sh
bash scripts/nurec/render_val40.sh /path/to/checkpoint.ckpt <출력이름>
```

`score_video.py`는 모델을 다시 돌리지 않고 채점이 저장한 pickle에서 궤적을
읽어 그립니다 — 화면과 CSV가 어긋날 수 없게 하려는 설계입니다. 그래서
pickle이 없으면 `render_val40.sh`가 `SCORE_SAVE_PICKLE=1`로 한 번
채점합니다.

---

## 6. 신뢰하기 전에 돌릴 검사

숫자를 믿기 전에 이 네 가지를 확인하십시오. 전부 모델과 무관하게, 데이터만으로
검증됩니다.

**1) 캘리브레이션 — 커버리지 한 줄**

채점 로그에서 `BEV cells reached by any camera`가 **95% 이상**인지 봅니다.
0에 가까우면 2장으로 돌아가십시오.

**2) 캘리브레이션 — 독립 재계산**

```bash
python scripts/nurec/verify_rectified_projection.py
```

**3) 카메라·프레임이 실제로 융합되는지**

```bash
python scripts/nurec/check_bev_fusion.py 4
```

입력을 하나씩 지우고 BEV가 얼마나 변하는지 잽니다. 정상이면 이렇게 나옵니다:

```
CAM_L0  0.1665      CAM_F0  0.8527      CAM_R0  0.1797
frame t-2  0.2566   t-1  0.2457   t-0(현재)  0.5966
```

세 카메라의 민감도 로브가 각각 좌·중앙·우에 **좌우 대칭으로** 나타나야
합니다. 어느 하나가 0이면 그 입력은 쓰이지 않고 있습니다.

**4) BEV에 이미지가 실제로 들어가는지**

```bash
python scripts/nurec/visualize_bev_features.py 6
```

`camera sensitivity`가 **1.0 근처**여야 합니다. 이 값이 0이면 2장의 사고가
재현된 것입니다. 참고로 캘리브레이션이 깨진 구 모델은 정확히 **0.00**이었고,
정상 모델은 **1.15**입니다.

**5) 테스트**

```bash
pytest tests/test_intrinsics_match_image.py \
       tests/test_nurec_rectified_geometry.py \
       tests/test_rear_axle_reference.py -q
```

---

## 7. 이 모델의 실측치

캘리브레이션 수정 전후를 같은 스플릿·같은 라벨·같은 스케줄로 비교한
결과입니다. 두 런의 유일한 의미 있는 차이는 카메라 투영입니다.

**검증셋 점수 (train과 겹치는 분할 — 런 추적용)**

```
epoch   score        NC       DAC        GT        EP
    0  0.7913    0.9547    0.9143    0.9781    0.9036
    5  0.9509    0.9929    0.9797    0.9961    0.9774
   10  0.9533    0.9933    0.9768    0.9965    0.9814
   15  0.9660    0.9961    0.9792    0.9976    0.9898
   20  0.9727    0.9971    0.9835    0.9988    0.9913
   25  0.9727    0.9972    0.9809    0.9985    0.9938
--------------------------------------------------------
구 30에폭(캘리브 깨짐)
       0.9250    0.9849    0.9682    0.9966    0.9706
```

**입력 ablation (에폭 20 모델, 같은 가중치에서 신호만 제거)**

```
                정상      이미지 차단        route 차단
score          0.9727    0.7836 (-0.189)   0.8392 (-0.134)
NC             0.9971    0.8910 (-0.106)   0.9832 (-0.014)
DAC            0.9835    0.9158 (-0.068)   0.8910 (-0.093)
GT             0.9988    0.9507 (-0.048)   0.9084 (-0.090)
EP             0.9913    0.9808 (-0.011)   0.9825 (-0.009)
```

읽는 법: **이미지는 NC를, route는 DAC와 GT를 담당합니다.** EP는 둘 다 거의
필요로 하지 않습니다 (전진량은 자차 운동에서 나옵니다). 두 신호가 서로의
대체재로 쓰이고 있었다면 이렇게 갈라지지 않습니다.

재현:

```bash
NUREC_ABLATE_INPUTS=images bash scripts/nurec/score_epdms.sh <ckpt>
NUREC_ABLATE_INPUTS=route  bash scripts/nurec/score_epdms.sh <ckpt>
```

로그에 `[drivesuprim] ABLATION ACTIVE`가 찍히는지 확인한 뒤에만 숫자를
채택하십시오. 환경변수가 서브프로세스에 전달되지 않으면 ablation 없이 돌고도
ablation 결과처럼 보고될 수 있습니다.

**데이터 규모**

```
클립      전체 1,603   학습 1,552   검증 458   제외 51
프레임    전체 65,671  학습 63,581
학습 토큰 46,509       검증 토큰 13,729
```

63,581 프레임 중 46,509개만 샘플이 됩니다. `num_history_frames=4`,
`num_future_frames=8`이라 각 클립의 앞 3 · 뒤 8 프레임은 중심이 될 수
없습니다 (1,552 × 11 = 17,072).

샘플 하나가 `bev_seq_len=3` 프레임 × 카메라 3대 = **이미지 9장**을 읽습니다.

---

## 8. 재학습 (선택)

```bash
source setup_nurec.sh
NUM_GPUS=4 BATCH_SIZE=8 NUM_WORKERS=8 LOG_EVERY=20 \
  bash scripts/nurec/train_staged.sh planonly
```

* 30에폭, 4×H200에서 약 22시간 (1에폭 44분)
* 실제 측정: batch 8 × 4 GPU에서 **2.67 s/step**
* 진행 상황은 `exp/train_logs/<이름>.log`에 tee됩니다. 진행 바와 랭크
  트레이스백은 hydra 로그에 안 남으므로 이 파일을 보십시오.
* 중단은 `bash scripts/nurec/stop_training.sh` — torchrun은 NCCL 커널 안에
  갇힌 랭크를 못 죽이고 나가버려서, 남은 랭크가 GPU를 잡은 채 100% CPU로
  돕니다. nvidia-smi로는 정상 학습과 구별되지 않습니다.

**학습 잡을 두 개 동시에 돌리지 마십시오.** GPU 사용률이 평균 46%로 여유가
있어 보여도, 실측 결과 **6.5배 느려졌습니다** (2.61 s/step → 17.08 s/step).
유휴 구간이 짧게 흩어져 있어 두 번째 잡이 채울 수 있는 종류가 아닙니다.

`train_staged.sh`는 `stage1`(인지 사전학습) / `stage2`(플래너, 트렁크 동결) /
`stage3`(엔드투엔드) 커리큘럼도 지원합니다. plan-only는 그 커리큘럼과
독립적인, 플래너만 처음부터 학습하는 베이스라인입니다.

---

## 9. 함정 목록

| 함정 | 증상 | 대응 |
|---|---|---|
| 캘리브레이션 불일치 | 아무 증상 없음. 로스 정상 하강, 점수 그럴듯 | 커버리지 한 줄 확인, `visualize_bev_features.py` |
| 컨트롤러 LQR | GT가 92%대로 낮음 | `NUREC_USE_MPC=1` |
| `use_route` 불일치 | 조용히 다른 입력 | 학습·채점 동일하게 |
| `pdms` 컬럼 인용 | 점수가 부풀려짐 | `score` 컬럼을 읽을 것 |
| val이 train과 겹침 | 점수가 낙관적 | 홀드아웃 로그 목록 지정 |
| fp16 | 재현 가능한 NaN | `bf16-mixed` |
| 카메라 5대 추가 | 캘리브레이션 불일치 | 3대 고정 |
| 학습 병렬 실행 | 6.5배 감속 | 순차 실행 |
| Ctrl-C 후 고아 랭크 | GPU 점유, 100% CPU, 정상처럼 보임 | `stop_training.sh` |
| `pgrep -f` 자기 매치 | 프로세스 상태 오판 | `/proc/<pid>/comm` 확인 |

---

## 10. 동봉된 것

```
navsim/                     모델·피쳐빌더·컨트롤러·hydra 설정 전체
assets/nurec/vocab/         궤적 vocabulary (4096×40×3) — 추론에 필수
scripts/nurec/              채점·비디오·진단 스크립트
tests/                      캘리브레이션·기하·컨트롤러 테스트
docs/nurec/                 상세 문서
setup_nurec.sh              환경변수
requirements.txt, setup.py
```

**동봉되지 않은 것:** 체크포인트(별도 전달), 데이터셋(로그·이미지·메트릭
캐시·PDM 라벨), nuPlan 맵.

핵심 파일 위치:

| 무엇 | 어디 |
|---|---|
| 에이전트 설정 | `navsim/planning/script/config/common/agent/drivesuprim_agent_bevformer_vov_v2_vits_planonly_scratch_nurec.yaml` |
| 모델 | `navsim/agents/drivesuprim/drivesuprim_model.py` |
| 피쳐빌더 + 캘리브 가드 | `navsim/agents/drivesuprim/drivesuprim_features.py` |
| BEV 인코더 | `navsim/agents/backbones/bevformer/encoder.py` |
| 정류 목표 | `navsim/planning/data/nurec_rectify/target.py` |
| MPC | `navsim/planning/simulation/planner/nurec_controller/` |
| 채점 진입점 | `navsim/planning/script/run_pdm_score_one_stage_gpu_ssl.py` |

추가 문서는 `docs/nurec/` 아래에 있습니다 — `SETUP.md`(데이터 준비),
`IMAGE_GEOMETRY.md`(정류 기하), `PAIRING.md`(로그-이미지 짝),
`EPDMS.md`(점수 정의), `RUNBOOK.md`(운영), `CHECKPOINTS.md`.
