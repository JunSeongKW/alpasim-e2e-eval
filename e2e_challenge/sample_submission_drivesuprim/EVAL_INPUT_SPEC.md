# 평가 환경 입력 규격 — NuRec plan-only

이 서버 밖에서 이 모델을 돌리려면 평가 환경이 아래를 정확히 맞춰 공급해야
합니다. 값 하나하나에 근거 파일을 달았으니, 의심스러우면 그 파일을 보십시오.

**틀리면 조용히 틀립니다.** 이 프로젝트에서 인트린식 하나가 어긋나 30에폭짜리
학습을 통째로 버렸고, 그 동안 로스도 점수도 정상으로 보였습니다. 4장을 특히
주의해서 읽으십시오.

---

## 1. 좌표계

모든 기하량의 기준은 **차량 뒷축(rear axle)** 입니다. NAVSIM이 rig 프레임이라
부르는 것과 같습니다.

```
x  전방 (+)
y  좌측 (+)
z  상방 (+)
```

* 궤적, 라우트, 카메라 외부 파라미터가 전부 이 프레임입니다.
* 3D 박스 중심은 뒷축보다 **1.47 m 앞**입니다. 박스 중심 기준 궤적을 그대로
  넣으면 GT가 조용히 나빠집니다.
* 출력 궤적도 같은 프레임으로 돌려받습니다.

---

## 2. 카메라 — 3대만

`bev_num_cameras: 3` 이며, 순서까지 고정입니다.

| 인덱스 | 이름 | 소스 디렉토리 |
|---|---|---|
| 0 | `CAM_L0` | `camera_cross_left_120fov` |
| 1 | `CAM_F0` | `camera_front_wide_120fov` |
| 2 | `CAM_R0` | `camera_cross_right_120fov` |

근거: `navsim/agents/drivesuprim/drivesuprim_features.py`의 `_bev_cam_names`,
`navsim/planning/data/nurec_attach_images.py:26-28`.

`Cameras` 데이터클래스에는 8대 슬롯(`cam_f0/l0/l1/l2/r0/r1/r2/b0`)이 있지만
**이 모델은 위 3대만 읽습니다.** 나머지 5대를 채워도 무시되고, 채우더라도
정류되지 않은 파라미터를 넣으면 안 됩니다.

### 이미지 스펙

```
해상도     512 x 256   (width x height)
채널       RGB 3채널
포맷       PNG (JPEG 도 가능하나 준비 파이프라인은 PNG 를 씁니다)
왜곡       없음 — 이미 정류된 핀홀 영상이어야 합니다
```

학습 로더는 `PIL.Image.open -> np.array -> torchvision.transforms.ToTensor()`를
사용합니다. OpenCV 기반 진단/영상 스크립트에서만 화면 출력 시 BGR 규약을 따릅니다.

경로는 로그의 `cams[<CAM>]["data_path"]`가 센서 루트 기준 상대경로입니다:

```
<clip-uuid>/camera_front_wide_120fov/<timestamp_us>.png
```

---

## 3. 카메라 내부 파라미터 (인트린식)

**모든 카메라가 동일한 값을 가져야 합니다.** 정류 목표가 하나이기 때문입니다.

```
cam_intrinsic = [[377.0,   0.0, 256.0],
                 [  0.0, 377.0, 128.0],
                 [  0.0,   0.0,   1.0]]

distortion    = [0, 0, 0, 0, 0]
```

근거: `navsim/planning/data/nurec_rectify/target.py` (`TARGET_K`,
`TARGET_DISTORTION`).

특징과 이유:

* **등방** — `fx == fy == 377`. 512×256은 2:1인데 원본 1920×1080은 1.78:1이라,
  단순 리사이즈하면 세로로 12.5% 눌립니다. 정류하면 fx·fy가 자유로워져 눌림이
  0이 됩니다.
* **주점이 정중앙** — `cx = 512/2`, `cy = 256/2`.
* **fx=377의 근거** — 크로스 카메라가 전방에서 약 67도 벌어져 있습니다(1,607
  클립 실측: p50 67.0, p99 68.4, max 70.2). fx=377이면 화각 68.4도라 카메라
  사이 사각지대가 99% 클립에서 닫힙니다. fx=412였을 때는 63.7도라 3.4도가
  비었고, 40 m에서 2.9 m 폭이 아무 카메라에도 안 잡혔습니다.
* 화각: **수평 68.4도, 수직 37.5도**.

---

## 4. 반드시 지켜야 할 계약 — 인트린식과 이미지의 짝

> **`cam_intrinsic`은 그 로그가 가리키는 이미지 파일의 것이어야 합니다.**

이걸 어기면 아무 예외도 나지 않고, 로스도 정상으로 내려가고, 점수도
그럴듯하게 나옵니다.

이 프로젝트에서 실제로 벌어진 일:

* 로그에는 nuPlan 핀홀(`fx=fy=1545, cx=960, cy=560`, 1920×1080)이 들어 있었고
* 이미지 파일은 정류된 512×256이었으며
* 주점 `cx=960`이 512픽셀 폭 밖이라
* **BEV 참조점의 0.89%만 이미지 안에 떨어졌고**
* `SpatialCrossAttention`은 마스크가 전부 False면 residual만 돌려주므로
* **BEV 피쳐에 이미지 정보가 문자 그대로 0**이 되었습니다.

30에폭을 학습하고 0.9259라는 그럴듯한 점수를 받은 뒤에야 발견했습니다.
검증 방법은 단순했습니다 — 그 체크포인트에 실제 이미지를 넣은 결과와 **순수
검은 화면**을 넣은 결과가 비트 단위로 동일했습니다.

### 방어 장치

`drivesuprim_features.py`의 `_assert_intrinsics_match_image()`가 매 샘플마다
주점이 이미지 범위 안에 있는지 확인하고, 아니면 `CalibrationMismatch`를
던집니다. 이 예외는 데이터로더 재시도 래퍼에서 **재시도하지 않고** 그대로
올라옵니다.

**가드만 믿지 마십시오.** 주점이 이미지 안에 있으면서 초점거리만 틀린 경우는
통과합니다.

### 확인 방법

실행 로그에서 이 한 줄을 보십시오:

```
[bevformer] BEV cells reached by any camera: 98.438%  (cell-camera-anchor mean 32.812%)
```

* **앞 숫자**가 격자 커버리지입니다. 이 장비 구성에서 **98% 근처**가 정상이고,
  못 미치는 셀은 차 앞 5 m 이내뿐입니다(수직 화각 37.5도에서 나오는 물리적
  사각지대).
* **뒤 숫자**는 (셀 × 카메라 × z앵커) 삼중항 비율이라 구조적으로 낮습니다.
  카메라 3대 중 보통 1대가 한 셀을 보므로 약 ⅓입니다. **커버리지로 읽지
  마십시오.**
* **둘 중 어느 쪽이든 0에 가까우면 인트린식과 이미지가 짝이 아닙니다.**

거리별 정상값(참고):

```
 0- 5 m :  79.6%      ← 유일한 구멍
 5-10 m :  98.6%
10-20 m :  99.5%
20-56 m : 100.0%
```

---

## 5. 카메라 외부 파라미터 (익스트린식)

```
sensor2lidar_rotation     3x3   카메라 광축을 rig 프레임으로 표현한 것
sensor2lidar_translation  3     rig 프레임에서의 카메라 원점
```

이름은 `lidar`지만 실제 기준은 **rig(뒷축)** 입니다. NAVSIM의 `sensor2lidar`
규약이고, 챌린지 드라이버가 `lidar2img`에 넣는 것과 같습니다.
근거: `navsim/planning/data/nurec_rectify/calibration.py:10-25, 58-70`.

투영은 아래로 만들어집니다:

```
lidar2img = K_4x4 @ inv(cam2rig)
```

실측 예시 (translation, 단위 m):

```
CAM_F0   [ 2.062, -0.072, 1.597]
CAM_L0   [ 2.590,  0.953, 0.978]
CAM_R0   [ 2.582, -0.975, 0.970]
```

---

## 6. 프레임과 시퀀스

```
프레임 간격    500,000 us  (2 Hz)
BEV 입력       현재 프레임 포함 3장  (bev_seq_len = 3)
필요 히스토리  현재 프레임 기준 과거 3장 이상
```

BEV 백본은 `t-2, t-1, t-0` 세 프레임을 자차 운동으로 warp해 융합합니다.
프레임당 이미지는 **3장(카메라) × 3(프레임) = 9장**입니다.

한 샘플이 되려면 `num_history_frames = 4`를 만족해야 하므로, 클립의 **앞 3
프레임은 샘플이 될 수 없습니다.** (채점만 한다면 미래 프레임은 필요 없지만,
학습 라벨을 만들려면 `num_future_frames = 8`도 필요합니다.)

---

## 7. 라우트

근거: `navsim/common/route_contract.py`, `dataclasses.py:143-167`,
`drivesuprim_features.py:_get_route_feature`.

```
키       route_waypoints
형태     (20, 3)  float32
프레임   rig(뒷축) — x 전방, y 좌측, z 는 0
패딩     NaN. 유효한 슬롯만 앞쪽에 채우고 나머지는 NaN
없을 때  전부 NaN 이면 "이 프레임에 라우트 없음" 으로 처리됩니다
```

슬롯 규약:

```
ROUTE_SLOTS          20
ROUTE_HORIZON_M      80.0        슬롯을 이 거리에 균등 배치
ROUTE_SPACING_M      80/19 ≈ 4.21 m   (호길이 기준)
ROUTE_NEAR_CUTOFF_M  40.0        이보다 가까운 슬롯은 버리고 앞으로 당깁니다
MIN_WAYPOINT_GAP_M   3.5         재샘플 간격 하한
MAX_WAYPOINT_GAP_M   4.5         상한. 벗어나면 그 뒤 전부 버립니다
```

근거리 컷오프 때문에 실제로 유효한 슬롯은 프레임당 **보통 4~10개**입니다.
20개가 다 차 있는 프레임은 정상이 아닙니다.

모델은 내부에서 `route_waypoints[:, :2] / 80.0`으로 정규화하고, 유효
슬롯 마스크(`route_mask`)를 함께 씁니다. **평가 환경은 정규화하지 말고
미터 단위 원값을 주십시오.**

`use_route` 설정을 **학습과 일치**시켜야 합니다. 이 모델은 **`use_route=true`**
로 학습되었습니다. `false`로 돌리면 `_status_encoding`의 앞 네 컬럼에 라우트
대신 `driving_command`가 들어가 조용히 다른 입력을 먹습니다.

---

## 8. 자차 상태

```
ego2global_translation   3      전역 위치
ego2global_rotation      4      쿼터니언 (w, x, y, z)
ego_dynamic_state        4      [vx, vy, ax, ay]  — rig 프레임
driving_command          4      one-hot
```

`driving_command` 인코딩 (`navsim/common/driving_command.py:60-63`):

```
LEFT     [1, 0, 0, 0]
STRAIGHT [0, 1, 0, 0]
RIGHT    [0, 0, 1, 0]
UNKNOWN  [0, 0, 0, 1]
```

이 모델은 `use_route=true`라 **driving_command 를 쓰지 않습니다.** 다만
로그 스키마상 필수이고, DriveSuprim 계열 베이스라인(V2-99 등)은 이걸
읽습니다.

---

## 9. 로그 프레임 스키마

`navsim_logs/trainval/<log_name>.pkl`은 프레임 딕셔너리의 리스트입니다.
타임스탬프는 **엄격히 증가**해야 합니다.
근거: `navsim/planning/data/nurec_converter.py`.

```
token                     str    프레임 고유 ID
timestamp                 int    마이크로초
log_name / scene_token    str
map_location              str
ego2global_translation    [3]
ego2global_rotation       [4]    쿼터니언
ego_dynamic_state         [4]    vx, vy, ax, ay
driving_command           [4]    one-hot
route_waypoints           (20,3) 위 7장
roadblock_ids             [str]
cams                      dict   CAM_* -> 아래
anns                      dict   학습·채점 라벨용 (추론만 하면 비워도 됩니다)
```

카메라 항목:

```
cams["CAM_F0"] = {
    "data_path":                "<clip>/camera_front_wide_120fov/<ts>.png",
    "sensor2lidar_rotation":    3x3,
    "sensor2lidar_translation": [3],
    "cam_intrinsic":            3x3,      # 4장의 TARGET_K
    "distortion":               [0]*5,
}
```

`anns`는 `gt_boxes`, `gt_names`, `gt_velocity_3d`, `instance_tokens`,
`track_tokens`를 갖습니다. **plan-only 추론에는 필요 없습니다** — 채점과
stage 1 인지 학습에만 쓰입니다.

---

## 10. 출력 규격

모델이 돌려주는 것:

```
궤적        (40, 3)   [x, y, heading]
시간 간격   0.1 s
지평선      4 s
프레임      rig(뒷축), 현재 프레임 기준 상대
```

4,096개 vocabulary 중 하나를 고른 뒤 리파인먼트를 거친 결과입니다.
Vocabulary 파일(`nurec_train_kmeans_4096x40x3.npy`, shape `(4096, 40, 3)`)이
추론에 **필수**입니다.

컨트롤러는 평가 목적에 따라 구분해야 합니다.

- 학습 라벨/오프라인 NuRec EPDMS를 재현할 때는 **linear MPC
  (`NuRecSimulator`)** 를 씁니다. NAVSIM hydra 기본 LQR로 바꾸면 안 됩니다.
- 공식 AlpaSim closed-loop challenge를 재현할 때는 참가자가 컨트롤러를 선택할 수
  없으므로 대회 서비스의 **`nonlinear` controller**를 그대로 씁니다. 현재 challenge
  호환 스크립트도 이 설정입니다.

---

## 11. 반입 전 자가 점검

데이터를 준비했으면 모델을 믿기 전에 이 순서로 확인하십시오.

**1) 코드가 온전한가** (데이터 불필요)

```bash
python scripts/nurec/verify_bundle.py
```

**2) 인트린식이 이미지와 짝인가**

실행 로그의 `BEV cells reached by any camera`가 **95% 이상**인지. 0에 가까우면
4장으로 돌아가십시오.

**3) 카메라·프레임이 실제로 융합되는가**

```bash
python scripts/nurec/check_bev_fusion.py 4
```

정상이면 세 카메라의 민감도 로브가 좌·중앙·우에 **좌우 대칭**으로 나타납니다:

```
CAM_L0  0.1962      CAM_F0  0.9023      CAM_R0  0.1602
frame t-2  0.2450   t-1  0.2408   t-0(현재)  0.7177
```

어느 하나가 0이면 그 입력은 쓰이지 않고 있습니다.

**4) BEV 에 이미지가 실제로 들어가는가**

```bash
python scripts/nurec/visualize_bev_features.py 6
```

`camera sensitivity`가 **1.0 근처**여야 합니다. 캘리브레이션이 깨진 모델은
정확히 **0.00**이었고, 정상 모델은 **1.15**입니다.

**5) 회귀 테스트**

```bash
pytest tests/test_intrinsics_match_image.py \
       tests/test_nurec_rectified_geometry.py \
       tests/test_rear_axle_reference.py -q
```

---

## 12. 요약 표

| 항목 | 값 |
|---|---|
| 카메라 | CAM_L0, CAM_F0, CAM_R0 (3대 고정) |
| 해상도 | 512 × 256, BGR, 정류 완료 |
| fx, fy | 377.0, 377.0 |
| cx, cy | 256.0, 128.0 |
| distortion | `[0,0,0,0,0]` |
| 화각 | 수평 68.4°, 수직 37.5° |
| 외부 파라미터 | `sensor2lidar_*`, rig(뒷축) 기준 |
| 프레임 간격 | 500,000 µs (2 Hz) |
| BEV 입력 프레임 | 3 (현재 포함) |
| 샘플당 이미지 | 9장 |
| 라우트 | `(20, 3)` float32, rig, NaN 패딩, 미터 단위 |
| 라우트 지평선 | 80 m, 슬롯 간격 4.21 m, 근거리 컷오프 40 m |
| `use_route` | **true** (학습과 반드시 일치) |
| 출력 궤적 | `(40, 3)`, 0.1 s 간격, 4 s, rig |
| Vocabulary | `(4096, 40, 3)` |
| 컨트롤러 | offline EPDMS: `NuRecSimulator`; 공식 closed-loop: AlpaSim `nonlinear` |
    | BEV 격자 | 56 × 56, 1 m/셀, 전방 0–56 m, 횡 ±28 m |
