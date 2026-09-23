# MPC / 랭킹 파라미터 직접 실험하는 법

작업 디렉터리: `alpasim/e2e_challenge/sample_submission_drivesuprim/`

재빌드 없습니다. 값만 바꾸고 실행하면 됩니다.

---

## 1. MPC 게인 실험

### 1-1. 설정 추가

`mpc_gain_variants.txt` 에 한 줄 추가합니다. 형식은 `이름|하이드라 오버라이드` 입니다.

```
myTest|controller.gains.idx_start_penalty=1 controller.gains.heading_weight=6.0
```

바꿀 수 있는 키 (기본값):

```
controller.n_horizon                            20      예측 스텝 수
controller.dt_mpc                               0.1     스텝 간격 = 재계산 주기(초)
controller.gains.long_position_weight           2.0     종방향 위치오차
controller.gains.lat_position_weight            1.0     횡방향 위치오차
controller.gains.heading_weight                 1.0     헤딩 오차
controller.gains.acceleration_weight            0.1     가속도 상태 크기
controller.gains.rel_front_steering_angle_weight 5.0    조향 명령 변화율
controller.gains.rel_acceleration_weight        1.0     가속 명령 변화율
controller.gains.idx_start_penalty              10      앞 N스텝 추종벌점 끄기
```

### 1-2. 실행

```bash
./run_mpc_gain_sweep.sh myTest
```

여러 개를 동시에 돌리려면 이름을 나열합니다. **3개까지만** 하십시오 — 렌더러가 설정당 3개씩 뜨고 각자 harmonizer 모델을 받으므로 그 이상은 기동 타임아웃에 걸립니다.

```bash
./run_mpc_gain_sweep.sh myTest myTest2 myTest3
```

이미 다른 실험이 돌고 있으면 포트가 겹치므로 슬롯을 옮깁니다.

```bash
SLOT_OFFSET=3 ./run_mpc_gain_sweep.sh myTest
```

### 1-3. 옵션

```bash
CLIPS_FILE=./flatvits_10clips_seed20260903.txt \   # 클립 목록 (기본 rank3clips.txt = 3개)
OUT_ROOT=/path/to/output \                         # 저장 위치
RENDER_VIDEO=false \                               # 영상 끄면 빨라짐
  ./run_mpc_gain_sweep.sh myTest
```

---

## 2. 랭킹 가중치 실험

### 2-1. 설정 추가

`assets/drivesuprim/flatvits_nodistill_ep19_eval/rank_variants/` 에 JSON 을 만듭니다.
기존 것을 복사해 고치는 편이 쉽습니다.

```bash
cd assets/drivesuprim/flatvits_nodistill_ep19_eval/rank_variants
python3 -c "
import json
c = json.load(open('base.json'))
c['pdm_rank_product_exponents'] = {'drivable_area_compliance': 5.0}
c['pdm_imi_rank_weight'] = 0.5
json.dump(c, open('myRank.json','w'), indent=1)
"
```

바꿀 수 있는 키:

```
pdm_rank_product_exponents   {}     항별 지수. <1 이면 약화, >1 이면 강화, 0 이면 제거
                                    항 이름: no_at_fault_collisions
                                             drivable_area_compliance
                                             ego_progress
pdm_imi_rank_weight          1.0    모방 항의 가중치
pdm_rank_product_terms       [3개]  곱에 넣을 항 목록
pdm_aggregate_rank_weight    0.0    pdm_score 집계 헤드 보너스
```

지수의 의미 — 좋은 후보 0.98 과 나쁜 후보 0.70 의 격차:

```
지수 0.0   1.00배   항 무력화
지수 0.3   1.11배   약화
지수 1.0   1.40배   현행
지수 3.0   2.74배   강화
```

### 2-2. 실행

`변형이름:드라이버포트:GPU:wizard베이스포트:컨테이너접두사` 형식입니다.

```bash
./run_rank_weight_sweep.sh "myRank:6900,6901,6902:0,1,2:6000:rk-my"
```

MPC 게인은 실험 1 우승값이 자동으로 깔립니다. 바꾸려면:

```bash
MPC_OVERRIDES="controller.gains.idx_start_penalty=1" \
  ./run_rank_weight_sweep.sh "myRank:6900,6901,6902:0,1,2:6000:rk-my"
```

---

## 3. 결과 보기

```bash
SP=/tmp/claude-1000/-home-kaist5/<세션id>/scratchpad
python3 $SP/mpc_compare.py <OUT_ROOT> base myTest myTest2
```

출력 지표:

| 열 | 뜻 |
|---|---|
| `score` | 챌린지 점수 (게이트 하나라도 위반하면 0) |
| `GT횡오차` | GT 궤적과의 횡방향 거리 — **목적 지표** |
| `주행m` / `완주율` | 얼마나 갔는지 |
| `조향율p95` | 조향 명령 변화율. 급조향 지표 |
| `횡가속p95` | 실제 횡가속도. 8 m/s² 넘으면 비물리적 |
| `저크p95` | 가속 명령 변화율 |

제어 지표는 **실패 시점 이전만** 집계합니다. 도로를 벗어난 뒤의 잔해가 원인처럼 보이는 것을 막기 위해서입니다.

---

## 4. 확인 사항

실행 후 값이 실제로 들어갔는지 두 곳에서 확인합니다.

```bash
cat <OUT_ROOT>/<변형>/controller-config.yaml            # MPC 게인 실측
grep 'RANK WEIGHTS' <OUT_ROOT>/_logs/<변형>/driver.log  # 랭킹 가중치 실측
```

---

## 5. 주의

- **다른 실험이 도는 중에는 포트가 겹칩니다.** `docker ps` 로 확인하고 `SLOT_OFFSET` 을 쓰십시오.
- **GPU 여유 확인.** 설정 하나가 드라이버 3 + 렌더러 3 + physics 3 을 씁니다.
- 중단은 컨테이너 제거로 합니다. `pkill -f` 로 패턴을 잡으면 자기 셸까지 죽는 일이 있습니다.

```bash
docker rm -f $(docker ps -aq --filter 'name=mg-')   # MPC 스윕
docker rm -f $(docker ps -aq --filter 'name=rk-')   # 랭킹 스윕
```

- `src/wizard/configs/controller/default.yaml` 은 **건드리지 않는 편이 낫습니다.** 오버라이드로만 실험하면 기준선이 항상 재현 가능합니다. 확정된 값만 나중에 반영하십시오.

---

## 6. 지금까지 확인된 것

```
idx_start_penalty  10 → 3      통과 0/3 → 2/3      가장 큰 효과
long/lat 가중치    2.0/1.0 → 0.5/6.0               효과 큼
drivable_area 지수 1.0 → 3.0   통과 2/3 → 3/3      커브 오프로드 해결
n_horizon          20 → 40                          역효과
ego_progress 지수  낮추면                           고속 클립에서 손해
imi 가중치         1.0 → 0.3                        효과 없음
```

미시험: `idx_start_penalty` 0~2, `heading_weight` 단독, `acceleration_weight` 단독, `dt_mpc`.
