# NuRec Full-4096 EPDMS 생성 파이프라인 인수인계

업데이트 시각: **2026-08-24 18:45 KST**

이 문서는 AXEv1.0에서 NuRec 데이터의 trajectory vocabulary EPDMS를 생성하기 위해 확인한 데이터, 결정사항, 코드 변경, 검증 결과, 현재 실행 상태와 재개 방법을 정리한 인수인계 문서다.

## 1. 최종 목표와 현재 범위

최종 산출물은 각 NAVSIM scene token에 대해 **Full NuRec 4096 vocabulary trajectory의 EPDMS 점수만** 저장한 pickle이다.

```text
NuRec USDZ + converted ego/object logs
  -> ClipGT vector map 및 route 추출
  -> NAVSIM-compatible logs
  -> metric cache
  -> Full NuRec 4096 trajectories 시뮬레이션
  -> token -> float16[4096] EPDMS
```

최종 artifact 계약은 다음과 같다.

```python
{
    "<scene_token>": np.ndarray(shape=(4096,), dtype=np.float16),
    ...
}
```

각 값은 `[0, 1]` 범위다. 충돌, drivable area, 진행 방향, progress, TTC, lane keeping 및 history comfort는 내부 계산에 필요하지만 최종 EPDMS-only artifact에는 별도 배열로 저장하지 않는다.

주의: 현재 DriveSuprim의 기존 per-metric BCE 학습 코드는 component dictionary를 기대한다. 이 문서의 EPDMS-only artifact를 학습 입력으로 직접 사용할 경우에는 scalar EPDMS target/loss 연결을 별도로 추가해야 한다. 현재 작업 범위는 올바른 EPDMS 생성 및 전달까지다.

## 2. 확정된 결정사항

- 메인 vocabulary: **필터하지 않은 Full NuRec 4096**
- 사용하지 않는 vocabulary:
  - Full NuRec 8192
  - DriveSuprim-range filtered 4096/8192
- vocabulary 파일 shape: `(4096, 40, 3)`, `float32`
- NuRec 저장 frame 간격: **0.5초**
- Scene window: history 4 + future 8 = 총 12 frames
- scene window stride: `frame_interval=1`
  - 다음 sample을 저장 frame 한 칸, 즉 0.5초씩 이동
  - `frame_interval=12`는 6초마다 한 sample만 남기므로 사용하지 않음
- 1,607 logs 중 route를 신뢰할 수 있는 **1,603개**를 train에 포함 (5.5 참조)
- validation은 train에서 seed 2026으로 10% random sampling하며 train과 겹침
- camera는 `CAM_L0`, `CAM_F0`, `CAM_R0` 세 대
- EPDMS 계산 자체에는 이미지가 필요하지 않음
- NuRec에는 신뢰 가능한 timestamp별 신호등 phase가 없으므로 `traffic_light_compliance=1.0`을 곱셈 중립값으로 사용

## 3. 데이터 및 출력 경로

### 원본 및 변환 데이터

| 항목 | 경로 |
|---|---|
| NuRec USDZ 1,607 clips | `/mnt/nfs/data/processed_dataset/alphasim/NuRec/sample_set/26.04_release` |
| 기존 NAVSIM-style logs 1,607개 | `/rhome/satyam/drivesuprim_workspace/nurec_satyam_ws/satyam/nurec_navsim_all/navsim_logs/nurec` |
| 렌더 이미지 root | `/mnt/nfs/data/processed_dataset/alphasim/nurec2img/26.04_release_10hz_3cam_png` |
| Full NuRec 4096 vocabulary | `/rhome/satyam/drivesuprim_workspace/nurec_satyam_ws/satyam/nurec_vocab/nurec_train_kmeans_4096x40x3.npy` |

### 준비 및 검증 데이터

| 항목 | 경로 |
|---|---|
| 전체 준비 root | `/rhome/satyam/drivesuprim_workspace/nurec_satyam_ws/satyam/nurec_axe` |
| map bundles | `/rhome/satyam/drivesuprim_workspace/nurec_satyam_ws/satyam/nurec_axe/maps` |
| map-attached full logs | `/rhome/satyam/drivesuprim_workspace/nurec_satyam_ws/satyam/nurec_axe/navsim_logs/trainval` |
| full index | `/rhome/satyam/drivesuprim_workspace/nurec_satyam_ws/satyam/nurec_axe/index.json` |
| 표본 검증 root | `/rhome/satyam/drivesuprim_workspace/nurec_satyam_ws/satyam/nurec_validation` |
| 표본 metric cache | `/rhome/satyam/drivesuprim_workspace/nurec_satyam_ws/satyam/nurec_validation/metric_cache` |
| 표본 EPDMS artifact | `/rhome/satyam/drivesuprim_workspace/nurec_satyam_ws/satyam/nurec_validation/traj_pdm/ori/vocab_score_4096_nurec_full4096_validation/nurec_full4096_epdms.pkl` |
| GT 시각화 | `/rhome/satyam/drivesuprim_workspace/nurec_satyam_ws/satyam/nurec_validation/gt_alignment_4clips.png` |
| 전체 map 작업 로그 | `/rhome/satyam/drivesuprim_workspace/nurec_satyam_ws/satyam/nurec_validation/full_map_prepare.log` |
| attach/index/validate 재실행 로그 | `/rhome/satyam/drivesuprim_workspace/nurec_satyam_ws/satyam/nurec_validation/full_map_prepare_rerun.log` |

## 4. EPDMS 정의와 신호등 처리

현재 vocabulary scoring 경로의 EPDMS는 다음과 같이 집계한다.

```text
multiplicative =
    no_at_fault_collisions
  * drivable_area_compliance
  * driving_direction_compliance
  * traffic_light_compliance

weighted =
    (5 * ego_progress
   + 5 * time_to_collision_within_bound
   + 2 * lane_keeping
   + 2 * history_comfort) / 14

EPDMS = multiplicative * weighted
```

NuRec에서는 동적 신호 상태를 만들 수 없으므로 `traffic_light_compliance=1.0`이다. 곱셈에서 1은 중립값이라 다른 metric의 값과 순위를 바꾸지 않는다. 이 결과는 **NuRec EPDMS with neutralized traffic light**로 해석해야 하며 공식 NAVSIM EPDMS 절대값과 직접 비교하면 안 된다.

별도로 AXE/DriveSuprim 모델 쪽에는 다음 처리가 들어가 있다.

- NuRec 실행에서 traffic-light BCE loss weight 0
- trajectory ranking에서 traffic-light prediction contribution 0
- checkpoint 호환성을 위해 prediction head 자체는 유지

## 5. 검증 과정과 결과

### 5.1 Map 및 route 표본 검증

전체 실행 전 116개 USDZ를 다양하게 추출해 검증했다.

- map pickle: 116/116 load 성공
- ego-to-lane matching: 전 표본 100%
- lane 수: 최소 44, 중앙값 215.5, 최대 724
- roadblock 수: 최소 23, 중앙값 96.5, 최대 283
- connector 수: 최소 0, 중앙값 3, 최대 25
- route 길이: 최소 1, 중앙값 9, 최대 26 roadblocks
- map-attached logs: 116
- frames: 4,749
- route가 있는 frames: 4,749/4,749
- frame 간격: 정확히 500,000 us, 오차 0
- map ego timestamp 최대 매칭 오차: 48,223 us
- required cameras metadata: `CAM_L0`, `CAM_F0`, `CAM_R0`

GT 시각화에서는 ego history/future, route, lane map 및 object boxes가 같은 좌표계에 정렬됨을 확인했다.

### 5.2 Scene stride 검증

현재 1,607 source logs의 12-frame window 수를 직접 비교했다.

| 설정 | 후보 windows |
|---|---:|
| `frame_interval=12` | 4,821 |
| `frame_interval=1` | 48,158 |

이전 vocabulary 생성 보고서의 usable windows는 44,944개였으며 해당 생성기의 추가 filtering 결과다.

route 커버리지 검사로 4개 log를 제외한 뒤(5.5) 최종 후보 window 수는 **48,038**이다. stride 1에서 log당 window 수는 `n_frames - 11`이므로 `65,671 - 11 x 1,603 = 48,038`이며, 같은 식이 제외 전 값 `65,835 - 11 x 1,607 = 48,158`을 정확히 재현한다. 최종 metric cache token 수는 full run의 metadata를 기준으로 다시 기록해야 한다.

### 5.3 Metric cache 표본 검증

lane 규모, connector 수, object 수가 서로 다른 8 clips에서 중앙 token을 선택했다.

- metric cache 8/8 생성 성공
- objects per selected token: 7–72
- route suffix 길이: 2–16
- exception/NaN 없음

### 5.4 Full NuRec 4096 EPDMS 표본 검증

최종 메인 vocabulary 파일로 8개 token × 4,096 trajectories를 계산했다.

| Token | Mean | Median | Max | Zero 비율 |
|---|---:|---:|---:|---:|
| `455c77a5e6869534` | 0.1647 | 0.0000 | 0.8770 | 67.75% |
| `5e211b6d4c21ecb5` | 0.6704 | 0.8569 | 1.0000 | 27.76% |
| `6bae6852a9353786` | 0.2032 | 0.0000 | 0.9326 | 66.46% |
| `9eb45ff63aeb1cca` | 0.2167 | 0.0000 | 1.0000 | 73.71% |
| `c40c285d37c178b4` | 0.2168 | 0.0000 | 0.9414 | 72.53% |
| `ce454cfaa74fb78c` | 0.5024 | 0.5000 | 1.0000 | 36.23% |
| `f449799108b65e73` | 0.1454 | 0.0000 | 1.0000 | 78.93% |
| `fd9b3fd1c265557a` | 0.2800 | 0.3572 | 1.0000 | 46.56% |
| **전체 32,768 scores** | **0.2999** | **0.0000** | **1.0000** | **58.74%** |

추가 확인 결과:

- shape: token마다 `(4096,)`
- dtype: `float16`
- finite: 32,768/32,768
- 전체 범위: `[0.0, 1.0]`
- 전체 고유 점수: 1,400종
- 환경별 평균 및 zero 비율이 달라 score collapse 없음

### 5.5 전체 1,607 clips route 커버리지

표본 116개에서는 ego-to-lane matching이 100%였으나, 전수 실행에서 예외가 드러났다.

| 상태 | clips |
|---|---:|
| ego가 lane에 100% 매칭 | 1,596 |
| 부분 매칭 | 8 |
| 매칭 0건, route 전무 | 3 |

route가 전혀 없는 3개는 lane 자체는 추출됐으나 ego 궤적이 그 위에 하나도 얹히지 않는다.

```text
0dc4e9cc-833e-4ddd-bfbb-4899f6a46b78   lanes 67   ego 202개 중 0개 매칭
1661c375-c900-4797-8c31-01f86a17d425   lanes 257  0개 매칭
e69eafaf-c315-4376-bab9-9be80ced6fe1   lanes 287  0개 매칭
```

부분 매칭 8개 중 7개는 미매칭 구간이 clip의 맨 앞이나 맨 뒤에 있어 무해하다. 앞 gap은 route_index가 0이라 전체 route가 유지되고, 뒤 gap은 마지막 suffix가 유지된다. 나머지 하나는 다르다.

```text
b8850f3f-086c-4310-9eff-cc13ce6d940b   33/202 매칭, route 2 roadblocks
                                       idx 33부터 169개 연속 미매칭 (clip의 84%)
```

`_route_suffixes`는 미매칭 시 직전 suffix를 유지하므로 이 clip은 84% 구간에 낡은 route가 붙는다. 해당 token의 `ego_progress`와 `driving_direction_compliance`는 ego가 이미 벗어난 centerline 기준으로 계산된다.

네 clip 모두 제외했다. 최종 준비 결과는 다음과 같다.

- logs: **1,603**
- frames: **65,671**
- frames with route: **65,671 / 65,671** (100%)
- max ego timestamp 매칭 오차: 50,020 us (한계 250,000 us)
- index의 train 목록과 디스크 파일 일치, val은 train의 부분집합

## 6. 검증 중 발견하고 수정한 문제

### 6.1 잘못된 `frame_interval=12`

NuRec logs는 이미 0.5초 frame이므로 12는 시간 sampling 값이 아니라 window stride 12였다. 이 설정은 약 90%의 학습 window를 버린다. NuRec split 설정을 `frame_interval=1`로 수정했다.

### 6.2 Human penalty filter 이후 `pdm_score` 불일치

기존 scorer는 component metric을 집계한 뒤 human penalty filter를 적용했다. 따라서 filter가 component를 neutralize해도 이미 만들어진 `pdm_score`는 갱신되지 않았다. 실제 8192 최대부하 표본에서 component는 정상적으로 달라졌지만 저장된 `pdm_score`가 전부 1.0이 되는 현상을 발견했다.

수정 내용:

- human penalty filter 적용 후 component로 EPDMS 재집계
- traffic-light 1.0을 곱셈 중립값으로 포함
- 최종 EPDMS와 실제 component 상태를 일치시킴
- regression test 추가

수정 후 Full NuRec 4096 표본에서는 위 표와 같이 정상 분포가 확인됐다.

### 6.3 EPDMS-only artifact

기존 생성기는 token마다 모든 component 배열을 저장했다. 현재는 `+epdms_only=true`일 때 최종 `pdm_score` 배열만 저장한다. 중간 `tmp.pkl`에는 재개와 디버깅을 위해 component 결과가 남을 수 있으나 최종 전달 artifact에는 포함되지 않는다.

### 6.4 재개 가능한 처리

- map 추출은 이미 존재하는 non-empty bundle을 건너뜀
- vocabulary scoring은 token별 `tmp.pkl`을 재사용
- metric caching은 기본적으로 기존 cache를 재사용
- metric cache를 강제로 다시 만들 때만 `NUREC_FORCE_METRIC_CACHE=true` 사용

### 6.5 Vocabulary scoring worker가 실패 하나로 전체 실행을 폐기

`gen_vocab_full_score.py`의 worker는 token 하나가 예외를 내면 `return None`으로 종료했다. 이는 두 단계로 번진다.

1. 해당 worker가 담당한 나머지 token을 전부 포기한다. `worker.threads_per_node=32`에서 chunk는 정확히 32개이므로 worker 하나가 약 1,500 token을 들고 있다.
2. `worker_map`의 마지막 flatten `[result for results in scattered_objects for result in results]`이 `None`을 만나 `TypeError: 'NoneType' object is not iterable`을 낸다.

flatten은 32개 chunk가 **모두 끝난 뒤** 실행되므로 크래시는 36~48시간을 다 돌고 최종 `pickle.dump` 직전에 발생한다. 실패가 결정론적이면 token별 cache로 재개해도 같은 지점에서 반복해서 죽는다.

수정 내용:

- 실패를 token별 row(`failed`, `reason`)로 기록하고 나머지 token 처리를 계속함
- metric cache가 없어 조용히 탈락하던 token도 `reason: "missing metric cache"`로 기록
- 결과 pkl 옆에 `<result_name>_report.json`을 남겨 요청/성공/실패/미집계 token 수와 목록을 제공
- 회귀 테스트 추가

### 6.6 원자적이지 않은 결과 저장

최종 artifact(약 395 MB)와 token별 `tmp.pkl` 모두 대상 경로에 직접 쓰고 있었다. NFS에서 중단되면 잘린 파일이 남는다. `tmp.pkl` 쪽이 특히 위험한데, 잘린 캐시가 남으면 `os.path.exists`가 계속 참이라 **이후 모든 재개가 같은 token에서 영구히 실패**한다. 양쪽 모두 임시 파일에 쓴 뒤 `os.replace`로 교체하도록 바꿨다.

### 6.7 Route 정렬 및 커버리지 검사 부재

`attach_map`은 최근접 ego timestamp를 찾아 route suffix를 붙이면서 최대 오차를 **보고만** 하고 임계값 검사를 하지 않았다. 표본 116개에서는 최대 48,223 us였으나 전수에서 한 clip이 크게 어긋나면 틀린 route가 조용히 붙는다. 또한 route가 아예 없거나 산발적으로만 매칭되는 clip을 걸러내지 못했다(5.5).

수정 내용:

- frame당 최근접 ego 오차가 250,000 us(반 프레임)를 넘으면 `RouteAlignmentError`
- route가 전혀 없는 log는 제외하고 `routeless_maps`에 기록
- ego-lane 매칭률이 50% 미만인 log는 제외하고 `unreliable_route_maps`에 기록
- 두 목록 합계가 전체 log의 1%를 넘으면 중단. 개별 clip 문제가 아니라 map 추출 자체가 깨진 신호이므로
- 제외된 log의 이전 출력 파일을 삭제해 재실행 시 index와 디스크가 어긋나지 않게 함
- 임계값은 `--max-timestamp-error-us`, `--min-ego-match-fraction`, `--max-routeless-fraction`으로 조정 가능

전수 실행 실측값은 최대 오차 50,020 us로 250,000 us 한계의 1/5이며, 매칭률은 제외 대상이 16.3%, 유지 대상 최저가 75.7%로 임계값 50% 양쪽에 여유가 크다.

## 7. 주요 코드 변경 위치

| 파일 | 역할 |
|---|---|
| `navsim/planning/data/nurec_map_data.py` | USDZ ClipGT parquet에서 compact map/route 추출 |
| `navsim/planning/data/nurec_map.py` | NuRec map bundle을 NAVSIM Map API로 노출 |
| `navsim/planning/data/nurec_attach_maps.py` | map location과 route roadblocks 부착, route 정렬/커버리지 검사 |
| `navsim/planning/data/nurec_validate.py` | frame/map/route/camera contract 검증 |
| `navsim/planning/script/config/common/train_test_split/nurec.yaml` | 4-history/8-future, stride 1 scene 설정 |
| `navsim/evaluate/pdm_score.py` | filter 이후 EPDMS 재집계 |
| `navsim/agents/tools/gen_vocab_full_score.py` | EPDMS-only 최종 artifact, token별 실패 격리, 원자적 저장, 실행 report |
| `scripts/nurec/prepare_usdz.sh` | 전체 map 준비 및 중단 후 재개 |
| `scripts/nurec/generate_epdms.sh` | metric cache + Full-4096 EPDMS 실행 |
| `scripts/nurec/visualize_gt.py` | map/route/ego/object GT 시각화 |
| `tests/test_epdms_aggregation.py` | EPDMS 집계 및 traffic-light neutral regression test |
| `tests/test_nurec_traffic_light.py` | traffic-light loss 제외 regression test |
| `tests/test_gen_vocab_score_rows.py` | 실패 token이 chunk를 폐기하지 않음, EPDMS-only 축약, 캐시 원자성 |
| `tests/test_nurec_route_alignment.py` | route 정렬 한계, route 부재, ego 매칭률 |
| `navsim/planning/simulation/planner/nurec_controller/` | NuRec Runtime linear MPC의 torch/GPU 포팅 (batched ADMM) |
| `navsim/planning/data/nurec_route.py` | PAI Track Route 20-slot 생성 |
| `navsim/common/route_contract.py` | route 상수 (slot 수, 간격, 40 m 절단, gap 허용범위) |
| `navsim/agents/drivesuprim/drivesuprim_model.py` | `RouteEncoder` / `RouteTokens`, decoder·refinement 연결 |
| `scripts/nurec/compare_controllers*.py`, `visualize_route.py` | LQR vs MPC 및 route 시각화 |

## 8. 현재 실행 상태

2026-08-25 05:19 KST에 full EPDMS 생성을 시작했다.

```text
pid   135228            (setsid 분리, 셸 종료와 무관)
log   $NAVSIM_TRAJPDM_ROOT/epdms_run.log
pid파일 $NAVSIM_TRAJPDM_ROOT/epdms_run.pid
```

시작 시점의 확정된 입력이다.

- clips **1,603**, frames **65,671**, frame 간격 **0.5 s**
- 채점 대상 token **48,038** = scene_loader ∩ metric cache, 양방향 누락 0
  (버려지는 17,633 = 1,603 × 11, clip마다 4-history/8-future 창을 못 채우는 앞뒤 프레임)
- vocabulary `(4096, 40, 3)`, 후보 하나는 40 pose × 0.1 s = 4 s
- 총 후보 평가 **48,038 × 4,096 = 196,763,648**
- 제어기는 navsim LQR이 아니라 **NuRec Runtime linear MPC** (`NUREC_USE_MPC=1`, 13장)
- 스모크에서 만든 per-token 359개를 재사용하고 시작

metric caching 단계는 캐시가 완비된 상태에서 **812.56 s**가 걸리며 48,038개를
재빌드 없이 확인만 한다.

## 9. 모니터링 및 재개 명령

모든 명령은 다음 repository에서 실행한다.

```bash
cd /rhome/satyam/drivesuprim_workspace/AXEv1.0-NuRec
```

### 현재 map 진행률 확인

```bash
find /rhome/satyam/drivesuprim_workspace/nurec_satyam_ws/satyam/nurec_axe/maps \
  -maxdepth 1 -type f -name '*.pkl' | wc -l

tail -f /rhome/satyam/drivesuprim_workspace/nurec_satyam_ws/satyam/nurec_validation/full_map_prepare.log
```

### map 작업이 중단된 경우 재개

```bash
export NUREC_USDZ_ROOT=/mnt/nfs/data/processed_dataset/alphasim/NuRec/sample_set/26.04_release
export NUREC_INPUT_LOG_ROOT=/rhome/satyam/drivesuprim_workspace/nurec_satyam_ws/satyam/nurec_navsim_all/navsim_logs/nurec
export NUREC_SENSOR_ROOT=/mnt/nfs/data/processed_dataset/alphasim/nurec2img/26.04_release_10hz_3cam_png
export NUREC_PREPARED_ROOT=/rhome/satyam/drivesuprim_workspace/nurec_satyam_ws/satyam/nurec_axe

NUREC_MAP_JOBS=8 bash scripts/nurec/prepare_usdz.sh
```

기존 map bundle은 자동으로 건너뛴다.

### 전체 Full NuRec 4096 EPDMS 생성

map preparation 및 validation이 성공한 뒤 실행한다.

```bash
export NUREC_PREPARED_ROOT=/rhome/satyam/drivesuprim_workspace/nurec_satyam_ws/satyam/nurec_axe
export NUREC_VOCAB_PATH=/rhome/satyam/drivesuprim_workspace/nurec_satyam_ws/satyam/nurec_vocab/nurec_train_kmeans_4096x40x3.npy
export NAVSIM_TRAJPDM_ROOT=/rhome/satyam/drivesuprim_workspace/nurec_satyam_ws/satyam/nurec_epdms
export NUREC_EPDMS_NAME=nurec_full4096

# Score against the controller that will drive, not navsim's LQR.
export NUREC_USE_MPC=1

# Throughput settings. Every one of these is measured, not guessed -- see 14.
export NUREC_CONTROLLER_DTYPE=float32   # 1.66x on its own, 3.13x with batching
export NUREC_SIMULATE_GROUP_SIZE=8      # tokens rolled out per GPU call
export CUDA_VISIBLE_DEVICES=4,5,6,7     # leave the rest to whoever else is on the box

NUREC_PDM_THREADS=24 setsid nohup bash scripts/nurec/generate_epdms.sh \
  > "$NAVSIM_TRAJPDM_ROOT/epdms_run.log" 2>&1 &
echo $! > "$NAVSIM_TRAJPDM_ROOT/epdms_run.pid"
```

`setsid` matters: the run is over a day long and must survive the shell closing.

Sizing follows from two measured numbers, so re-derive it rather than copying it
onto different hardware: a token costs about 5.7 s of GPU and 22.8 s of CPU, so a
worker wants the GPU only a fifth of the time and it takes roughly six workers
per card to keep one busy. Worker count then sets GPU memory -- six workers of
eight tokens each is about 27 GB of a 48 GB card, which fits; twelve-token groups
do not, and thirty-six workers died on it. See 14.2.

최종 예상 경로:

```text
/rhome/satyam/drivesuprim_workspace/nurec_satyam_ws/satyam/nurec_epdms/
  ori/vocab_score_4096_nurec_full4096/nurec_full4096_epdms.pkl
```

중단 후 동일 명령을 다시 실행하면 기존 metric cache와 token별 score cache를 재사용한다.

### 강제 metric cache 재생성

정말 cache를 다시 만들어야 할 때만 사용한다.

```bash
NUREC_FORCE_METRIC_CACHE=true NUREC_PDM_THREADS=32 bash scripts/nurec/generate_epdms.sh
```

## 10. 전체 작업 실측 비용

2026-08-25 실측이다. 이전 판의 36–48시간 추정은 **단일 worker 시간을 worker 수로
나눈 값**이었고, 선형 확장을 가정한 잘못된 계산이었다.

token 하나를 32-worker 부하에서 분해하면 다음과 같다.

| 단계 | 시간 | 비중 |
|---|---|---|
| MPC 시뮬레이션 (batch 4097) | 152.11 s | 71.1 % |
| 스코어링 (batch 4097) | 57.94 s | 27.1 % |
| MPC batch-1 (human penalty filter) | 3.69 s | 1.7 % |
| 나머지 (cache 로드/변환/traffic) | 0.19 s | 0.1 % |

병목은 GPU 위의 MPC다. batch-4096 MPC 한 번은 GPU를 독점하면 22.66 s인데
8-worker가 한 GPU를 공유하면 152 s가 된다. 즉 GPU 공유는 병렬 이득 없이 거의
완전히 time-slice된다. 따라서 상한은 `4 GPU ÷ 23 s = 0.174 tok/s`다.

| worker | 실측 처리량 | 전체 예상 | GPU 메모리/장 | load avg |
|---|---|---|---|---|
| 1 | 0.0174 tok/s | — | — | — |
| 16 | 0.147 tok/s | 90.8 h | 7.0 GB | ~25 |
| **32** | **0.164 tok/s** | **81.4 h** | 14.1 GB | ~45 |
| 이론 상한 | 0.174 tok/s | 76.7 h | — | — |

32 worker가 상한의 94%이므로 worker를 더 늘려도 의미가 없다. metric caching
0.23 h를 더해 **총 약 81.6시간**이다.

시도했다가 기각한 것들이다.

- **float32 전환**: 1.36배(22.66 → 16.64 s)에 그친다. 이 MPC는 FP64 ALU가 아니라
  커널 런치 바운드다. 대신 4,096개 중 30개 proposal이 최대 4.87 m 어긋난다.
- **human penalty filter의 batch-1 MPC를 본 batch에 병합**: 전체의 1.7 %다.

디스크는 per-token 컴포넌트 파일이 74 KB × 48,038 = 약 **3.6 GB**다.

주의: 이 노드는 공유 자원이다. 다른 사용자의 NuRec 5-cam 렌더(GPU당 약 4 GB)와
CarlaUE4가 동시에 돌고 있어 36코어 기준 load가 25–70을 오간다. 위 수치는 그
상태에서의 실측이며 외부 부하에 따라 변한다.

## 11. 전체 완료 후 필수 검증

`scripts/nurec/validate_epdms.py`가 아래 항목을 48,038개 전체에 대해 자동으로
확인한다. 하나라도 hard check에 걸리면 종료코드 1을 반환한다.

```bash
python scripts/nurec/validate_epdms.py \
  --trajpdm-root "$NAVSIM_TRAJPDM_ROOT" \
  --name nurec_full4096 \
  --expected-tokens 48038
```

| 검사 | 성격 |
|---|---|
| per-token 파일 수 == index의 토큰 수 | hard |
| 잔여 `.part` 파일 없음 (writer가 중간에 죽지 않았음) | hard |
| 9개 키 전부 존재, 예상 밖 키 없음 | hard |
| 모든 value shape `(4096,)`, dtype `float16` | hard |
| NaN/Inf 0개, 모든 값 `[0, 1]` | hard |
| `pdm_score`가 동봉된 컴포넌트의 EPDMS 집계와 **bit-exact** 일치 | hard |
| `traffic_light_compliance`가 전 토큰 전 후보에서 1.0 | hard |
| run report의 failed/unaccounted가 0 | hard |
| EPDMS-only pkl과 per_token이 표본 200개에서 정확히 일치 | hard |
| 전 후보 0점 토큰 비율 (2% 초과 시 경고) | advisory |
| 전 후보 동일점수 토큰, 분포 통계 | advisory |

검증기 자체는 결함 9종(잘린 파일, 키 누락, shape 오류, NaN, 범위 이탈, dtype
오류, 인덱스 뒤섞기, 신호등 중립화 파손, 잔여 `.part`)을 일부러 주입해 전부
잡는 것을 확인했다. 항상 통과만 하는 검증기가 되지 않도록 이 negative test를
거치지 않은 변경은 하지 않는다.

`--limit N`으로 앞 N개만, `--workers`로 병렬도를 조절한다. 2,248 토큰 기준
1.3초.

## 12. 제어기 교체와 학습 연결

### 12.1 navsim LQR 대신 NuRec Runtime MPC

vocabulary 후보를 채점할 때 "이 궤적을 실제로 주행하면 어떻게 되는가"를 답하는
주체가 제어기다. navsim 기본값은 kinematic bicycle 위의 LQR인데, 이는 대회
Runtime이 쓰는 제어기가 아니다. `NUREC_USE_MPC=1`이면 NuRec Runtime의 linear
MPC(dynamic bicycle 8-state, condensed QP, horizon 20 × 0.1 s)를 torch/GPU로
포팅한 `NuRecSimulator`로 교체한다.

수치 동등성은 vendored NVIDIA 원본 대비 계층별로 확인했다.

| 계층 | 허용오차 |
|---|---|
| vehicle model | `atol=1e-12` |
| linearization | `rtol=1e-11` |
| MPC control | `atol=9.0e-06` |

바꾼 이유는 라벨이 실제로 달라지기 때문이다. EPDMS 상관계수는 0.915로 높지만
**상위 10개 후보가 하나도 겹치지 않는다**(top-100 49 %, top-1000 90 %). 학습이
소비하는 것은 상위 순위이므로 상관계수는 여기서 의미가 없다. 요청 궤적 대비
최종 추종오차 중앙값은 LQR 7.62 m, MPC 0.60 m다.

원본 대비 의도적으로 다르게 둔 곳이 하나 있다. 참조 궤적의 지평이 끝까지
소진되면 원본은 정지 상태를 이어 붙이는데, 그러면 모든 후보가 궤적 끝에서
제동해 속도가 12.0 → 6.25 m/s로 무너진다. 여기서는 종단 속도를 유지하며
연장한다.

### 12.2 Ray worker가 GPU를 못 보던 문제

nuplan의 `worker_map`은 `Task(fn=fn)`을 `num_gpus=None`으로 제출하고, Ray는
이를 0으로 읽어 worker에 `CUDA_VISIBLE_DEVICES=""`를 강제한다. 결과적으로 모든
worker에서 `torch.cuda.is_available()`이 False가 되어 MPC가 조용히 CPU로
떨어진다. 답은 맞지만 약 70배 느려지고 로그에는 아무 흔적도 남지 않는다.

`gen_vocab_full_score.py`의 `map_with_gpu_share()`가 Ray에 GPU 지분을 요청해
이를 막는다. 32 worker / 4 GPU에서 GPU당 정확히 8 프로세스로 균등 배분된다.
이로써 `resolve_device()`의 `pid % device_count` 추측도 실질적으로 무효화된다.

### 12.3 학습이 읽는 artifact

손실 함수는 `scores[token][metric]` 형태로 8개 metric을 읽는다. 최종
`*_epdms.pkl`은 `{token: (4096,) 배열}` 하나뿐이므로 **그대로 넘기면
TypeError가 난다.** 학습은 반드시 per-token 컴포넌트 디렉토리를 봐야 한다.

```text
agent.config.ori_vocab_pdm_score_dir = <출력>/ori/vocab_score_4096_<name>/per_token
```

`train.sh`가 `NUREC_ORI_PDM_SCORE` 옆의 `per_token`을 자동으로 잡고, 없으면
실행 전에 실패한다. 파일당 74 KB이며 `_LazyTokenScores`가 LRU로 지연 로딩하므로
15 GB 단일 pickle을 상주시키지 않는다.

`pdm_score`라는 key 이름은 legacy다. **그 안에 들어 있는 값은 EPDMS다.** v1
값은 `v1_terms` / `_comfort`에 따로 있다. 저장된 `pdm_score`가 컴포넌트
재집계와 bit-exact로 일치함을 확인했다(`max|diff| = 0.000e+00`).

### 12.4 Route

PAI Track Route는 20 slot, 유효 약 10개(42.105–80.0 m), arclength 간격
4.210526 m, ego rig frame, 미사용 slot은 NaN이다. `use_route`의 기본값은
False이고 어떤 agent config도 이를 켜지 않으므로, `train.sh`가
`agent.config.use_route=true`를 넘긴다(`NUREC_USE_ROUTE=0`으로 끌 수 있다).
route는 NuRec 데이터의 속성이므로 NuRec 진입점에서만 켜는 것이 맞다.

### 12.5 Human penalty filter와 지도 커버리지

라벨을 읽을 때 반드시 알아야 하는 항목이다. 값이 틀린 것은 아니지만, 메트릭
목록이 시사하는 것보다 실제 신호량이 적다.

`pdm_score.py`의 human penalty filter는 인간 GT가 어떤 metric에서 0점을 받으면
**그 metric을 해당 프레임의 후보 4,096개 전부에 대해 1.0으로 중화**한다.

```python
if human_pdm_result[column].iloc[0] == 0:
    gt[column] = np.ones_like(gt[column])
```

인간궤적 50 토큰을 직접 채점한 실측 발동률이다.

| metric | 인간 GT가 0점 → 전 후보 면제 |
|---|---|
| drivable_area_compliance | **42.0 %** |
| lane_keeping | **34.0 %** |
| driving_direction_compliance | **20.0 %** |
| no_at_fault_collisions | 2.0 % |
| time_to_collision_within_bound | 2.0 % |
| ego_progress | 0 % |

완료 토큰 2,843개에서 토큰당 평균 **1.37개** metric이 상수가 되고, 아무것도
면제되지 않은 토큰은 **37.7 %**뿐이다. 따라서 면제가 잦은 프레임의 순위 신호는
충돌·TTC·progress에서 나온다고 보아야 한다.

원인은 우리 코드가 아니라 지도 커버리지다. 확인한 사항이다.

- 지도 자체 질의(`points_in_polygons`)와 독립 shapely 검사는 4,000개 표본에서
  **불일치 0.00 %**다. 기하 판정은 정확하다.
- 좌표 정렬도 정상이다. 자차가 drivable 밖인 프레임은 50개 중 3개뿐이고,
  이탈 방향의 부호가 혼재한다(+0.11 / +7.06 / −2.76 m). 체계적 오프셋이라면
  한쪽으로 쏠려야 한다.
- 인간궤적이 벗어날 때의 이탈거리는 median 2.0 m, p90 11.0 m, 최대 26.7 m다.
  폴리곤이 미세하게 타이트한 수준이 아니라 복원된 지도에 영역이 실제로 없다.
- 안쪽 프레임의 경계까지 여유거리는 median **1.51 m**다. Pacifica 반폭이 약
  1.0 m이므로 정상 주행 중에도 차체 모서리가 경계에서 0.5 m 안쪽이다. metric이
  중심이 아니라 **네 모서리**를 보기 때문에 중심 기준 26 %가 모서리 기준
  42 %로 올라간다.

근본 원인은 `DRIVABLE_AREA` 레이어가 비어 있다는 것이다. metric cache 20개의
레이어 구성을 세어보면 다음과 같다.

| 레이어 | 개수 | 존재하는 clip |
|---|---|---|
| LANE | 1,118 | 20/20 |
| ROADBLOCK | 588 | 20/20 |
| LANE_CONNECTOR | 289 | 16/20 |
| INTERSECTION | 34 | 16/20 |
| **DRIVABLE_AREA** | **0** | **0/20** |
| **CARPARK_AREA** | **0** | **0/20** |

nuPlan에서 drivable 판정을 지배하는 것은 도로면 전체를 덮는 `DRIVABLE_AREA`인데
NuRec USDZ에는 이를 만들 `drivable_space.parquet`가 아예 없다. 따라서 판정에
쓰이는 것은 차선 폭 `ROADBLOCK`과 교차로 34개뿐이며, 실질적으로 navsim보다
훨씬 엄격한 정의로 채점된다. **채점 코드와 레이어 어휘는 navsim과 동일하고,
다른 것은 내용물이다.**

`road_boundary.parquet`(246행)는 존재하고 `nurec_map_data.py` 211행이 읽지만
어떤 레이어로도 쓰이지 않는다. 다만 이것은 닫힌 폴리곤이 아니라 열린
폴리라인이므로, 도로면으로 만들려면 마주보는 경계선을 짝지어 폴리곤화하는
별도 작업이 필요하다.

시각화(`<validation>/metric_cache_check/drivable_gap.png`)로 확인한 실패는 세
가지 서로 다른 유형이며, 대응 방법이 다르다.

| 유형 | 예시 clip | 팽창 0.25 m | 팽창 1.0 m | 해결책 |
|---|---|---|---|---|
| 지도 커버리지가 끊김 | ad134e5b | 11→11 | 11→11 | **팽창 무효.** 복원 범위 문제 |
| 접속부 폴리곤 미연결 | 56bf7a4b | 14→11 | 14→4 | 팽창 또는 폴리곤 연결 |
| 차체 모서리만 스침 | 22f408ec | 2→0 | 2→0 | 팽창 0.25 m로 해결 |

50 토큰 전체에 대한 팽창 실측이다.

```
팽창 0.00 m -> 42.0 %      0.25 m -> 24.0 %      1.00 m -> 22.0 %
팽창 1.50 m -> 16.0 %      3.00 m -> 14.0 %
```

0.25 m에서 모서리 스침이 해소되며 크게 떨어지고, 그 뒤로는 정체한다. 남는
14~22 %는 매핑되지 않은 곳을 실제로 주행한 경우라 팽창으로 메울 수 없다.
시각화의 세 번째 행에서 `road_boundary`가 `ROADBLOCK`과 거의 겹치는 것에서
보이듯, `DRIVABLE_AREA`를 복원하더라도 폭이 크게 넓어지지는 않는다. 얻는 것은
주로 커버리지 연장이며 그것은 원본 복원 품질에 달려 있다.

필터를 끄면 안 된다. 끄면 인간궤적이 42 %의 프레임에서 0점을 받고, 모델은
사람처럼 주행하는 것을 회피하도록 학습한다. 지금 동작이 맞다.

개선하려면 drivable 폴리곤을 차체 반폭만큼 팽창시키거나 지도 복원 커버리지를
넓혀야 하는데, 둘 다 라벨을 바꾸는 변경이므로 재생성이 필요하다. 학습 결과를
보고 판단한다.

### 12.5.1 metric cache 구성요소 감사

`drivable_area_map` 외의 입력도 그것을 소비하는 metric 관점에서 점검했다.
180 토큰 표본이다.

**이상 없음**

| 구성요소 | 확인 |
|---|---|
| `human_trajectory` | 전부 8 pose / 4 s |
| `past_human_trajectory` | 전부 1.5 s (history comfort 입력) |
| `trajectory` (PDM 기준) | 전부 5.0 s |
| `scene_type` | 전부 ORIGINAL |
| `observation` 박스 크기 | median 8.0 m² (차량 크기), 200 m² 초과 0 %, 자차가 t=0에 객체와 겹치는 경우 0건 |
| red light 토큰 | 0개 — 신호등 중립화와 일관 |
| `centerline` 샘플링 | 간격 균일 약 1.0 m |

**`route_lane_ids`: 의심했다가 해소**

route id의 25.8 %가 `drivable_area_map`에 없다. 그러나 클립 전체 지도에는
**전부 존재**하며(유령 참조 0개), 없는 것들은 모두 자차에서 101 m 이상
(median 184 m, max 454 m) 떨어져 있다. `drivable_area_map`의 크롭 반경이
약 100 m라 잘린 것뿐이고 4 s 궤적이 닿지 않는다. 결함이 아니다.

**신호 손실 요인 (값은 옳으나 변별력이 없어지는 경우)**

*centerline 앞쪽 여유 부족.* PDM의 progress는 후보 중 최대값으로 정규화하므로,
centerline이 짧으면 모든 후보가 끝에 도달해 같은 값이 된다.

| 앞쪽 여유 | 토큰 | progress 상수 비율 | progress 평균 |
|---|---|---|---|
| 0–25 m | 18 | **50.0 %** | 0.925 |
| 25–50 m | 19 | 42.1 % | 0.855 |
| 50–100 m | 54 | 20.4 % | 0.720 |
| 100 m 이상 | 89 | 20.2 % | 0.646 |

전체의 **20 %가 앞쪽 여유 50 m 미만**이다. 여유가 충분해도 20 %는 상수이므로
그쪽은 실제 동점이다. 즉 progress가 상수인 24.9 %는 "전부 만점"만으로 설명되지
않으며, 절반 가까이는 centerline 소진이 원인이다.

*관측 객체 전무.* 전 시점 객체가 0개인 토큰이 **3.3 %**이며 그 전부에서
collision이 상수다. 빈 도로면 정상이나 그 프레임은 충돌·TTC 신호가 없다.
전체로는 collision 13.9 %, TTC 15.0 %가 상수다.

*centerline 역방향.* 인간 GT가 centerline 기준 뒤로 진행하는 토큰이 6/180이나
그중 4개는 진행량이 −0.0 m인 정차 상태다. 실제 역방향은 약 **1 %**이며, 그
프레임은 human penalty filter가 progress·direction을 면제하므로 잘못된 벌점은
생기지 않는다.

*자차와 centerline의 이격.* median 0.46 m로 대체로 양호하나 1 m 초과가 17.8 %,
3 m 초과가 2.2 %, 최대 7.10 m다. `lane_keeping`이 이 거리를 사용한다.

**정리.** 잘못된 값은 발견되지 않았다. 발견된 것은 모두 "그 프레임에서 해당
metric이 변별하지 않는다" 유형이며, 원인은 네 갈래다 — 지도 커버리지(drivable),
centerline 길이(progress), 빈 장면(collision·TTC), human penalty filter.

### 12.6 카메라 이미지 (미해결)

3-camera 렌더는 1,603 clip 전부 완비되어 있으나 **타임스탬프가 로그 grid와
맞지 않는다.**

- 모든 clip이 30.6–63.9 ms 늦게 시작한다(중앙값 45.3 ms). 171 clip이 60 ms
  허용치를 넘는다.
- 모든 clip이 한 프레임 일찍 끝나 마지막 로그 프레임에 대응 이미지가 없다
  (1,559 clip, 결손 약 0.45 s).

현재 기준으로는 1,603 중 **41 clip만 통과**한다. 5-camera 렌더도 타임스탬프가
동일하므로 렌더셋 선택으로는 해결되지 않는다. 위상 보정 재렌더를 요청했다.
요청서와 clip별 요구 타임스탬프는 다음에 있다.

```text
<validation>/RERENDER_REQUEST.md
<validation>/required_render_timestamps.json    # 1,603 clips, 65,671 timestamps
```

`sync_images.sh`는 1,603개 로그 pickle을 제자리에서 다시 쓴다. EPDMS 실행이
같은 파일을 읽는 동안에는 돌리지 않는다. 올바른 순서는 다음과 같다.

```text
EPDMS 완료 → sync_images.sh → nurec_validate → train.sh
```

## 13. 알려진 제한사항

- 동적 신호등 phase가 없으므로 신호 위반을 평가하지 않는다.
- 따라서 공식 NAVSIM EPDMS와 절대값 비교는 유효하지 않다.
- ClipGT intersection roadblock은 여러 connector lane을 포함할 수 있다. 표본 metric cache와 GT 정렬은 통과했지만 full run 완료 후 scene별 score collapse/outlier 검사는 반드시 수행해야 한다.
- 1,607 clips 중 4개는 route를 신뢰할 수 없어 제외했다(5.5). 데이터의 0.25%에 해당한다. NuRec 재추출이나 lane 매칭 개선으로 되살릴 여지가 있다.
- 유지된 clips 중 7개는 clip 앞이나 뒤에 짧은 ego 미매칭 구간이 있다. 해당 구간은 직전 route suffix를 유지하며, 전체의 75.7% 이상이 매칭된 clips라 영향은 국소적이다. 정밀도가 더 필요하면 `--min-ego-match-fraction`을 올린다.
- 이미지 생성 완료 여부는 EPDMS 계산을 막지 않는다. 이미지는 이후 camera-based AXE training에만 필요하다.
- `/rhome` 및 alphasim storage는 NFS이므로 간헐적인 `rpc_wait` 지연이 발생할 수 있다. 이는 계산 로직 오류가 아니며 재개 cache로 대응한다.
- 이미지는 아직 부착되지 않았고 이유가 둘이다. 첫째, `nurec_attach_images`가
  1,603개 로그 중 41개만 붙인다 — 로그는 프레임이 41개인데 렌더는 40장이고
  타임스탬프가 60 ms 한계에 대해 일관되게 57.9 ms 벌어져 있어, 대응 이미지가
  없는 마지막 프레임 하나 때문에 클립 전체가 탈락한다. scene filter가
  history 4 / future 8이므로 로그의 마지막 프레임은 future로만 쓰여 이미지가
  필요 없다. 둘째, 경로만 붙여서는 부족하다 — 픽셀은 f-theta인데 로그의
  calibration은 고정 NuPlan pinhole이라 BEV projection이 기하학적으로 성립하지
  않는다. `docs/nurec/IMAGE_GEOMETRY.md`에 실측과 남은 작업을 정리했다.
  EPDMS 계산에는 영향이 없다.
- 지도 커버리지 때문에 drivable/lane_keeping/direction이 각각 42/34/20 %의
  프레임에서 상수가 된다(12.5). 값이 틀린 것은 아니나 신호량이 줄어든다.
- 같은 성격의 신호 손실이 세 갈래 더 있다(12.5.1): centerline 앞쪽 여유가
  50 m 미만인 토큰 20 %에서 progress가 포화하고, 관측 객체가 전무한 토큰
  3.3 %에서 collision·TTC가 상수가 되며, 약 1 %는 centerline이 역방향이다.
  세 경우 모두 잘못된 벌점은 생기지 않는다.
- 이 노드는 공유 자원이다. 다른 사용자의 렌더와 시뮬레이터가 같은 GPU/CPU를
  쓰므로 처리량 실측치는 그때의 외부 부하에 종속된다.
- `NuRecSimulator`의 통합 방식은 alpasim `System` 클래스를 받아 직접 대조했고
  열두 항목 중 열하나가 일치, 하나가 버그로 드러나 수정했다(14.1). 남는 차이는
  세 가지뿐이며 셋 다 성격이 다르다. (a) 참조 궤적이 소진된 뒤의 처리 — Runtime은
  매 사이클 새 계획을 받아 그 상황을 겪지 않으므로 4초 개루프 롤아웃에서는
  "원본과 동일"이 정의되지 않는다. (b) 배치 GPU 솔버가 OSQP가 아닌 ADMM이라
  제어값이 `atol=9e-6`에서 일치한다. (c) 원본은 상대 pose를 float32로 절삭해
  누적하는데 이식본은 float64다 — 이식본이 더 정밀한 방향의 불일치다.

## 14. 보고 프레임 수정과 처리량 (2026-08-25)

The Runtime's `System` class became available and closed the two things the MPC
port had been carrying as assumptions. It also exposed a bug.

### 14.1 The port reported the wrong point on the car

`VehicleModel` integrates velocity and acceleration at the centre of gravity,
1.59 m ahead of the rig origin. navsim's state array is rear-axle by convention
(`ego_state_to_state_array` fills it from `rear_axle_velocity_2d`), and the
Runtime's `System._build_dynamic_state_in_rig_frame` converts before reporting.
The port wrote the CG values straight into the array:

```text
v_rig_y = v_cg_y - yaw_rate * l_rig_to_cg
a_rig_x = a_cg_x + yaw_rate**2 * l_rig_to_cg
a_rig_y = a_cg_y - yaw_acceleration * l_rig_to_cg
```

Positions were never wrong -- the model already integrates x/y from the rig-frame
lateral velocity, which is why it matched the reference to 1e-12. Only the
reported velocity and acceleration were, and `history_comfort` and TTC read
exactly those. On a rotating body the rear axle is the calmer point: median
|a_y| dropped 6x and p90 4x once converted, which is why comfort is measured
there in the first place.

Measured before fixing: comfort changed on 0.07-0.22% of candidates, and the
tracking figures were bit-identical afterwards (final errors 0.891 / 0.270 /
0.206 / 0.277 m, unchanged to three decimals), confirming the fix touched
reporting and nothing else.

`tests/test_nurec_rig_frame_reporting.py` checks the three formulas term by term,
and `tests/test_nurec_reported_state_matches_reference.py` runs one proposal
through the vendored `VehicleModel` + `LinearMPC` stepped the way `System._step`
steps it and compares every reported field. That second test is the one that
would have caught this: the layer tests covered the model, the linearisation and
the QP, but nothing had ever compared the row navsim reads.

The audit against `System` found eleven other things already correct: the step
order, the rig-frame reference transform, the sample times and interpolation, the
MPC cadence, Ford Fusion parameters (the Runtime does not override them), all
seven gains, the initial steering seeding, and the untransformed initial velocity.

### 14.2 Throughput

Measured on an idle RTX A6000, per token:

```text
batch  4096 float64   17.75 s   1.00x     (one token per call)
batch 32768 float64   12.83 s   1.38x
batch  4096 float32   10.68 s   1.66x
batch 32768 float32    5.68 s   3.13x
```

One token's 4096 proposals leave a 48 GB card less than half busy, so tokens are
rolled out in groups (`NUREC_SIMULATE_GROUP_SIZE`). `simulate_proposals_from_states`
takes a starting state per proposal, which is what makes grouping possible;
`tests/test_nurec_token_batching.py` asserts a token scores identically whichever
group it lands in.

float32 was expected to stall the ADMM iteration, which terminates on a 1e-4
residual. It does not. Across fourteen scored scenes the correlation against
float64 is 0.9986-1.000000 and 0.19% of metric values move. Beware the obvious
check here: top-k overlap looked catastrophic on one scene (top-1 0%), but that
scene has nine distinct scores among 4096 candidates and 58 of them tie for
first -- argsort was picking a different member of the tie. Compare *how many
candidate scores changed*, which is tie-independent, not rank overlap.

End to end, the files the pipeline writes differ from a float64 single-process
unbatched recomputation on 0.31% of stored values.

Measured 0.45-0.49 tok/s on four cards, about 27 hours for 48,038 tokens against
roughly 61 hours unbatched in float64.

Three things that did not work, so nobody re-tries them:

- **More processes per card.** 1/2/4 processes gave 0.057/0.050/0.051 tok/s per
  card -- flat to slightly worse. Kernels from separate contexts serialise.
- **MPS.** 10% slower. The kernels are already large enough that it only adds
  overhead.
- **Bigger groups.** 16 and 24 tokens are 5-6% cheaper per token but double and
  triple the memory, which forces the worker count down. Worker count is what
  keeps the cards fed, so the trade loses.

Also measured and not worth doing: rolling out on the idle CPU cores alongside
(107.8 s per token, 6x slower than a GPU, +5% total), and pre-decompressing the
metric caches (lzma is 0.01 s of a 22.8 s CPU phase).

### 14.3 Two traps in the batched path

Both were found by a smoke run and would have killed a day-long run:

- **Out of memory is not always `torch.cuda.OutOfMemoryError`.** cuSOLVER's
  workspace allocation raises a plain `RuntimeError` with "out of memory" in the
  message. The regrouping fallback catches both now; it caught only the former at
  first, and the whole run died on the second.
- **The allocator holds the rollout workspace through the scoring phase**, which
  is four fifths of the cycle, denying it to the five other workers on the card.
  `_simulate_group_once` hands it back after each group. Measured no slower.

Run a smoke on a scratch output root before any long run. It cost twenty minutes
and caught both of these.
