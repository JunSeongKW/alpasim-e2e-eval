# 짝이 맞아야 하는 것들

이 파이프라인에는 **따로 고를 수 있지만 따로 골라서는 안 되는** 값이 몇 개 있다.
어긋나도 예외가 나지 않고, 로그도 정상으로 보이고, 점수만 조용히 나빠진다.
그래서 여기 모아둔다. 새로 발견하면 여기에 추가할 것.

---

## 1. 카메라 K — 학습 정류 vs 평가 드라이버

```
K = [[377, 0, 256], [0, 377, 128], [0, 0, 1]]   512x256, distortion 0
```

| 어디 | 무엇 |
|---|---|
| 학습 | `navsim/planning/data/nurec_rectify/target.py` 의 `TARGET_*` |
| 평가 | 챌린지 드라이버 `driver.py` 의 `_VIRTUAL_*` |

렌더는 f-theta 어안이고 모델은 512x256 핀홀을 읽는다. 양쪽이 같은 핀홀로 정류해야
`lidar2img` 가 같아진다. 어긋나면 **한 카메라로 학습하고 다른 카메라로 채점**된다.

**지키는 방법** — `tests/test_nurec_rectified_geometry.py::test_the_target_pinhole_is_the_drivers`
가 드라이버 소스를 파싱해 비교한다. 빨간불이면 두 값이 다른 것.

K 를 바꾸려면 **셋이 함께** 움직여야 한다: target.py + driver.py + 그 K 로 재정류한
이미지. 그리고 그 K 로 학습한 체크포인트로만 평가한다.

---

## 2. 궤적 vocab — PDM 점수 vs 모델 vs 평가

4096개짜리 vocab 이 **두 개** 있고 호환되지 않는다.

| | 파일 | 4초 끝점 x (index 39) | 쓰는 곳 |
|---|---|---|---|
| NuRec | `nurec_train_kmeans_4096x40x3.npy` | -4.9 ~ 159.2 m (중앙값 35.2) | 2026-08-26 이후 학습 |
| NAVSIM | `test_4096_kmeans.npy` | -1.9 ~ 58.7 m (중앙값 17.9) | 그 이전 체크포인트 |

> 이 표는 2026-08-26 까지 `-1.0 ~ 31.6` / `-0.6 ~ 11.7` 로 적혀 있었다. 그 값은
> **index 7 의 값이고, index 7 은 0.8 초지 4 초가 아니다.** vocab 은
> `TrajectorySampling(num_poses=40, interval_length=0.1)` 이라 4 초는 index 39 다.
> GT future trajectory 쪽이 0.5 초 간격 8 포즈(=4 초)라, 그 규약을 vocab 에
> 잘못 적용하면 index 7 이 4 초로 보인다. 두 배열을 비교할 때는 vocab 을
> `np.arange(4, 40, 5)` 로 뽑아야 GT 8 포즈와 시각이 맞는다.

per-token PDM 점수는 **vocab 인덱스로 매겨진 길이 4096 배열**이다. 점수 파일과
그것을 만든 vocab 은 하나의 산출물이고, 짝이 어긋나면 점수가 가리키는 궤적과
모델이 고르는 궤적이 달라진다 — 예외 없이, 조용히.

**지키는 방법** — `setup_nurec.sh` 의 `NUREC_VOCAB` 로만 고른다.

```bash
NUREC_VOCAB=nurec    # 기본. data/epdms 를 만든 vocab
NUREC_VOCAB=navsim   # 이전 체크포인트(epoch=03-step=2656 등) 평가 시
```

`source setup_nurec.sh` 첫 줄에 `vocab=` 가 찍힌다. 확인하고 시작할 것.

---

## 3. 컨트롤러 — vocab 에 묶여 있다

| | 컨트롤러 | 시대 |
|---|---|---|
| NAVSIM | LQR over kinematic bicycle (`PDMSimulator`) | `NUREC_VOCAB=navsim` |
| NuRec | linear MPC over dynamic bicycle (`NuRecSimulator`) | `NUREC_VOCAB=nurec` |

EPDMS 는 "이 궤적을 컨트롤러가 따라갔을 때 무슨 일이 일어나는가" 를 점수로 만든다.
컨트롤러가 다르면 같은 궤적도 다른 점수를 받는다. 학습 타깃과 모델 채점이 같은
컨트롤러여야 비교가 된다.

vocab 과 컨트롤러는 **같은 두 시대를 가리키므로** `setup_nurec.sh` 가 하나로 묶었다.
`NUREC_VOCAB` 만 고르면 `NUREC_USE_MPC` 가 따라온다.

```
NUREC_VOCAB=navsim   ->  test_4096_kmeans.npy              + LQR
NUREC_VOCAB=nurec    ->  nurec_train_kmeans_4096x40x3.npy  + MPC   (기본)
```

교차시킬 이유가 정말 있으면 `NUREC_USE_MPC` 를 직접 주면 덮어쓴다.

`source setup_nurec.sh` 첫 줄에 `vocab=` 와 `controller=` 가 같이 찍힌다.

> 2026-08-26 에 이 항목이 어긋난 채로 한 번 채점했다. `score_epdms.sh` 가
> 컨트롤러를 지정하지 않아 config 기본값 LQR 로 돌았는데, 마침 대상이 NAVSIM
> 체크포인트라 결과적으로는 맞는 조합이었다. 지금은 vocab 이 정한다.

## 4. 체크포인트 — 기하 vs vocab

체크포인트 하나에 **K 와 vocab 이 함께 박혀 있다.** `inspect_checkpoint.py` 는
텐서 모양만 보므로 100% 로드되어도 기하와 vocab 이 맞는지는 알려주지 못한다.

| 체크포인트 | K | vocab |
|---|---|---|
| `epoch=03-step=2656` (NAVSIM planonly) | 412/366 | NAVSIM |
| 2026-08-26 이후 학습분 | 377 | NuRec |

평가할 때 이 표를 보고 `NUREC_VOCAB` 을 맞출 것. K 는 드라이버 쪽이라 바꿀 수
없으니, 다른 K 로 학습된 체크포인트를 평가하면 그만큼 손해를 보고 시작한다.

---

## 5. 이미지 트리 — 정류본 vs 그것을 가리키는 로그

`data/prepared` 의 로그에 든 `data_path` 는 `data/images` 기준 상대경로다.
이미지 트리를 갈아끼우면 `finalize_data.sh` 를 다시 돌려 로그를 재작성해야 한다.
`setup_nurec.sh` 의 체크가 `prepared root` / `rectified images` 를 각각 보지만,
**둘이 서로 맞는지는 보지 않는다.**

**지키는 방법** — 이미지 트리를 바꿨으면 항상

```bash
bash scripts/nurec/finalize_data.sh
```

프리플라이트가 `checked_images` 로 실제 파일 존재를 세므로, 어긋나면 거기서 걸린다.

---

## 점검 순서

학습이나 채점을 걸기 전에:

```bash
source setup_nurec.sh                                   # vocab= / controller= 확인
$PYTHON_BIN -m pytest tests/test_nurec_rectified_geometry.py -q   # K 일치
nvidia-smi                                              # 18.8 GB/장
```

`score_epdms.sh` 도 시작 시 같은 두 값을 찍는다.

| 대상 | 명령 |
|---|---|
| NAVSIM 시대 체크포인트 | `NUREC_VOCAB=navsim bash scripts/nurec/score_epdms.sh <ckpt>` |
| NuRec 시대 (기본) | `bash scripts/nurec/score_epdms.sh <ckpt>` |
