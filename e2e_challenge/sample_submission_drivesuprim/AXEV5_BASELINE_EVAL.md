# axe-v5 (제출 원본) 458 클립 평가 환경

대회에 제출한 이미지 `axe-v5`를 현재 개발본과 같은 458 클립에서 재평가하기 위한 환경.

## 이미지

```
696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v5
image ID  sha256:5e2ddfa9889d5ee75b84e14343dc4954fd03c598ef494fb8bd4cf129b370a346
CMD       python -m drivesuprim_challenge.driver
```

| 에셋 | 크기 | SHA256 |
|---|---|---|
| drivesuprim_vov.ckpt | 1.37 GB | 4be6312fc5d7a587ac5edd33723f3c9b3a4d5345196e0bdc7218251b36d59467 |
| dd3d_det_final.pth | 323 MB | 50892839b8d2be9002f74d8290c26c0623e103f973c7184f2d59ff98465104c3 |
| test_8192_kmeans.npy | 3.9 MB | cc44a31e75a53406db59f026f0358de97931e726f10254542f98d2a87a38ad35 |

## 현재 개발본과의 차이

### 모델

| | stage3 / flat ViT (현재) | axe-v5 |
|---|---|---|
| 백본 | bevformer_m / vits_flat | vov (VoVNet + DD3D) |
| vocabulary | 4096 | 8192 |
| fp16 | 0 | 1 |
| inference interval | 100000 us | 0 |
| max batch / wait | 1 / - | 2 / 5 ms |

### 랭킹

axe-v5에는 아래 설정 키가 존재하지 않는다. 패치 이전의 원본 DriveSuprim 랭킹이다.

- `feasibility_*` (drivable area / collision 1차 필터)
- `pdm_rank_product_exponents`, `pdm_rank_product_exponents_refine`
- `pdm_imi_rank_weight`, `pdm_imi_rank_weight_refine`

`DRIVESUPRIM_CONFIG_PATH` 오버라이드도 지원하지 않으므로 설정 주입이 불가능하다.

### MPC

MPC는 이미지가 아니라 시뮬레이터(`controller.gains.*`)에 있다. axe-v5는 오버라이드 없이 대회
제출 당시의 기본 게인으로 돌린다.

| 게인 | axe-v5 | 현재 튜닝본 |
|---|---|---|
| long_position_weight | 2.0 | 0.5 |
| lat_position_weight | 1.0 | 6.0 |
| idx_start_penalty | 10 | 3 |

`idx_start_penalty` 10 -> 3 은 지금까지 찾은 가장 큰 개선 레버였다. 따라서 이 비교는
모델 + 랭킹 + MPC 를 합친 전체 차이를 본다. 모델 기여만 분리하려면 axe-v5 를 튜닝 게인으로
한 번 더 돌려야 한다.

### 동일하게 유지한 것

- 시뮬레이터 이미지 `alpasim-base:0.89.0-f012862-casadi372`
- `CHALLENGE_COMPAT_MODE=1` (controller=nonlinear, force-GT 핸드오버)
- 458 클립(`nurec_val_clips_ids.txt`), SIM_STEPS=199, 16 워커, GPU 0-3
- 채점 코드 경로. 로컬 수정본은 `video.py` / `data.py` 시각화 전용뿐이다.

## 스크립트

| 파일 | 역할 |
|---|---|
| run_axev5_driver.sh | 드라이버 1개. 이미지 ENV를 덮어쓰지 않는다 |
| run_axev5_drivers_group.sh | (GPU, 포트) 쌍별 복제본 관리 |
| run_axev5_eval.sh | 체크포인트 label 게이트 제외 (axe-v5에는 해당 label이 없다) |
| run_axev5_sweep.sh | N 워커 오케스트레이션 |

실행:

```bash
CLIPS_FILE=$PWD/nurec_val_clips_ids.txt \
RUN_DIR=<runs>/val458_axev5 \
WORKERS=16 GPUS="0 1 2 3" BASE_PORT=6900 BASEPORT=6400 PREFIX=av5 \
RENDER_VIDEO=false SERVICE_STARTUP_TIMEOUT_SEC=5400 \
  ./run_axev5_sweep.sh
```

## 환경 구축에서 걸린 것

1. 기존 런처가 `DRIVESUPRIM_USE_FP16=0`, `INFERENCE_INTERVAL_US=100000`,
   `MAX_BATCH_SIZE=1`, `BACKBONE_TYPE` 등을 강제로 주입한다. 제출본 그대로 재현하려면
   전부 빼야 해서 전용 런처를 새로 만들었다.
2. axe-v5는 `DriveSuprim policy load complete` 로그를 stdout으로 내보내지 않는다.
   기존 sweep의 준비 판정이 이 문자열을 세므로 영원히 대기한다. 포트 리슨 기준으로 바꿨다.
   (참고: `-p` 로 publish 한 포트는 docker-proxy가 즉시 리슨하므로 이 판정도 느슨하다.
   실질 대기는 드라이버의 `DRIVESUPRIM_SESSION_STARTUP_TIMEOUT_S`가 담당한다.)
3. 모델을 GPU로 올리는 시점이 첫 추론 때다. 기동 직후 GPU 사용량은 CUDA 컨텍스트(877 MiB)뿐이다.

## 검증

1클립 end-to-end 통과. 클립 `00040136` 에서 axe-v5 = 0.000 (offroad, progress 0.126),
같은 클립에서 현재 모델 = 1.000.
