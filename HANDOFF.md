# HANDOFF — 이 파일 하나로 다음 에이전트가 이어받는다

마지막 갱신: 2026-10-07 11:50 KST (Codex)

에이전트(Claude Code, Codex 등)는 세션을 **시작할 때 이 파일과 `git log -10` 을 읽고**,
**끝낼 때 이 파일을 갱신하고 커밋**한다. 대화 원문은 옮기지 않는다. 규칙은 `AGENTS.md`.

## 1. 실행 중인 작업

2026-10-07 Codex: 사용자 요청으로 stage3_5cam_ep05_20261007의 441 평가를 GPU 0–7 / 각 6 / 총 48 worker로 진행한다. gRPC 8과 카메라별 NONE 렌더링은 원래 설정을 유지한다. 사용자는 높은 VRAM 사용량만으로 중단하지 말고 실제 OOM/평가 실패에만 대응하라고 명시했다. AGENTS의 GPU/16-worker 기본값보다 이번 요청이 우선한다.
- **중요한 오류 수정**: 제공된 spatial_cross_attention.py는 기존 이미지의 DRIVESUPRIM_FORCE_PYTORCH_MSDA=1을 읽지 않았다. 설치된 MMCV CUDA 커널이 이 GPU를 지원하지 않아 매 추론에서 오류를 출력하고 0 텐서를 반환했다. 단위 검증 입력에서 실제 0 / 정답 1을 확인했다. 이전 GPU 검증의 finite 출력만으로는 이를 발견하지 못했다.
- 별도 평가 policy 어댑터에서 해당 환경변수를 적용해 **원본 모델에 이미 있는 PyTorch fallback**을 선택한다. axe-v9가 사용한 같은 연산 경로다. 모델 원본 폴더/가중치/소스/BEV/threshold .6/.4/gate는 변경하지 않았다. policy 시작 때 GPU 샘플이 정확히 1인지 검증해 잘못된 커널이면 기동 실패하도록 했다.
- 새 이미지 ID `sha256:b798c846bc8f44ebc702f189f803b8da8accd437fc824e04020cfcb7a88777c7`. 이전 잘못된 이미지 ed845b...와 그 33개 완료/48개 진행 기록은 `runs/leaderboard-stage3-5cam-ep05-20261007/topology-history/20261007-115013-invalid-cuda/`에 삭제 없이 격리했다. **33개를 정상 점수로 재사용하지 않고 441개 전부 재평가**한다.
- CPU/CUDA 실제 전체 forward와 strict checkpoint, 5카메라 보정/순서/동기화/ego_geom 검증을 새 이미지에서 통과했다. CUDA kernel 오류가 없고 GPU operator assertion 통과. 원본 SHA256SUMS 전체 재검증도 성공했다. 기록은 `runs/prepare-stage3-5cam-ep05/{cpu,gpu}_validation_fixed_operator.json` 및 해당 로그, `invalid_cuda_operator.json`.
- launcher `e2e_challenge/5cam_eval/accelerate.py --reuse-services`: 기존 57 native renderer/physics/controller와 정확한 compose/network 설정을 보존하고 driver/runtime만 재시작한다. config 재생성에 따른 포트 변경/renderer 재기동을 피한다. PID는 `.cache/stage3-5cam-441.launcher.pid`, 로그는 `runs/leaderboard-stage3-5cam-ep05-20261007.speed48.progress.log` / `.speed48.wizard.log`.
- 최종 설정: gRPC 8, NONE, 모델 batch 1, driver CPU 8/32GiB, dev441×1/MPC1/.25/3/하모나이저/채점/정밀도 동일. VRAM은 run의 `vram-speed48.csv`/`vram-speed48-peaks.json`에 기록한다. 완료 후 GPU 해제 및 기존34+신규1 CPU joint leaderboard fit 자동 진행.

## 2. 최근 결과

### 5카메라 Stage 3 ep05 평가 준비 (2026-10-07)

- **11:50 정정**: 이전 CUDA 검증은 unsupported MMCV kernel이 0을 반환해도 finite만 검사하여 잘못 통과했다. GPU 결과 33개를 무효로 격리하고, 별도 어댑터가 기존 axe-v9 환경변수를 존중하도록 수정했다. 원본 폴더 변경 없이 새 CPU/CUDA 전체 forward 및 constant-sample=1 검증을 통과했다. 아래의 초기 이미지/기동 기록은 수정 전 이력이다.
- 모델 원본: `../models/stage3_5cam_ep05_20261007/AXEv1.0-NuRec/`. 전체 SHA256SUMS 검증 성공. checkpoint SHA `79c625f3a29c49da6b9605b1062de5e8e6ec5da8391362b25507e35725bb8b8e` (epoch=4, step=4075).
- 원본 폴더는 변경하지 않았으며 사용자 재확인 후 전체 체크섬 검사를 다시 통과했다. 원본 YAML의 drivable threshold=0.6 / agent confidence threshold=0.4, 원본 `_build_feasibility_mask`, ego_geom의 length/width×0.98을 유지한다. 별도 오버레이에서 시도했던 axe-v9 threshold=0.5/0.3·구 gate·shrink 미적용 변경은 취소하고 원본 구성으로 이미지를 재빌드했다.
- 모델 YAML을 변환할 때 바꾸는 것은 checkpoint/vocab 배포 경로, training=False/only_ori_input=True, pretrained 초기 다운로드 방지, README의 평가 명령에 맞춘 use_route=True뿐이다. 5개 카메라는 원본의 bev_num_cameras=5로 결정된다. flat-image 경로의 n_camera=3도 원본값으로 보존한다. 학습용 분기를 추론용으로 전환하고 카메라/API 입력을 연결하며 모델 내부 튜닝 값은 덮어쓰지 않는다.
- axe-v9 이미지 digest `85134c1ea9f09d140be610ca063c50ec60e99980c392e5bf81fe6563af503765`로 의존성/정밀도를 고정하고, 제공된 navsim/nurec_pf 소스·Hydra 설정·4096 vocab·checkpoint를 올렸다. 기존 3카메라 이미지와 소스는 변경하지 않음.
- 별도 드라이버 오버레이 `e2e_challenge/5cam_eval/drivesuprim_challenge/`: L1/L0/F0/R0/R1 순서, 후방 alias, 5개 동기화, [1,3,5,3,256,512] 영상/[1,3,5,4,4] 보정 입력. K377 pinhole 유지.
- 신규 소스는 `ego_geom` 입력을 요구한다. 차량 크기와 rig→box 중심을 세션별 API에서 받아, 학습 feature builder·기존 evaluator의 2% shrink와 동일하게 `[length×0.98,width×0.98,centre_x]`를 전달한다.
- 동일 441 USDZ에 후방 카메라 보정/촬영시각이 모두 존재함. offline NAVSIM의 418 유효 클립만 사용하는 방식으로 줄이지 않았다.
- GPU 없는 CPU에서 strict checkpoint(missing=0/unexpected=0), 학습/실시간 영상 보정, 카메라 순서·누락·동기화, 실제 forward의 유한한 40×3 출력 확인. 최종 이미지 재검증 기록: `runs/prepare-stage3-5cam-ep05/cpu_validation.json`, `cpu-validation.log`, `scene_audit.json`, `evaluation_config.yaml`, `image-id.txt`. `original_model_integrity.json`에는 원본 소스 368개 파일의 바이트 일치 및 경로/추론 모드/use_route 외에는 YAML 값 차이가 없음을 기록했다.
- 원본 설정을 보존한 최종 이미지에서 protobuf 세션/5개 실시간 PNG 콜백 및 실제 CPU forward 재검증 통과. `run.sh --check` 성공, GPU 점유 시 신규 실행을 exit75로 중단하며 run 폴더/드라이버가 생성되지 않음. leaderboard validator는 누락/중복 rollout/결측 score를 거부하는 것도 확인했고 기존 evaluator 테스트3개, ruff/black/bash 문법 검사를 통과했다.
- 즉시 실행 전 재검토(2026-10-07): 원본 전체 SHA256SUMS 재검증 성공. 실제 실행 스크립트 dry run과 Docker Compose 문법 검증을 통과했으며 441개 renderer-mount USDZ 링크/16 driver·worker·renderer·controller endpoints/MPC 1/0.25/3 일치를 확인했다. 이미지 3종 및 renderer cache 2762 MiB가 모두 로컬에 있다. 디스크 여유 1052.5 GiB, 포트 7160–7175/19900–19989와 신규 run 경로가 비어 있었다. `uv` wizard·leaderboard CLI는 offline 기동된다. 증거: `runs/prepare-stage3-5cam-ep05/deployment_validation.json`, `deployment-dryrun.log`, `deployment-dryrun/`.
- 실제 image entrypoint도 GPU 없이 검증: 평가와 같은 read-only/cap-drop ALL/no-new-privileges/32 GiB/8 CPU/tmpfs 2 GiB 조건에서 strict checkpoint 로드 및 실제 gRPC get_version 성공. 임시 CPU 컨테이너 하나만 정확한 이름으로 종료/제거했다. 증거: `driver_deployment_validation.json`, `driver-readonly-startup.log`.
- 기동 후 GPU 4의 평가 driver 안에서 원본 모델의 실제 5-camera CUDA forward도 확인했다. 원본 소스/모델 설정을 변경하지 않고 검사 helper만 메모리에서 CPU→CUDA로 전환했다. 카메라 순서/보정/ego_geom 및 유한한 40×3 출력 검증 성공: `gpu_validation.json`, `gpu-validation.log`. 모든 16 driver가 exact checkpoint missing=0/unexpected=0으로 CUDA 로드를 완료했다.
- `run.sh`: 4–7만 사용, 각 4 replicas(총16), 441×1 rollout, dev, lat/lon/idx=1/0.25/3, no video, FP32(기존 내부 ViT AMP 유지), reranker 없음. 원본 rollout 보존. GPU 점유/이미지 변경/결과 덮어쓰기 거부, `--resume` 제공.
- `leaderboard.py`: 기존 독립 local26+reference8을 `existing_subjects.json`에 고정; 34개 동일 441 입력 확인. 완료 후 신규 `stage3-5cam-ep05`와 총35개를 CPU에서 ZOIB 재적합(seed42/1000epochs/16particles, rank MC100000/seed0). 결과 `runs/leaderboard-35-with-stage3-5cam-ep05/local_leaderboard.csv`, `capability_ranking.csv`, `paired_with_axe_v9.txt`.
- 새 모델은 5카메라뿐 아니라 BEV 종방향 -56~56m, 격자56×112, key/value28×56, stage3 ep05 및 제공된 gate 소스/threshold도 바뀌었다. 동일 평가 계약에서 모델 구성 전체를 비교하며, 카메라 수만의 효과로 해석하지 않는다. 연구 질문과의 연결은 상황별 후방 입력 필요성을 분석할 441 주행 기록을 확보하는 데 있다.

### axe-v9 만점 미달 CSV 사유 추가 (2026-10-06)

- 기존 `e2e_challenge/axe_local_eval/data/axe_v9_441_nonperfect_clips.csv`의 202행/기존 ID·점수·순서를 유지하고 사유 및 근거 지표를 추가했다.
- 0점 97개: 책임 충돌 17, offroad 39, corridor 측면 이탈 40, 평가 RPC 연결 오류 1.
- 부분 점수 105개: 후방 충돌 이후 집계 제외 + progress 부족 70, 충돌 없는 progress 부족 35.
- 점수식은 3개 hard gate 중 하나라도 발생하면 0, 아니면 min(progress_clipped_rel / 0.8, 1). GT 주행거리 5m 미만은 progress 1점.
- 후방 충돌 자체에 별도 감점 계수는 없다. DEFAULT_MODIFIERS가 모든 충돌/도로 이탈 이후 시점을 집계에서 제외하므로 후방 충돌 클립도 부분 progress가 될 수 있다. 충돌이 없었을 경우의 점수는 추정하지 않았다.
- 원본 score_metrics와 점수식을 대조해 정상 평가 201개가 일치함을 검증. 평가 오류 1개는 원본 오류를 보존하고 주행 실패와 구분했다.

### axe-v9 441개 중 만점 미달 클립 CSV (2026-10-06)

- 원본: `runs/leaderboard-merged-route-ep30/aggregate/results-summary.json` (리랭커 없는 axe-v9, 클립당 1 rollout).
- 전체 441개 중 평균 scene score < 1인 202개 추출; 만점 239개 제외.
- 파일: `e2e_challenge/axe_local_eval/data/axe_v9_441_nonperfect_clips.csv`. 열: `clip_id`, `average_scene_score`, `clipgt_id`. 점수 오름차순, 동점이면 clip_id 순.
- 원본 `score`를 클립별 평균하고 반올림 없이 기록. 재채점하지 않음. clip_id는 clipgt_id에서 `clipgt-` 접두어를 뺀 UUID.

### 441 전체 통합 leaderboard 갱신 (2026-10-06)

- 완료된 독립 로컬 26개 + 참조 8개, 동일 441개 장면으로 ZOIB 동시 적합. 경고·제외 없음.
- 결과: `runs/leaderboard-261006/capability_ranking.csv`, `manifest.json`; 사용자 형식 CSV: `e2e_challenge/axe_local_eval/data/local_leaderboard_261006.csv`.
- axe-v10 = aug-ep29-final + reranker γ=0.01: Rank 10, PCS 2526, 장면점수 0.7180, at-fault 거리 1.2563 km, 95% 순위 구간 5–15.
- Rank 1: axe-v9+rerank-g0.02 (PCS 2683, 구간 1–7), Rank 2: axe-v9 (PCS 2683, 구간 1–7).
- Rank는 구간 상한 오름차순 → at-fault 거리 내림차순. PCS 단순 정렬이 아니다. 아래 이전 적합 값들과 PCS를 혼용하지 않는다.
- γ=0.005는 129 rollout만 완료돼 제외. ep24egobox-1roll은 기존 3-rollout 평가의 파생본이어서 중복 제외. axe-v8와 footprint 미적용 ep24는 장면당 3회 평균.
- axe-v10 수치는 제출 구성과 같은 checkpoint/reranker의 441 실행 결과이며, 제출 이미지 자체의 441 재평가나 TRT 441 결과가 아니다.

### ★ axe-v10 TensorRT 속도 (2026-10-02, 보고서 `e2e_challenge/axe_local_eval/trt_bench/REPORT_TRT_SPEED.md`)

이미지 백본(ViT-S)만 TensorRT FP16 엔진으로 바꿨다. 나머지는 PyTorch FP32 그대로다.

| | FP32 | TensorRT | 배율 |
|---|---:|---:|---:|
| 단독 측정 평균 / p50 | 58.1 / 57.6 ms | 41.0 / 40.4 ms | 1.42 / 1.43배 |
| 시뮬레이터 안 평균 / p50 (8클립, 1,494회) | 142.5 / 158.4 ms | 106.7 / 109.8 ms | 1.34 / 1.44배 |
| 추론 FPS (시뮬 안, 평균 기준) | 7.0 | 9.4 | |
| 드라이버당 VRAM | 3,235 MiB | 3,314 MiB | +79 MiB |
| 클립당 시뮬레이션 시간 | 256 s | 251 s | −2% (렌더링이 병목) |
| 8클립 판정 | 7 pass / 1 fail | 동일 | 7클립 점수 완전 동일 |

- 기존 `DRIVESUPRIM_USE_FP16=1` 은 0.95배로 오히려 느리다. ViT 가 이미 내부에서 bf16 으로
  돌고 있어서다(`bevformer_vits_amp=True`).
- 시뮬레이터 안 지연이 단독보다 2.7배 큰 것은 같은 GPU 의 렌더러와 경합하기 때문이다.
  최솟값은 단독 측정값과 같다. 옮겨 쓸 수 있는 수치는 ms 가 아니라 배율이다.
- 엔진: `trt_bench/engines/backbone_fp16_0e7d2b37b21b47eb.plan` (44.5 MiB, TRT 10.13, sm_90,
  git 제외). TensorRT 는 이미지에 넣지 않고 호스트 `../trt_site/`(3.2 GB, onnx 1.16.2
  포함)를 마운트해서 썼다.

### ★ 리랭커는 모델에 따라 정반대로 작동한다 (2026-09-27)

같은 arm(cache-centre-max), 같은 γ=0.02, 같은 계약으로 두 체크포인트에 리랭커를
붙여 리랭커 없는 자기 자신과 짝지어 비교했다.

| | axe-v9 | disjoint-ep04 |
|---|---:|---:|
| 장면점수 (리랭커 없음 → 있음) | 0.6993 → 0.7299 | 0.6328 → 0.6472 |
| 차이 | **+0.0305** | **+0.0144** |
| 개선 / 악화 클립 | 66 / 55 | 56 / **59** |
| 0점 → 득점 | 30 | 23 |
| **corridor 이탈** | 0.0909 → **0.0726** | 0.1224 → **0.1247** |

**핵심은 corridor 이탈이다.** 리랭커의 본업은 route 에서 벗어나는 궤적에 벌점을
주어 corridor 이탈을 줄이는 것인데, axe-v9 에서는 20% 줄였고 disjoint-ep04 에서는
오히려 늘렸다. 후자는 본업에서 실패하고 부수 효과(0점률 −0.014, 차선위반 −0.011,
도로이탈 −0.007)로 총점을 벌었다. 클립 건수로도 악화(59)가 개선(56)보다 많고,
총점이 오르는 것은 개선 폭이 커서다(개선 +23.56 / 악화 −17.16 / 순 +6.37).

해석: 리랭커는 route 준수라는 **사람이 고른 고정 기준을 모든 상황에 균일하게**
적용하는 장치다. 같은 γ·같은 규칙이 모델 상태에 따라 도움과 해로 갈린다는 것은
γ 를 더 튜닝해 메울 문제가 아니라 **비용 항의 가중을 상황이 결정해야 한다**는
쪽을 가리킨다. 연구 명제("필요한 정보는 상황마다 다르고 사람이 미리 정하면 안
된다")의 직접적 증거다. 제출 후보로서는 가치 없음(0.6472 vs 제출본 0.6993).

PCS 는 2421.1 → 2403.0 으로 오히려 내려갔다(표준편차 43~46 이라 구분 불가).
장면점수 +0.0144 가 PCS 에서 사라지는 것은 개선이 쉬운 클립에 몰려 있기 때문이다.
실행: `runs/rr441-g0p02-disjointep04-cache-centre-max/`.

### ★ 통합 리더보드 (2026-09-27, `runs/leaderboard-260927/capability_ranking.csv`)

로컬 20 주체 + 공식 참조 8 주체를 **한 번에** 적합했다(경고 없음). ZOIB 는 장면별로 주체를 가로질러
적합하므로 주체 집합이 바뀌면 PCS 값이 움직인다. 순위를 비교할 때는 항상 같은 적합의
값끼리만 본다.

| 순위 | 구간 | 주체 | PCS | 장면점수 | at-fault 사고간거리 |
|---:|:---|---|---:|---:|---:|
| 1 | 1-6 | **axe-v9+rerank-g0.02** | 2700 ±54 | 0.7299 | 1.326 |
| 2 | 1-6 | axe-v9 (제출본) | 2703 ±54 | 0.6993 | 1.298 |
| 3 | 1-8 | merged-ep29-bestmpc | 2641 ±54 | 0.7046 | 1.241 |
| 9 | 6-13 | axe-v9+rerank-g0.1 | 2494 ±44 | 0.7242 | 1.382 |
| 12 | 10-15 | **disjoint-ep04+rerank** | 2403 ±43 | 0.6472 | 1.133 |
| 13 | 9-15 | disjoint-ep29 | 2409 ±45 | 0.6526 | 1.123 |
| 14 | 7-15 | aug-ep29-final | 2465 ±47 | 0.6951 | 1.084 |
| 15 | 9-15 | **disjoint-ep04** | 2421 ±46 | 0.6328 | 1.054 |
| 16 | 16-17 | aug-ep19 | 2248 ±39 | 0.6273 | 1.029 |
| 19 | 18-19 | aug-ep04+rerank | 1943 ±36 | 0.4308 | 0.443 |

구간이 겹치는 1-5 위는 통계적으로 구분되지 않는다. **순위 1 위는 리랭커 γ=0.02** 이지만
이는 구간 상한이 같을 때 at-fault 거리로 동률을 깬 결과이고, PCS 점 추정으로는
axe-v9 이 4 포인트 앞선다(표준편차 54). 공식 지표에서 승리로 주장할 수 없다.

리랭커를 붙인 disjoint-ep04 는 순위가 15 → 12 로 올라가지만 PCS 점 추정은
2421 → 2403 으로 내려간다. 순위 상승도 동률 처리의 결과일 뿐이다.

### stage3 재학습 계열 441 (2026-09-25 ~ 26, 전부 axe-v9 동일 환경·리랭커 없음)

| 지표 | aug-ep04* | aug-ep19 | aug-ep29 | disj-ep04 | disj-ep29 | axe-v9 |
|---|---:|---:|---:|---:|---:|---:|
| 평균 점수 | 0.4308 | 0.6273 | **0.6951** | 0.6328 | 0.6526 | 0.6993 |
| 0점률 | 0.499 | 0.238 | **0.211** | 0.265 | 0.231 | 0.220 |
| GT 오차 m | 2.945 | 1.824 | **1.802** | 2.168 | 1.936 | 1.959 |
| corridor 이탈 | 0.220 | 0.107 | **0.077** | 0.122 | 0.091 | 0.091 |
| 도로 이탈 | 0.200 | 0.075 | 0.079 | 0.088 | 0.088 | 0.089 |
| 전체 충돌 | 0.261 | 0.311 | 0.293 | 0.256 | 0.263 | **0.236** |
| at-fault 충돌 | 0.122 | 0.061 | 0.061 | 0.059 | 0.054 | **0.039** |
| 주행거리 m | 142.7 | 140.1 | 152.4 | 155.4 | 160.4 | **165.2** |
| 완주율 20s | 0.526 | 0.671 | **0.704** | 0.683 | 0.688 | 0.733 |

\* aug-ep04 만 리랭커 γ=0.02 포함(그 시점 계획대로). 나머지는 리랭커 없음.

판정 세 가지.

1. **aug-ep29-final 이 재학습 계열 최고**: 장면점수 0.6951 로 axe-v9(0.6993)과 −0.0042.
   0점률·corridor·GT오차·도로이탈은 제출본보다 낫고, 남은 약점은 **충돌 하나**
   (0.293 vs 0.236, at-fault 0.061 vs 0.039). 이 하나가 PCS 를 10 위로 끌어내린다.
2. **aug 와 disjoint 의 수렴 양상이 다르다**: aug 는 ep04→ep29 에서 +0.26(0.43→0.70),
   disjoint 는 +0.020(0.633→0.653)뿐이다. disjoint 는 ep04 부터 이미 0.63 이라 학습이
   빨리 붙지만 거기서 거의 멈춘다. 30 epoch 예산에서는 aug 쪽이 확실히 유리하다.
3. **증강은 정적 정보만 강화한다**: aug 계열에서 계획편차(0.409→0.197→0.161),
   GT오차, corridor 이탈은 단조 개선되는데 at-fault 충돌은 ep19·ep29 모두 0.061 로
   정체다. 동적 상호작용 학습에는 기여하지 않는다는 신호다.

### ★ 리랭커 441 최종 (2026-09-24)

| 지표 | γ=0.02 | γ=0.1 | axe-v9 |
|---|---:|---:|---:|
| 평균 점수 | **0.7299** | 0.7242 | 0.6993 |
| 0점률 | 0.1905 | **0.1701** | 0.2200 |
| corridor 이탈 | 0.0726 | **0.0635** | 0.0909 |
| 도로 이탈 | 0.0884 | **0.0658** | 0.0886 |
| 전체 충돌 | **0.2245** | 0.2766 | 0.2364 |
| at-fault 충돌 | **0.0363** | 0.0454 | 0.0386 |
| 주행거리 m | **165.4** | 153.6 | 165.2 |
| GT 오차 m | 1.856 | **1.717** | 1.959 |

- **γ=0.02 가 제출 후보**: axe-v9 대비 장면점수 +0.0306, trade-off 없음. 단 PCS 로는
  axe-v9 과 구분되지 않는다(2704 vs 2706, 구간 둘 다 1-6). 공식 지표에서 승리로
  주장할 수 없다.
- γ=0.1 은 route 준수 전문: 0점률·이탈·GT오차 최저지만 충돌 +0.040, 주행거리 −12 m.
  PCS 9 위로 떨어진다.
- 설정: cache=1, centre_dx=1.467, aggregate=max, min_overlap=8.
- 방법론: 무작위 층화 40클립 검증이 정확(+0.0398 예측 vs +0.0306 실측). 실패군 편향
  38클립의 가중 추정은 방향만 맞고 크기가 1/3 로 과소.

### 방법론 주의 (재현 시 반드시 지킬 것)

- **시뮬레이터 비결정성**: 같은 코드·체크포인트에서 워커 16→4 만으로 38클립 중 8개
  판정이 뒤집혔다. 비교는 반드시 같은 워커 수의 짝지은 기준선과 한다.
- **PCS 는 at-fault 사고간 거리에 큰 가중**을 둔다. 장면점수 순위와 역전될 수 있다
  (aug-ep29: 장면점수 4위권 → PCS 10위).
- **부분 실행과 전체 평균을 비교하지 않는다**: `compare_on_clips.py` 가 교집합을 먼저
  잡고 모든 실행을 그 교집합으로 재집계한다.

## 3. 마지막 커밋 이후 바뀐 것

- 실제 GPU operator 오류를 발견했다. 원래 모델의 CUDA dispatch가 기존 FORCE_PYTORCH_MSDA 환경변수를 무시해 unsupported MMCV kernel에서 0 텐서를 받았다. 상수 샘플의 실제 0 / 정답 1 검증으로 점수 오염을 확인했다.
- 별도 policy 어댑터에서 원본 PyTorch fallback을 선택해 axe-v9와 같은 환경을 적용하고, GPU operator 결과 검증을 startup에 추가했다. 원본 모델 폴더/소스/가중치/설정과 gRPC 8/NONE/MPC/정밀도는 불변이다.
- 새 이미지에서 CPU 및 CUDA 전체 5카메라 forward, exact checkpoint, 보정/입력/ego_geom 검증 통과. 원본 전체 SHA256SUMS 통과. 검증 결과 `runs/prepare-stage3-5cam-ep05/*_validation_fixed_operator.json`와 로그.
- 잘못된 33개 완료와 진행 중 결과를 topology-history/20261007-115013-invalid-cuda에 rename하여 보존하고 leaderboard에서 제외했다. 441개를 새 환경으로 재평가한다.
- accelerate.py에 --reuse-services 추가: 정확한 기존 compose/ports를 사용하고 57 native 서비스는 유지해 driver/runtime만 재기동한다. runtime 자체가 평가 집계를 수행하며 종료코드/집계 파일을 확인한 뒤 35 주체 joint fit을 자동 실행한다.
- 사용자 최신 요청대로 VRAM이 높다는 이유만으로 pause하지 않는다. 실제 OOM/평가 오류에만 대응한다. ruff/black 통과. 새 실행의 VRAM/처리량을 확인하고 완료 후 원장과 leaderboard를 갱신한다.

## 4. 다음 단계

- 현재 실행을 중복 기동하지 않고 관찰한다:
  ```bash
  cd /home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim
  tail -n 30 runs/leaderboard-stage3-5cam-ep05-20261007.speed48.progress.log
  tail -n 30 runs/leaderboard-stage3-5cam-ep05-20261007/driver-start.log
  tail -n 30 runs/leaderboard-stage3-5cam-ep05-20261007.speed48.wizard.log
  ```
- 모델의 원본 gate/threshold/BEV/입력 구성을 유지하며 비교한다. 기존 34개에는 과거 MPC·reranker 차이가 있는 주체도 있으므로, 동일 계약의 axe-v9와 짝비교하고 전체35 leaderboard는 각 제출 구성의 비교로 해석한다.
- 현재 48-worker 실행은 중복 시작하지 않는다. unsupported CUDA 결과가 있는 topology-history는 재사용하지 않는다. 수정 후 완료 결과만 보존/재개한다. VRAM이 높다는 이유만으로 중단하지 않고 실제 오류에 대응한다. warm native 서비스를 유지할 때 `accelerate.py --reuse-services`를 사용한다. 과거 `run.sh --resume`는 16-worker용이다.
- 완료 결과: `runs/leaderboard-stage3-5cam-ep05-20261007/aggregate/results-summary.json`; 전체35 fit은 launcher가 자동 실행한다. 후처리만 재실행할 때 `.venv/bin/python e2e_challenge/5cam_eval/leaderboard.py`. 완료 후 EXPERIMENTS.md에 원장 행을 추가하고 HANDOFF/커밋을 갱신한다.

- 이후 비교는 261006 통합 적합을 사용하고, 새 441 결과 추가 시 전체 주체를 다시 적합한다.

1. (TensorRT 를 제출본에 넣기로 하면) TRT 런타임 라이브러리를 이미지에 넣고,
   엔진을 COPY 하고, `trt_backbone` 헬퍼를 `drivesuprim_challenge` 안으로 옮긴다.
   그다음 441 패리티를 확인한다. 더 빠르게 하려면 다음 후보는 BEV 인코더(약 14 ms,
   `FORCE_PYTORCH_MSDA=1` 이라 순수 PyTorch), trajectory head(12 ms), agent head(9 ms)다.
2. 리랭커 결과를 상황 의존 가중으로 확장할지 결정. 지금 결과는 고정 γ 의 한계를
   보여주는 음성 증거이고, 다음 실험은 γ 를 상수가 아니라 driving context 의 함수로
   두는 쪽이 원래 명제를 직접 겨냥한다. γ 를 더 촘촘히 스윕하는 것은 곁가지다.
3. aug-ep29-final 의 충돌만 axe-v9 수준으로 내리면 제출본을 넘는다. 리랭커는 route
   준수 축이라 이 축을 건드리지 않는다 — 충돌 축에 직접 작용하는 항이 필요하다.

## 5. 미결 질문 (사용자 결정 필요)

- 이번 실행의 GPU 범위/속도 우선순위는 사용자가 확정했으며 미결 질문 없음. 실제48-worker VRAM 피크와 처리량, 최종441 점수는 실행에서 확인한다. 아래 연구·제출 결정은 과거 기록.
- gRPC 8/NONE/원본 모델 유지 확정. GPU 호환 오류 수정은 평가 환경 복구이며 원본 모델 변경이 아니다. 이전 33개는 재평가한다. VRAM 근접만으로 중단하지 말라는 조건을 유지한다. 최종 점수/35 주체 순위/실제 처리량은 평가 완료 후 산출한다.
- 원본 설정 유지 여부는 사용자가 확정했다. 별도 임계값 튜닝이나 axe-v9 gate 이식은 진행하지 않는다.

- CSV 사유 추가 요청 미결 없음. 부분 점수의 물리적 원인(저속/정체 등)은 요약 지표만으로 단정하지 않음.

- axe-v9 클립 CSV 요청의 미결 사항 없음.

- 이번 leaderboard 정리 요청의 미결 사항 없음. 아래 항목은 이전 연구·제출 결정 기록이다.

- TensorRT 를 제출 이미지에 넣을지. 모델 추론은 1.34–1.44배 빨라지지만, 공식 환경에
  실시간 마감이 있는지는 모른다. 로컬 시뮬레이터는 동기식이라 점수에 영향이 없다.
  이미지는 약 3.2 GB 늘어난다.

- 리랭커 γ=0.02 를 제출본으로 교체할지. 장면점수 +0.031 이지만 PCS 는 동률
  (2699.6 vs 2703.3, 구간 둘 다 1-6). 공식 지표로는 교체 근거가 없다.
- aug-ep29-final 을 더 학습할지(충돌 축은 정체, 나머지는 계속 개선 중이었다).
- disjoint 계열을 계속 볼지. ep04→ep29 개선폭이 aug 의 1/13 이고, 리랭커를 붙여도
  0.6472 로 제출본에 한참 못 미친다.
- corridor 필터 후속: 하드 게이트 대신 감점(demotion) 방식으로 갈지.

## 6. 고정 경로

- 체크포인트: `../models/` (ep29: `../models/260923_epoch=29-step=30330.ckpt`, 이미지 `alpasim-e2e-axe-v9:260923-ep29-step30330`).
- 데이터셋: `data/nre-artifacts -> /home/kaist5/Dataset/alpasim/data/nre-artifacts` (심볼릭 링크, 지우지 말 것).
- 렌더러 공유 캐시: `.cache/renderer-shared/` (비면 `warm_renderer_cache.sh` 가 자동으로 채움).
- 10클립 목록: `e2e_challenge/route_cache_filter/clips10.txt` (선정 근거 `clips10.md`).

## 7. 다른 서버에서 재구성 (git 으로 오지 않는 것)

| 항목 | 이 서버 위치 | 옮기는 방법 |
|---|---|---|
| 데이터셋 (NuRec 아티팩트, 장면) | `data/nre-artifacts -> /home/kaist5/Dataset/alpasim/data/nre-artifacts` | 대상 서버의 데이터 경로로 심볼릭 링크 재생성 |
| 도커 이미지 | `alpasim-base:0.89.0`, `nvcr.io/nvidia/nre/nre-ga:26.04`, 드라이버 이미지 `alpasim-e2e-*` | 베이스는 pull, 드라이버 이미지는 `e2e_challenge/sample_submission_drivesuprim` 의 Dockerfile 로 재빌드(체크포인트는 ../models 에서) |
| 체크포인트 | `../models/*.ckpt` | rsync |
| 렌더러 공유 캐시 2.7 GB | `.cache/renderer-shared/` | rsync 하거나 첫 실행 때 `warm_renderer_cache.sh` 가 자동 생성 |
| host venv | `.venv/` (uv) | `uv sync` (`UV_OFFLINE` 은 캐시가 생긴 뒤부터) |
| 실행 결과 | `runs/<run>/aggregate/results-summary.json` | 요약 JSON 만 rsync 하면 충분 |

git 으로 오는 것: 런처·드라이버 패키지·오버레이 코드·문서·실험 원장.

### GitHub 사본(`mine`)과 Git LFS

NVlabs 원본은 테스트 픽스처(usdz/asl/ply 등 20개, 약 225 MB)를 LFS 로 관리한다. GitHub 는
LFS 포인터가 가리키는 객체가 없는 push 를 거부하므로(GH008), 개인 사본에는 LFS 객체까지
올려 두었다(`git lfs fetch --all origin` 으로 NVlabs 에서 받은 뒤 `git lfs push --all mine`).
무료 LFS 용량 1 GB 중 약 225 MB 를 쓴다.

- 다른 서버에서 받을 때 대역폭을 아끼려면: `GIT_LFS_SKIP_SMUDGE=1 git clone -b junseong/e2e-eval git@github-junseong:JunSeongKW/alpasim-e2e-eval.git alpasim`
  (평가에는 픽스처가 필요 없다. pytest 를 돌릴 때만 `git lfs pull`.)
- 이후 커밋에는 LFS 파일이 없으므로 종료 루틴의 `git push mine` 그대로 된다.
