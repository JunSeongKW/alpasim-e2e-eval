# HANDOFF — 이 파일 하나로 다음 에이전트가 이어받는다

마지막 갱신: 2026-10-13 15:51 KST (Claude Code)

에이전트(Claude Code, Codex 등)는 세션을 **시작할 때 이 파일과 `git log -10` 을 읽고**,
**끝낼 때 이 파일을 갱신하고 커밋**한다. 대화 원문은 옮기지 않는다. 규칙은 `AGENTS.md`.

## 1. 실행 중인 작업

- **추론 시간 비교 대기열 (2026-10-13 15:50~, `runs/g112-trt-latency-20261013.progress.log`, PID 611643).** axe-v9 / g112 PyTorch / g112 TensorRT 를
  공식 ec2 preset(16 드라이버 × 동시 2, 렌더러 GPU 0–3)으로 같은 첫 32장면씩(슬롯 32개를 한 번 채우는 최소, 사용자 요청) 순서대로, 이어서 모델 단독 벤치(GPU 4–6). 사용자 요청("gpu 비워지면 바로 진행").
  시작 시점 GPU 0–7 은 다른 연구원 RL 작업(axe-rlfail)이 사용 중 → 8장이 모두 4 GB 미만이 될 때까지 대기. 표: `runs/ec2-timing-summary-20261013.md`.
  441 동등성 평가는 사용자 지시로 보류("평가는 하지 말고 기다려줘").
- **g112 TensorRT 제출 이미지 빌드·로컬 검증 완료, push·submit 안 함.** 런북 `e2e_challenge/g112_trt/SUBMISSION_RUNBOOK_G112_TRT.md`.
  이미지 `alpasim-e2e-drivesuprim-stage3:g112-kv28-trt` = `sha256:2b0c5e0067d6…`(11.0 GB). 검증 로그 `runs/g112-trt-build/validate.progress.log`.
  TRT 허가 문의 NVlabs/alpasim#205 는 10-13 기준 답변 없음.
- img768raw_stage3_ep04 441 평가 완료(2026-10-13 11:31–14:23, RPC 실패 1클립 재시도 14:33 → 441·인프라 실패 0). 결과 2절.
- g112·768 재평가 완료(2026-10-12 23:13), GPU 반납. 결과는 2절.
- 대회 규정(ec2, 16 드라이버 × 동시 2 = 32 병렬) 실행 시간 측정 **사용자 요청으로 중단(2026-10-12 09:2x), GPU 0–7 반납, 컨테이너 0.**
  axe-v9 441 전체 측정은 중단 시점까지 `runs/ec2-timing-axe-v9-20261012/`(완료 롤아웃은 남음, 집계 없음). 나머지 모델 측정은 시작 전.
  재개: `nohup e2e_challenge/pkg3cam_eval/ec2_timing_queue.sh 20261012 128 runs/latency-bench-20261012_0839 >> runs/ec2-timing-queue-20261012.progress.log 2>&1 &`
  (모델별 첫 128 장면, 441 시간은 추정). axe-v9 전체 441 기준값이 필요하면 `run_ec2_timing.py --model axe-v9 --date <새 날짜> --wait`.
  모델 단독 지연 6개는 완료: `runs/latency-bench-20261012_0839/summary.json`.
- 3카메라 번들 2개 441 평가 완료(2026-10-11 15:32–21:13): g112_kv28(40 워커) → vidarrender(48 워커, RPC 실패 1클립 `retry_infra.py` 로 재시도, 440 클립 불변 확인)
  → 41주체 적합 `runs/leaderboard-41-with-pkg3cam-20261010/local_leaderboard.csv`. GPU 0–7 반납. DrivoR 는 사용자 승낙 전까지 일시정지 유지.
- DrivoR 441×3 평가 완료(2026-10-12 06:16): 두 모델 1323 롤아웃, RPC 실패 3건(ours 2, original 1)은 finalize 의 가벼운 재개로 재시도해 0건.
  적합 `runs/leaderboard-drivor-8ref-20261009/`, 세 방식 `runs/drivor-protocols-20261009/protocols.md`, 통합 표 `runs/drivor-combined-alpasim441-20261009/combined_alpasim441.md`. GPU 0–7 반납.
- 휴지통 삭제 완료(2026-10-10 05:03): 대형 파일 1,587개 포함 `runs/.trash-cleanup-20261009` 전부 삭제, runs/ 581 GB → 20 GB. 완료 시점 디스크 여유 769 GB.
- GPU 사용 승인: 이 평가에 한해 0–7 전부(2026-10-09 15:05 "8장 전부, VRAM 최대").
- axe-v9 441 재실행 완료(2026-10-11 05:03–09:08, `runs/curve-axe-v9-441-20261010`, 드라이버 컨테이너 정리됨). junhyeok RL(axe-rlfreq1)은 10-11 04:54 경 종료.
  09:08 이후 GPU 0–7 모두 비어 있음.

## 2. 최근 결과

### vits512_g112_kv28 TensorRT 제출 이미지 (2026-10-13, `e2e_challenge/g112_trt/`)

- axe-v9 이미지 위에 g112 441 평가 때 마운트한 것(navsim, nurec_pf, 어댑터, 체크포인트, 설정)을 그대로 굽고,
  ViT 이미지 백본만 TensorRT 10.13 FP16 엔진으로 교체(엔진 exact=H100, ampere_plus=cc≥8.0). 드라이버 기동 시 공식 GPU 위에서
  PyTorch 백본과 비교해 상대 오차 ≤5% 일 때만 교체, 그 외 모든 실패는 PyTorch 로 기동. 호출 중 예외 시 그 프로세스는 PyTorch 로 계속.
- 엔진: 백본 1회 7.27 → 0.99 ms(exact) / 1.09 ms(ampere_plus), 상대 오차 1.1%. 모델 전체 68.2 ms(PyTorch) → 약 51 ms 추정.
- `validate.sh` [1]–[8] 통과: axe-v9 레이어·실행 설정·환경변수 67개 유지, 코드 422파일이 441 평가 마운트와 동일(policy.py 4줄 hook만 추가),
  공식 환경만으로 24 s 로드·`active engine=exact`·get_version 응답·VRAM 4.1 GB, ampere_plus 엔진 동작, fallback 4종(엔진 없음/off/TRT 없음/검증 거부) 모두 PyTorch 로 기동.
- 첫 빌드는 `libnvonnxparser.so.10` 을 빌드 전용으로 잘못 보고 빼서 이미지 안에서 TRT import 가 실패(→ PyTorch 로 기동)했다. 넣고 재빌드해 해결.
- `validate.sh` [9](프로세스 내): TRT 가 실제로 쓰이고 PyTorch 와 같은 후보 선택·궤적 차 0.0 m, TRT 예외 시 PyTorch 전환 정상. 단 "다른 shape 입력" 검사가
  PyTorch 두 번 호출 결과의 비트 불일치로 실패(테스트를 상대 오차 1e-3 비교로 수정, 재실행 전). 지연(198.8 vs 204.8 ms)은 다른 연구원 RL 작업(axe-rlfail, GPU 0–7)과
  GPU 를 나눠 쓴 측정이라 무효. GPU 가 비면 `GPU=4 validate.sh` 재실행.
- 미검증: 이 이미지로 441 폐루프 동등성(`run_parity.sh`, 사용자 지시로 보류), 공식 ec2 토폴로지(렌더러가 GPU 0–3 필요), H100 외 GPU 실측.

### img768raw_stage3_ep04 441 최종 (2026-10-13, 441·인프라 실패 0)

- 장면 0.591, 0점 32.4%, 책임 충돌 8.2%, 도로 이탈 12.2%, corridor 13.4%, 책임 사고당 km 0.74.
- − vits768(두 실행 평균) −0.052 [−0.090, −0.015], − g112 평균 −0.040 [−0.077, −0.003], − axe-v9 −0.110 [−0.155, −0.064].
  환경은 vits768 과 같고 가중치만 다르다(epoch 4 vs 9). 덜 학습된 체크포인트가 낮다는 확인이며 논문 기여는 아니다.

### vits768 vs vits512_g112 재평가 (2026-10-12, 각 441×1 두 번, 40 워커)

- 768: .6427 → .6439, g112: .6290 → .6333 (같은 모델 두 실행 차 ≤ .0044, 클립 7–8% 통과/실패 뒤바뀜).
- 768 − g112: 1차 +.014 [−.024, +.051], 2차 +.011 [−.028, +.048], 두 실행 평균 +.012 [−.021, +.046] → 유의차 없음(개루프 PDM 차 .0035 도 작음).
- 책임 사고당 km: 768 1.04/0.96, g112 1.03/1.09. 도로 이탈은 g112 가 낮음(7.3/7.0 vs 8.2/9.3).

### 같은 441·플래너만 다른 공정 비교 (2026-10-12, `runs/drivor-literature-20261012/fair_441.md`)

- 남는 주체: ours, DrivoR original, VaVAM(비선형), Alpamayo 1 (모두 dev·441×3·기본 비선형 MPC). VaVAM 선형은 참고, alternative_*·논문 행·axe 계열은 제외(이유는 문서).
- ours − VaVAM: 장면 +22.5 [+18.8, +26.3], 책임 충돌 −6.9; ours − Alpamayo 1: 장면 −21.5, 진행률 −29.4. 강제 GT 중 drive() 호출 차이(181 vs 198 회)는 DrivoR 가 무상태라 영향 없음.

### axe 계열 6개 + 기준 8개 PCS (2026-10-12, `runs/leaderboard-axe6-8ref-20261012/table.md`)

- 14 주체만 적합(alpamayo1=2000, alternative_2=600): axe-v9 2391 (1–2위) > vits768 2327 > g112 2289 > 6layer 2235 ≈ vitl 2260 > vidar 2199 > alpamayo1 2000.
  axe-v9 는 인프라 실패 없는 재실행(0.7006)을 사용(채점 실행 0.6993 은 RPC 실패 1클립으로 적합 불가). 41주체 적합과는 PCS 척도가 다르다.
- 지연: 모델 단독(배치 1, H100) 만 모델 비교에 쓴다. dev 실행 중 Drive 평균(355–816 ms)은 렌더러와 GPU 공유·워커 수 차이가 지배. ec2 저부하(4 롤아웃) axe-v9 Drive 169 ms(p99 331), 32병렬 실측은 중단으로 없음.

### AlpaSim 문헌 비교: 평가 기준별 표 (2026-10-12, `runs/drivor-literature-20261012/tables.md`)

- 441 장면을 쓴 논문은 OPTED·ECO뿐. 나머지(GroundAct 913 public_2601, Qwen-Drive 916, Alpamayo-R1·RoaD·Qwen-RobotNav 920, COMPACT-VA 910, 부분집합들)는 장면 세트가 다르다. 원본 DrivoR를 AlpaSim에서 평가한 논문 없음(NavSafe-∞ 는 Isaac Sim, DrivoR DS 64.7).
- 우리 두 모델을 7개 정의로 재채점: 대회·AlpaSim 기본 집계·Alpamayo-R1·RoaD(측면 포함/제외)·충돌 유형별(OmniDreams)·원지표(FlashDrive). 학습 범주 Z/P/C 와 공정 비교 참고 사항(8절) 포함.
- 제로샷·기본 MPC·441 끼리: ours 33.4 > DrivoR 29.7 > DiffusionDrive 28.9 > LTFv6 26.5 > VaVAM 11.0 (DiffusionDrive+ECO 35.2).

### DrivoR 2종 441×3 최종 (2026-10-12, 기준 8주체와 적합)

| 모델 | PCS(구간) | 장면 점수 | 책임충돌 | 도로이탈 | 4 m 이탈 | 진행률 | 책임사고당 km |
|---|---|---|---|---|---|---|---|
| ours (DrivoR+Z_safe) | 1869 (2–4) | 33.4 | 11.1 | 18.1 | 23.4 | 46.0 | 0.24 |
| original (DrivoR V1) | 1859 (2–4) | 29.7 | 16.3 | 19.9 | 25.7 | 45.5 | 0.19 |
| Alpamayo 1 (기준) | 2000 | 55.0 | 5.5 | 10.9 | 25.2 | 75.5 | 0.91 |
| VaVAM 선형 MPC (기준) | 1840 (2–4) | 31.4 | 16.4 | 15.2 | 26.6 | 59.6 | 0.35 |

- 짝비교 441 장면(ours−original): 장면 점수 +3.7 [+1.2, +6.3], 책임충돌 −5.1%p [−8.0, −2.3], 도로이탈 −1.7 [−3.5, −0.2], 4 m 이탈 −2.3 [−4.8, +0.4]. PCS 는 구간 겹침(2–4).
- 세 채점 방식 모두 순서 동일(RoaD: 0.24 vs 0.20 km/사고). 대회 방식 재계산은 공식 집계와 1323×2 롤아웃 완전 일치.

### 10-10 3카메라 번들 2개 (41주체 적합, `runs/leaderboard-41-with-pkg3cam-20261010/`)

| 모델 | 순위(구간) | PCS | 장면 점수 | 책임사고당 km | 책임충돌 | 도로이탈 | 차로이탈 |
|---|---|---|---|---|---|---|---|
| stage3-vits512-g112-kv28 | 21 (13–25) | 2400 | .6290 | 1.03 | 8.2% | 7.3% | 12.5% |
| stage3-vits512-vidarrender | 28 (23–29) | 2276 | .5461 | 0.69 | 10.2% | 11.8% | 16.3% |
| (참고) axe-v9 | 2 (1–7) | 2703 | .6993 | 1.30 | 3.9% | 8.9% | 9.1% |

- 짝비교(장면 점수): g112 은 6layer·disjoint-ep04·vits768 과 차이 없음(CI 0 포함), axe-v9 대비 −.070 [−.114, −.027]. vidarrender 는 g112 대비 −.083, disjoint-ep04 대비 −.054 [−.093, −.014] 로 유의하게 낮음.
- 커브 근정지(사람 이동 중): g112 36% / vidarrender 39% / axe-v9 27% (직선 8/9/6%). 커브 안 통과 속도 사람 대비 g112 .57, vidarrender .54, axe-v9 .93 (`curve_speed_profile.png`).

### 커브에서 사람보다 먼저 거의 서는가 (2026-10-10, `e2e_challenge/axe_local_eval/curve_stop_vs_human.py`)

- 기준: 사람 궤적(usdz rig)과 **같은 도로 위치**에서 비교. 커브 구간 = 15 m 창 최대 헤딩 변화 창 + 진입 20 m, 커브 |yaw|≥30°(83), 직선 <15°(324, 같은 규칙의 구간).
  근정지 = ego <1 m/s 1초 이상, 그 위치 ±2 m 에서 사람은 줄곧 ≥3 m/s. 첫 충돌 이후는 제외.
- 프레임별 기록이 있는 axe 계열 4개(vits512-6layer, vitl512, vits768, disjoint-ep04): 커브 구간 근정지 33–40%(사람보다 ≤3 s 늦게 도착한 경우만 28–34%) vs 직선 5–10%.
  커브 구간 통과 속도 중앙값이 사람의 0.66–0.72배(직선 0.99–1.02). 표시된 근정지는 사람이 그 자리를 중앙값 4.7 m/s 로 지나간 곳, 도착 지연 중앙값 0.6 s, 정지 2.1 s, 26% 는 끝까지 다시 출발 못 함.
- **axe-v9 (재실행 `runs/curve-axe-v9-441-20261010`, 결과 `runs/curve-stop-vs-human-20261010/summary.md`)**: 커브 구간 근정지 27%(≤3 s 늦게 도착 20%) vs 직선 6%(6%),
  커브 통과 속도 사람의 0.74배(직선 0.99). 같은 문제가 있으나 변형 4개보다 약하다.
  재현 대조: 평균 점수 .7000 vs 채점 실행 .7009, 440 클립 중 점수 동일 306, |Δ|<0.05 375, 합격/불합격 뒤바뀜 38 (실행 간 비결정성; 집계는 일치).
- **g112_kv28 (10-11, `runs/curve-stop-vs-human-20261011/`)**: 커브 구간 근정지 36%(≤3 s 늦게 도착 30%) vs 직선 8% (Fisher p=2e-9), 커브 통과 속도 사람의 0.63배(전 모델 중 최저).
  axe-v9 대비 같은 커브 클립 짝비교 근정지 19 vs 11건(McNemar p=0.20), 속도비 0.61 vs 0.75(Wilcoxon p=0.16) → 더 나빠 보이나 유의하지 않음. disjoint-ep04·6layer 와는 차이 없음.
- 채점 실행 요약(`runs/leaderboard-merged-route-ep30`): 무사고 완주 롤아웃의 주행거리/사람 중앙값 커브 0.79 vs 직선 0.98, 점수 .432 vs .788, 도로이탈 14% vs 6%, 차로 이탈 20% vs 6%.

### DrivoR 2종(mingyu 번들) curated_val 441 평가 준비 (2026-10-09, `e2e_challenge/drivor_eval/README.md`)

- 번들 상태 `MODEL_INFERENCE_READY_ADAPTER_IMPLEMENTATION_PENDING` → PAI gRPC 어댑터를 구현해 **라이브 스모크까지 통과**.
  번들 해시 검증 PASS, CPU FP32 골든 일치(2모델), H100 FP16 은 mingyu CUDA 골든과 선택 인덱스 동일·후보 ≤0.025 m(=mingyu 자신의 CUDA↔CPU 차).
- 라이브 스모크(장면 2·롤아웃 1, FP32): 두 모델 모두 추론 398회 오류 0, 인프라 실패 0. 000525f6: ours .953 / original .890, 0026708c: 둘 다 0(5cam 때도 후방추돌난 클립).
  롤아웃 1개 5.8분(2 worker), drive 1회 전처리 ~410 ms + 모델 ~68 ms, 드라이버 GPU 1.4–1.7 GB.
- 리플레이 적합성(ASL 5개 × 6구성, 구성당 ~1,000 drive, `runs/drivor-conformance-20261009/matrix_summary.txt`):
  FP16·입력 그대로 → 모델당 1회 `Nonfinite scores`(시뮬 첫 스텝 가속도 −21.6 m/s² 에서 64후보 충돌 logit ≈ −36 → FP16 sigmoid underflow → 전부 −∞).
  FP32 는 오류 0, FP16 대비 4 s 지점 차 중앙값 0.01 m(p95 0.8/1.4 m). NAVSIM 범위 clamp 는 오류 0 이지만 28% 스텝을 바꾸고 4 s 지점 p95 9–13 m 이동.
  본 평가 첫 판에서 FP32 도 실패(clipgt-032b6f21 첫 상태 가속도 −57.6, 직전 +68.8 → 충돌 logit −147). 롤아웃 44개 중 16개가 처음 몇 상태에 |a|>10(최대 −761), 그 뒤 |ax| ≤ 9.0
  (`runs/drivor-conformance-20261009/start_spikes_44rollouts.jsonl`) → 가속도만 ±10 m/s² 경계(`--ego-clamp physical`). 두 실패 입력 모두 FP16/FP32·두 모델에서 유한, 충돌 logit −3.9/+2.6 으로 정상화.
- 명령 규칙 보정(NAVSIM trainval 20.7만 프레임, `command_calibration.json`): 공식 3 m/20 m(실효 40 m) 균형정확도 .884, 가능 최선 .889.
- 후방 B0: 회전만 접합하면 바로 뒤 추종차가 반투명/소실 → 지면+15 m 원통 프록시 접합 채택(`runs/drivor-conformance-20261009/rear_b0_stitch_compare.png`).

### 4개 모델 연산 시간 (2026-10-09, `runs/latency-bench-20261009_1346/REPORT.md`)

- 단독 측정(GPU 4–7 각 1모델, `_run_agent` batch1, 200회): axe-v9 59.6 ms / vits512_6layer 62.3 / vits768_3layer 60.7 / vitl512 149.1 ms (2.5배, 6.7 FPS, 백본 111 ms, 337M, 3.3 GB). ViT-S 3개는 사실상 동일.
- 441 평가 중 Drive 호출 평균(356/683/603/816 ms)은 부하·디스크 차이가 지배적이라 모델 비교에 쓰지 않는다.

### ★ 10-08 3카메라 번들 3개 최종 (39주체 적합 `runs/leaderboard-39-with-pkg3cam-20261008/`, CSV `e2e_challenge/axe_local_eval/data/local_leaderboard_261008_pkg3cam.csv`)

| 주체 | Rank | PCS | 장면점수 | at-fault 거리 | 구간 | 0점률 | 책임충돌 |
|---|---:|---:|---:|---:|:---:|---:|---:|
| axe-v9 | 2 | 2702 | 0.6993 | 1.2982 | 1-7 | .220 | .039 |
| axe-v10 | 10 | 2536 | 0.7180 | 1.2563 | 5-15 | | |
| stage3-vits768-3layer | 16 | 2447 | 0.6427 | 1.0442 | 10-22 | .277 | .070 |
| vits512-disjoint-baseline-ep04 | 22 | 2361 | 0.6001 | 0.9763 | 15-26 | .315 | .070 |
| stage3-vits512-6layer | 23 | 2357 | 0.6154 | 0.9308 | 16-26 | .281 | .073 |
| stage3-vitl512-3layer-lr1 | 24 | 2346 | 0.5872 | 0.8832 | 16-26 | .324 | .102 |

- 세 번들 모두 441/441·인프라 오류 0. 드라이버 로그 FATAL 0·NaN 0·예측 83,496 회씩.
- 해상도 768 이 셋 중 최고, BEV 6층은 3층(disjoint)과 PCS 동률, ViT-L 은 책임충돌 .102 로 가장 나쁨. 셋 다 axe-v9 대비 책임충돌 1.8–2.6배가 격차의 주원인.
- 시나리오 분석(axe-v9 × 차종·국가): `../scenario_analysis/analysis_result_vehicle_country/` (팀 레포 쪽, 커밋 안 함).

### 10-08 3카메라 번들 441 (같은 38주체 적합 `runs/leaderboard-38-with-pkg3cam-20261008/`)

| 주체 | Rank | PCS | 장면점수 | at-fault 거리 | 구간 |
|---|---:|---:|---:|---:|:---:|
| axe-v9 | 2 | 2703 | 0.6993 | 1.2982 | 1-7 |
| axe-v10 | 10 | 2541 | 0.7180 | 1.2563 | 5-15 |
| stage3-vits768-3layer | 15 | 2446 | 0.6427 | 1.0442 | 10-21 |
| vits512-disjoint-baseline-ep04 | 22 | 2360 | 0.6001 | 0.9763 | 15-25 |
| stage3-vits512-6layer | 23 | 2359 | 0.6154 | 0.9308 | 16-25 |

- 768 은 441/441·인프라 0, VRAM 피크 78.3 GB. 6layer 는 441/441·인프라 0. 둘 다 axe-v9 대비 책임 충돌이 ~1.8배(0.070–0.073 vs 0.039).

### 동일441클립의3모델최종비교 (2026-10-07 19:56)

- axe-v9 / 신규disjoint3cam / 직전5cam 순 mean.6993/.6001/.4316, atfault거리1.2982/.9763/.4516km. 같은36fit PCS2704/2353/1896, Rank2/21/26,95%구간1–7/15–23/26–27. 36주체441적합경고·제외0,rawrank CSV와mean/거리/7열정규CSV일치검증통과.
- perfect239/211/67,zero97/139/126,partial105/91/248,후방87/69/150,책임17/31/37,도로39/39/51,rawGT4m corridor40/72/48. 신규0점단일사유31충돌+39offroad+69corridor=139;rawcorridor72와구분한다. 충돌·이탈raw는중복가능하며axe-v9역사적RPC1은원래0점/충돌미상으로보존한다.
- 신규mean은axe-v9보다−.0993/5cam보다+.1685. 같은441점수pair신규vsaxe-v9 개선82/악화132/동률227,vs5cam 개선238/악화82/동률121. 후방기록은87→69(−18),5cam150→69(−81),새책임충돌·GT이탈은v9보다많다. 후방회피의도나카메라만의효과를단정하지않는다.
- `final_result_audit.json`에최종441/원439보존/신규RPC0/36fit/같은441/점수거리일치검증. comparison CSV/JSON/paired441 CSV와tracked36CSV를작성하고EXPERIMENTS.md에완료행append했다. 재시도클립은1점과실제GT corridor0점으로확정,초기인프라0점두행을보존후대체했다.


- 19:46 KST 정상439개의중간결과: disjoint512 mean.6005/at-fault-distance.9750km/perfect210(47.84%)/zero138(31.44%)/partial91(20.73%)/rear69(15.72%)/atfault31(7.06%)/offroad39(8.88%)/corridor68(15.49%). 같은439axe-v9 mean.6982/거리1.2971/후방86(19.59%), 신규점수 개선82/악화131/동률226. 모델후방충돌은줄었으나책임충돌17→31·corridor40→68으로평균점수는낮다. PCS/순위는인프라2복구후최종36fit값을보고한다.

- 신규disjoint512 종료시각문의(17:29): 완료0이며48개진행중이므로 확정성능점수는 아직없다. 첫묶음각worker의실제Session/InitialStep/StepEvent에서 진행률·walltime·최근5분속도로ETA산출, 사용자예상20:10전후/19:45–20:30(리더보드약5분포함). 모든GPU100% 및프로세스정상확인, 중단/설정변경없음.

### 같은439클립의3개모델중간비교 (2026-10-07 19:50)

- 비교UUID집합을 comparison_439_clip_ids.txt에저장하고3모델각439행/동일UUID를검증했다. CSV `runs/leaderboard-vits512-disjoint-baseline-ep04-20261007/three_model_comparison_439.csv`. axe-v9 / 신규disjoint3cam / 직전5cam 순 mean .6982/.6005/.4290, atfault거리1.2971/.9750/.4506, rear86/69/150.
- perfect238/210/65,zero97/138/126,partial104/91/248,atfault17/31/37,offroad39/39/51, raw GT4m corridor40/71/48. 원인단일분류corridor68/39와raw71/48을혼용하지않고지표별중복가능을명시한다. axe-v9역사적RPC1의0점은원점수로보존하며그클립의충돌지표미상을명시한다.
- 신규모델은5cam보다mean+.1716/rear81개감소,axe-v9보다mean−.0976/rear17개감소이나책임충돌/GT이탈은더많다. camera만의인과효과로해석하지않는다. PCS/순위는2클립복구후441 최종36fit에서제공한다.

### 5카메라 원본 모델 최종 441 결과 (2026-10-07)

- 기존34+신규1의같은441로jointZOIB적합,경고/제외없음. 새모델Rank25/PCS1894/구간25–26/평균.4316/at-fault.4516. axe-v9같은적합Rank2/PCS2721/구간1–7/평균.6993/at-fault1.2982. 과거별도fit PCS와혼용하지않는다.
- axe-v9→5cam: 만점239→67/0점97→126/부분105→248, rear87→150(19.7%→34.0%). scene score개선69/악화278/동률94. 새모델0점의원인분류 collision_at_fault37/offroad50/corridor39;인프라오류0. raw metric은offroad51/corridor48로중복이있으므로원인분류와구분한다.
- 전체441 분석접근210/관찰가능176/가속+1m/s27(15.3%),front10m없음152중22/있음24중5. rear미상은baselineRPC1이며무충돌로간주하지않는다. 기존rear28개없어짐중다른hard실패없음9,새rear90/둘다59/둘다없음263/미상1. 이모델은전체rear감소를보이지않고,후방카메라만의인과효과를단정하지않는다.
- 결과정규CSV e2e_challenge/axe_local_eval/data/local_leaderboard_261007.csv(35행,원래7열/정수rank구간). 전441 CSV·통계·그래프 /home/kaist5/data/junseong/stage3_5cam_rear_response_20261007/{all_441_paired_results.csv,population_summary.json,01_rear_collision_comparison.png,02_approach_speed_change.png,REPORT.txt}. 대표6영상은14:46완료.
- final_result_audit.json에441/35행/점수/manifest검증및완료시각을기록,EXPERIMENTS.md에새완료실험행을append했다. 현재image/source체크섬보존과실제CPU/CUDAforward검증은초기준비보고서에서확인된상태를유지하며,최종결과는새validphase441만사용한다.

### 5cam 중간 결과 (2026-10-07 13:48 스냅샷)

- 완료334개만metrics.parquet를별도inputs에링크해동일공식aggregation으로채점했다. 원run/ASL/영상IO/설정은변경없다. 결과 runs/interim-stage3-5cam-ep05-20261007/20261007-134838/paired_interim_summary.json 및 paired_interim_clips.csv.
- 같은334개 axe-v9→5cam: 평균점수.6917557436→.4091936621, 만점175→46,0점77→100,부분82→188,후방충돌64→116,at-fault거리1.2763008551→.4343266637km. 신규동일클립점수 개선52/악화212/동률70.
- 후방전이: 둘다46/기존만18/신규만69/둘다없음200/기존RPC결측1. 기존만18중다른hard실패없이회피5개. 기존RPC1은무충돌로오인하지않는다. 현재후방충돌감소경향확인되지않으며속도/영상인과판정은아직미실행이다.
- 거리계산은원aggregation의sum(dist_traveled_m)/sum(offroad_or_collision_at_fault)/1000과일치한다. 공식신규334및기존전체441의결과와수치일치검증완료. PCS/순위는441완료후35주체jointfit결과로보고한다.

### 후방 접근 시 가속 반응 분석 준비 (2026-10-07)

- 사용자 목적은 후방 카메라 추가 후 후방 접근에 자차가 가속해 충돌을 방지하는 경향을 보는 것이다. 같은441개의 rear collision 전이/다른hard실패/점수/progress CSV 및 발생률 그래프를 만든다. 신규441 ASL의 차량 pose로 rear 접근·gap·closing speed·TTC 및 가속+1m/s 반응, 앞차10m, 기록GT 속도변화 대비 차이를 계산한다. 초기GT/충돌·이탈·종료 censor를 적용한다.
- axe-v9 원본441 run은 aggregate/results-summary.json만 남아 있다. 87 rear collision과 self-pair353무충돌/87충돌/1RPC결측을 확인했다. physical failure_reason은 정상 관측으로 남기고 RPC/결측만 비교 불가로 분리한다.
- 대표클립은 안전하게회피/계속충돌/새충돌에서 UUID순최대2개씩, 부족분은접근클립으로채워최대6개. GPU해제후4/5가비면 같은immutable axe-v9 image85134/ckpt364c8/dev/MPC1/.25/3로6개만재실행한다. 원본점수와재실행차이를공개하고원본리더보드를덮어쓰지않는다. 별도run rear-response-axe-v9-replay-20261007/driverprefixaxe-rear-response-v9/ports7260~7261,22000~.
- 각사례는 두모델의 전방/후방좌/후방우 비교MP4(후방영상은v9에서는모델입력이아님표기), 속도·후방거리·상대속도 PNG, 시계열CSV를 만든다. 전체441 통계와 선택사례 비율을 구분한다. 모델가중치/BEV등차이로 카메라 인과효과나 의도는 주장하지 않는다.
- 결과폴더 /home/kaist5/data/junseong/stage3_5cam_rear_response_20261007. 분석요청/측정정의는 analysis_request.json. 실제ASL pose parser와 ffmpeg/font를 확인했고7tests/ruff/black검증성공, 실제MP4인코딩테스트포함. 본분석과큰영상IO는원평가완료후에만한다.

- 2026-10-07 평가 진행 상황 요청: 현재 유효한 441 재평가의 완료 표지만 집계했다. 격리된 CUDA 오류 결과는 포함하지 않았다. 최근 30–60분 처리량과 클립당 평균 약 907초를 기준으로 종료 시각을 추정했다. 실행 설정 변경 없이 GPU 0–7 / 48-worker 평가를 유지한다.

### nurec2img 렌더 제외 목록과 441 평가 대조 (2026-10-07)

- nurec-302c5c99-2762-4f5d-950e-1122f3e7c20a는 curated_val 441과 현재 generated-user-config에 포함된다. 기존 axe-v9 정상 평가 점수0.2707195643440281/failure_reason null. 현재5cam에서는 USDZ의rear L1/R1 캘리브레이션과 촬영시각을 확인했고 실시간 render_rgb→driver.submit_image로 입력한다. 오프라인nurec2img PNG/학습제외목록을 읽지 않으며441을440으로 줄이지 않는다.
- 8/25 전방3cam 재렌더 요청은 준비된1603개 로그 전체이며18개 후진도 포함한다. 그보다 앞선준비 단계에서 원시1607개 중 route0건3개 및route84%미매칭1개를 제외했다(EPDMS.md5.5).
- 9/18 후방70fov/virtual112 추가 작업은1603+26.01고유757에서 학습후진18개를 더 제외한2342개를 사용했다. 8/28에 학습후진기준을강화했다는policy기록도존재하지만 전방요청은기존목록을유지했다. 실제후방완료2342/실패0.
- 확인된두단계 제외22개(경로정보4+후진18)를 `e2e_challenge/axe_local_eval/data/nurec_render_exclusions_20261007.csv`에 정리했다. 각ID/사유/단계/전방요청포함/후방대상포함/441포함/출처를기록했다. 그22개 중441과겹치는것은302c5c99 하나이다. 감사JSON `runs/prepare-stage3-5cam-ep05/nurec_render_exclusions_audit.json`.
- 학습제외54개와렌더제외22개는동일목록이아니다. 학습제외manual_ddc_review30/ghost_annotation3은 후방렌더target에포함되고실제완료됐다. 3개route_geometry_invalid는기존1603요청밖이었다.

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

- `g112_trt/latency_compare.sh`: 세 모델 공식 preset 추론 시간 + 모델 단독 지연 대기열. `g112_trt/register_trt_model.py`: TRT 이미지를
  run_pkg3cam 모델(baked)로 등록(run_parity.sh 도 사용). `run_ec2_timing.py`: 441 시간 추정이 짧은 실행에서 실패해도 지연은 저장, `--table --models`.
  `run_bench_models.py`: TRT 모델 추가. `test_trt_image.py`: 다른 shape 검사를 상대 오차 비교로.

- `e2e_challenge/g112_trt/`: vits512_g112_kv28 TensorRT 제출 이미지. `build.sh`(엔진 → 컨텍스트 → 오프라인 docker build),
  `build_engines.py`(실제로 도는 백본을 찾아 ONNX → exact/ampere_plus FP16 엔진 + manifest), `trt_backbone_runtime.py`(드라이버 기동 시
  검증 후 교체, 절대 예외를 올리지 않음), `validate.sh`(제출 전 로컬 점검 10항목), `test_trt_image.py`(프로세스 내 궤적·fallback·지연),
  `run_parity.sh`·`compare_parity.py`(441 동등성, 미실행), `SUBMISSION_RUNBOOK_G112_TRT.md`.
- `pkg3cam_eval/run_pkg3cam.py`: `--gpus`(기본 0–7, 이 계정 카드 4–7 만 쓰는 실행용), model.json `"baked": true` 면 마운트 없이 이미지 그대로 실행,
  `vits512_g112_kv28_trt` 등록. `retry_infra.py`: `--gpus`(기본 0,1).

- `drivor_eval/extra_protocols.py`(원지표·OmniDreams·RoaD 본문 정의), `drivor_eval/literature_tables.py`(문헌 수치 + 우리 재채점 → 기준별 표). 문헌 수치는 원문 대조(2026-10-12).

- `pkg3cam_eval`: 10-10 번들 2개(vits512_g112_kv28, vits512_vidarrender) 등록. models.sh 에 BUNDLE_DATE, run_pkg3cam MODELS=(subject, date), pkg3cam_leaderboard 출력 폴더는 가장 새 번들 날짜, final_fit_when_done.sh 는 대기할 실행과 모델 목록을 인자로 받는다.

- `drivor_eval/run_drivor441.py`: 드라이버를 argv[0] `python`(실행 파일은 drivor-eval env, PATH 앞에 env bin)으로 띄운다. 채점·어댑터 무관, 재개 provenance 대상 아님.

- `axe_local_eval/curve_stop_vs_human.py`: 정책 속도를 같은 위치의 사람 속도와 비교해 커브 구간의 근정지를 센다(2절 기준). 여러 실행을 한 표로.
- `axe_local_eval/run_curve_axe_v9.sh`: 제출 이미지 axe-v9 를 채점 계약 그대로 441 재실행. GPU 4–7 이 빌 때까지 대기하고, rollout.asl 은 롤아웃마다 metrics.parquet 생성 후 삭제(끝에 175 GB 일괄 삭제 방지).
  끝나면 공식 실행과 재현 대조 → 커브 분석까지 자동.

- `drivor_eval/rescore_protocols.py`: 같은 롤아웃을 세 채점 방식(대회=OPTED·ECO / Alpamayo·Qwen-Drive / RoaD)으로 계산. 대회 방식 재계산은 공식 집계와 762 롤아웃 완전 일치.
  RoaD 방식은 후방 추돌을 무시하는데, 기록 재생 뒤차가 ego 를 관통하며 같은 접촉이 0.7 s 뒤 정면·측면으로 바뀌므로(ours 92건) 후방으로 시작된 접촉 구간 전체를 무시하도록 구현.
  중간 결과(135 장면): `runs/drivor-protocols-20261009_2100/protocols.md`. 최종 표 채점 방식은 사용자 결정 대기.

- `tools/delete_trash_when_idle.sh`: .trash-* 디렉터리를 평가·후처리가 없을 때(10분 간격 두 번 확인) ionice idle 로 한 파일씩 삭제.

- 같은 441 장면을 쓴 논문 2편의 AlpaSim 수치를 `drivor_eval/published_alpasim441.json` 에 원문 그대로 정리: OPTED(arXiv 2609.20756, Table I), ECO(arXiv 2609.31383, Table III).
  ECO 의 VaVAM(비선형) 행이 공식 기준 vavam-nonlinear 와 모든 지표가 일치해 열 대응(collision_at_fault, offroad, left_corridor_laterally, progress_clipped_rel, avg_dist_between_incidents_at_fault)을 확정.
- `combined_table.py`: 우리 2개 + 공식 기준(Alpamayo 1, VaVAM 2종) + 논문 보고값을 한 표(CSV/MD)로, "직접 실행 아님·조건이 완전히 같지 않을 수 있음" 주석(한/영) 포함.
- 통합 표에 '학습 도메인'·'참고 사항' 열(행별 학습 데이터 공개 여부, 연결 방식 차이, 미기재 조건). Alpamayo 는 'PAI-AV·NuRec 학습 포함 여부 미공개'로 표기.
- `finalize_after_eval.sh 20261009` 를 nohup 대기 중(`runs/drivor-finalize-20261009.progress.log`): 두 실행 complete → fit_drivor.py → combined_table.py → `runs/drivor-combined-alpasim441-20261009/`.
- 일시정지·재개 시험 끝까지 성공: pausetest 실행이 pause 전 1개 + resume 후 1개로 2/2 집계(000525f6 0.893, 0026708c 0).

## 4. 다음 단계

- g112 TRT 제출: #205 허가 확인 → (원하면) `run_parity.sh` 로 441 동등성 → 런북 3–5절(약관·횟수 → 태그를 이미지 ID 로 → `ECR_URI=... validate.sh` [10] → push → submit `--controller-gains` 필수).

- (완료) axe-v9 커브 분석: 2절.

- DrivoR 본 평가 진행 중(1절). 끝나면 `driver_receipts.json` 의 errors=0 확인 → 인프라 실패 클립 있으면 재시도 → `fit_drivor.py --original runs/drivor-drivor-original-441x3-20261009 --ours runs/drivor-ours-hugsim-final-441x3-20261009`.
- 이전 항목(완료된 10-08 번들 흐름, 참고용):
- 6layer 완료 → 768 자동 시작 확인. vitl 번들 완성 시 `MODEL=vitl512_3layer_lr1 e2e_challenge/pkg3cam_eval/prepare.sh` → `run_pkg3cam.py --model vitl512_3layer_lr1 --smoke <gpu>` (smoke 의 peak 로 드라이버 메모리 확인) → `chain_after.sh leaderboard-stage3-vits768-3layer-20261008 vitl512_3layer_lr1 <extra MiB>`.
- 인프라(RPC) 실패 클립이 있으면 10-07 infra-retry 방식으로 재시도 후, `pkg3cam_leaderboard.py <완료 모델들>` 로 36+N 공동 적합 → 사용자 형식 CSV.

## 5. 미결 질문 (사용자 결정 필요)

- **DrivoR 평가(2026-10-09)**: 기본값으로 진행 중이며 사용자/mingyu 확인 필요: (1) 정밀도 FP32 + 가속도 ±10 m/s² 경계(번들 기본 FP16 아님).
  (2) 후방 — 프록시 접합(추천, mingyu 메모 우선순위 1) vs 3경로 attention mask(우선순위 2, 모델 코드 수정·테스트 필요).
  (3) 441×3(기준과 동일)으로 진행. GPU 0–7 은 승인됨. (4) mingyu 에게 FP32·가속도 경계 공유 필요.
  참고: 이 평가는 axe 계약(16/16/16, MPC 1/0.25/3)이 아니라 번들 프로토콜(기본 MPC, 441×3)을 따른다.

- 이번441최종집계/36리더보드보고에미결없음. 2인프라클립복구완료이며실제모델실패0점은유지했다. 기존axe-v9의역사적RPC1은그대로보존하고미상충돌지표를명시한다.

- 인프라2개복구는기존동일평가완료의범위에포함되며미결승인없음. 439 우선중간보고는사용자명시요청이다. 모델실패클립은재시도하지않으며기존439결과를그대로보존한다.

- 신규disjoint512 ETA문의에사용자결정없음. 첫48개완료전이라추정범위가넓으며실제처리량으로후속갱신한다.

- 신규 3cam 체크포인트 평가에 미결 승인 없음. GPU0–7/최대한 빠른 실행은 명시 승인됐다. 이번 요청은 axe-v9 설정을 유지하므로 5cam 원본의 .6/.4 임계값을 이식하지 않는다. 결과와 정확한 처리시간은 실행 후 측정한다.

- 최종441/리더보드보고에는미결없음. 대표6개후방접근영상/paired그래프는14:46완료. 원본5cam카메라만의효과를분리하는ablation은이번범위에포함되지않았다.

- 이번중간결과보고에사용자결정없음. 후방분석예약PID2544720복구완료,최종성능/접근가속영상결과는원평가와분석완료를기다린다.

- 후방반응분석은사용자요청범위에서예약하며미결승인없음. 실제영상/지표의사용자제공은평가/분석완료후남아있다. 후방카메라만의인과효과를분리할동일체크포인트입력ablation은이번자동분석에포함하지않는다.

- 진행률/종료 예상 보고에 추가 사용자 결정 없음. 현재 실행을 그대로 유지한다.

- 이번 실행의 GPU 범위/속도 우선순위는 사용자가 확정했으며 미결 질문 없음. 실제48-worker VRAM 피크와 처리량, 최종441 점수는 실행에서 확인한다. 아래 연구·제출 결정은 과거 기록.
- 렌더 제외 이력/441 포함 여부 조사 미결 없음. 실제 평가는 해당 클립을 포함한 441개 실시간 렌더링을 유지한다.
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
| DrivoR 번들·env | `../mingyu/DRIVOR_ALPASIM_NUREC441_HANDOFF_20261008_2005`, conda `drivor-eval`(+링크 `kjs-DrivoR-alpasim441`) | 번들 rsync, env 는 번들 `environment/requirements-model.txt` + `drivor_eval/drivor-eval-freeze.txt` |

git 으로 오는 것: 런처·드라이버 패키지·오버레이 코드·문서·실험 원장.

### GitHub 사본(`mine`)과 Git LFS

NVlabs 원본은 테스트 픽스처(usdz/asl/ply 등 20개, 약 225 MB)를 LFS 로 관리한다. GitHub 는
LFS 포인터가 가리키는 객체가 없는 push 를 거부하므로(GH008), 개인 사본에는 LFS 객체까지
올려 두었다(`git lfs fetch --all origin` 으로 NVlabs 에서 받은 뒤 `git lfs push --all mine`).
무료 LFS 용량 1 GB 중 약 225 MB 를 쓴다.

- 다른 서버에서 받을 때 대역폭을 아끼려면: `GIT_LFS_SKIP_SMUDGE=1 git clone -b junseong/e2e-eval git@github-junseong:JunSeongKW/alpasim-e2e-eval.git alpasim`
  (평가에는 픽스처가 필요 없다. pytest 를 돌릴 때만 `git lfs pull`.)
- 이후 커밋에는 LFS 파일이 없으므로 종료 루틴의 `git push mine` 그대로 된다.
