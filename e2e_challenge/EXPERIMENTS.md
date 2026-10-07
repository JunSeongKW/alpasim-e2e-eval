# 실험 원장 (append-only)

한 실험 = 한 줄. 점수는 `runs/<run>/aggregate/results-summary.json` 의 rollout 평균. 새 실험을 끝내면 맨 아래에 추가한다.
생성: 2026-09-23 (기존 결과 폴더에서 자동 수집). 이후 행은 손으로 적는다.

| 날짜 | run (runs/…) | 클립 | rollout | 평균 점수 | 0점 비율 | corridor 이탈률 | 메모 |
|---|---|---:|---:|---:|---:|---:|---|
| 2026-09-11 | vizA_baseline | 1 | 1 | 1.0000 | 0.000 | 0.000 | |
| 2026-09-11 | vizB_egobox | 1 | 1 | 0.0000 | 1.000 | 0.000 | |
| 2026-09-12 | axe-ep24-curatedval-dev | 441 | 1323 | 0.6396 | 0.283 | 0.122 | |
| 2026-09-12 | axe-ep24-curatedval-egobox | 441 | 1323 | 0.6742 | 0.238 | 0.118 | |
| 2026-09-12 | axe-v8-asubmitted-ec2 | 117 | 117 | 0.5346 | 0.368 | 0.197 | |
| 2026-09-12 | axe-v8-latency-conc2 | 24 | 24 | 0.4778 | 0.375 | 0.208 | |
| 2026-09-12 | submit-v1-ec2 | 117 | 351 | 0.5634 | 0.345 | 0.205 | |
| 2026-09-13 | leaderboard-axe-v4 | 441 | 441 | 0.2073 | 0.465 | 0.188 | |
| 2026-09-13 | leaderboard-axe-v5 | 441 | 441 | 0.2275 | 0.565 | 0.216 | |
| 2026-09-13 | leaderboard-stage3-ep30 | 441 | 441 | 0.6652 | 0.238 | 0.134 | |
| 2026-09-14 | leaderboard-flatvits-ep25 | 441 | 441 | 0.5554 | 0.336 | 0.118 | |
| 2026-09-15 | leaderboard-epzero-ep29 | 441 | 441 | 0.6961 | 0.229 | 0.079 | |
| 2026-09-15 | leaderboard-epzero-ep29-basegains | 441 | 441 | 0.6977 | 0.202 | 0.077 | |
| 2026-09-16 | leaderboard-merged-ep29-bestmpc | 441 | 441 | 0.7046 | 0.213 | 0.080 | |
| 2026-09-17 | leaderboard-merged-route-ep30 | 441 | 441 | 0.6993 | 0.220 | 0.091 | |
| 2026-09-17 | viz10-axe-v9 | 10 | 10 | 0.7050 | 0.200 | 0.100 | |
| 2026-09-21 | alpamayo15-val50 | 36 | 36 | 0.5235 | 0.444 | 0.306 | |
| 2026-09-22 | routecache-h1-after | 10 | 10 | 0.4140 | 0.500 | 0.250 | |
| 2026-09-22 | routecache-val441 | 441 | 441 | 0.6852 | 0.243 | 0.080 | |
| 2026-09-22 | routecache3-after | 10 | 10 | 0.4050 | 0.500 | 0.500 | |
| 2026-09-22 | routecache3-before | 10 | 10 | 0.1547 | 0.800 | 0.800 | |
| 2026-09-22 | routecache4-after | 10 | 10 | 0.2202 | 0.700 | 0.667 | |
| 2026-09-23 | routecache-val441-center1s | 441 | 441 | 0.6598 | 0.263 | 0.099 | |
| 2026-09-23 | routecache-val441-center1s-repair | 17 | 17 | 0.5902 | 0.294 | 0.176 | |
| 2026-09-23 | rerank10-before | 10 | 10 | 0.2362 | 0.700 | 0.600 | 동일 이미지·설정 대조군. 점수·영상 생성 후 Docker 종료 오류로 wizard exit 1. |
| 2026-09-23 | rerank10-rerank | 10 | 10 | 0.2324 | 0.700 | 0.500 | route 리랭커 ON; 선택 변경 1/1,896프레임. 상세 `route_reranker/RESULTS_10CLIPS.md`. |

| 2026-09-24 | rr441-g0p02-cache-centre-max | 441 | 441 | 0.7299 | 0.191 | 0.073 | 리랭커 γ=0.02, cache-centre-max. axe-v9 대비 +0.0306, 전 지표 개선 |
| 2026-09-24 | rr441-g0p1-cache-centre-max | 441 | 441 | 0.7242 | 0.170 | 0.064 | 리랭커 γ=0.1. 0점률·이탈 최저지만 충돌 +0.040, 주행거리 −12 m |
| 2026-09-24 | leaderboard-260923-ep29-step30330 | 441 | 441 | 0.6886 | 0.236 | 0.091 | ep29 체크포인트. axe-v9 대비 −0.0107, at-fault 충돌 2배 |
| 2026-09-25 | rr441-g0p02-augep04-cache-centre-max | 441 | 441 | 0.4308 | 0.499 | 0.220 | stage3_aug_ep04 + 리랭커 γ=0.02. 미수렴 체크포인트: 계획편차 0.409, GT오차 2.95 |
| 2026-09-25 | leaderboard-aug-ep19 | 441 | 441 | 0.6273 | 0.238 | 0.107 | stage3_aug_ep19, 리랭커 없음, axe-v9 동일 환경. 충돌 0.311 로 최악 |
| 2026-09-26 | leaderboard-aug-ep29-final | 441 | 441 | 0.6951 | 0.211 | 0.077 | stage3_aug_ep29_final. axe-v9 −0.0042 로 사실상 동급, 0점률·corridor·GT오차는 우위. at-fault 0.061 이 PCS 10위로 끌어내림 |
| 2026-09-26 | leaderboard-disjoint-ep04 | 441 | 441 | 0.6328 | 0.265 | 0.122 | stage3_disjoint_ep04. 같은 epoch 의 aug 보다 +0.20 높음(aug-ep04 는 리랭커 포함 0.4308) |
| 2026-09-26 | leaderboard-disjoint-ep29 | 441 | 441 | 0.6526 | 0.231 | 0.091 | stage3_disjoint_ep29. ep04→ep29 개선폭 +0.020 뿐, aug 계열(+0.26)과 대비. disjoint 가 수렴을 일찍 멈춤 |
| 2026-09-27 | rr441-g0p02-disjointep04-cache-centre-max | 441 | 441 | 0.6472 | 0.252 | 0.125 | stage3_disjoint_ep04 + 리랭커 γ=0.02. 리랭커 없는 같은 체크포인트 대비 +0.0144(axe-v9 에서는 +0.0305). 개선 56 / 악화 59 클립 — 건수로는 지고 폭으로 이김. corridor 이탈은 오히려 증가(0.122→0.125) |

메모 채울 것: 체크포인트/이미지, GPU, 바꾼 변수 하나, 판정(기준선 대비). 실행 중: 없음(2026-09-26 10:18 체크포인트 대기열 종료). `../models/` 의 11개 체크포인트 전부 441 평가 완료. 통합 리더보드: `runs/leaderboard-260927/capability_ranking.csv` (로컬 20 + 참조 8 주체 동시 적합, 경고 없음).


| 날짜 | run (runs/…) | 클립 | rollout | 평균 점수 | 0점 비율 | corridor 이탈률 | 메모 |
|---|---|---:|---:|---:|---:|---:|---|
| 2026-10-07 | leaderboard-stage3-5cam-ep05-20261007 | 441 | 441 | 0.4316 | 0.286 | 0.109 | 원본 stage3_5cam_ep05_20261007 / 5카메라 / BEV 56×112 / 임계값 .6/.4 / dev·MPC1/.25/3 / GPU0–7×6. 지원되지 않는 CUDA 연산은 원본 PyTorch fallback으로 호환 복구 후 전441 재평가. 기존34+신규1 joint fit PCS1894, Rank25(25–26), at-fault0.4516km. axe-v9 평균.6993/후방충돌87 대비 .4316/150. 상황별후방입력필요성 분석: 접근210/반응관찰176/가속+1m/s27; 카메라 인과효과 미검증. |

| 2026-10-07 | leaderboard-vits512-disjoint-baseline-ep04-20261007 | 441 | 441 | 0.6001 | 0.315 | 0.163 | 20261007_vits512_disjoint_baseline_stage3_epoch04-step4075.ckpt / train·val 분리 / 3카메라 / axe-v9 코드·BEV56×56·임계값·추론 그대로 / checkpoint만교체 / dev·MPC1/.25/3 / GPU0–7×6. 정상439고정+renderer연결오류2만GPU0,1병렬복구(최종1점/GT corridor0점), 최종인프라오류0. 같은36주체fit PCS2353·Rank21(15–23)·atfault0.9763km; axe-v9 PCS2704/Rank2/mean.6993/rear87, 5cam PCS1896/Rank26/mean.4316/rear150 대비 신규rear69. 입력필요성/학습분리통제비교자료이며카메라만의인과효과는미검증. 최종19:56 KST완료. |
