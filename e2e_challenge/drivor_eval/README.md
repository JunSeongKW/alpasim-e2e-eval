# DrivoR 2종 AlpaSim curated_val 441 평가 (mingyu 인수인계 번들)

대상 번들: `../mingyu/DRIVOR_ALPASIM_NUREC441_HANDOFF_20261008_2005` (체크포인트·모델 코드는 번들 그대로, 수정하지 않음).
정책 2개: `drivor_original`(Original V1, 계수 1/1/0/5/5/2), `ours_hugsim_final`(A8 Z_safe + B12 + D10 ALL4, 12/25/2/6/5/2).

## 프로토콜 (번들 `protocol/PROTOCOL_DIFFERENCES.md`)

- `+e2e_challenge=dev +nurec_scenes=curated_val`, 장면당 롤아웃 3회(기준 주체 8개와 같은 441×3).
- **MPC 는 기본값**(`MPC_OVERRIDES=""`). axe 평가의 조정값 1/0.25/3 을 쓰지 않는다.
- 441 장면으로 아무것도 조정하지 않는다. 어댑터는 두 모델에 똑같이 적용하고 점수를 보기 전에 고정한다.
- 리더보드는 공식 기준 8개 + DrivoR 2개만으로 한 번에 적합한다(`fit_drivor.py`). 우리 39주체 표에 섞지 않는다.

## 어댑터 (`drivor_driver.py`, `pai_views.py`)

| 항목 | 내용 | 근거 |
|---|---|---|
| 카메라 | PAI f-theta → nuPlan 1920×1080 왜곡 핀홀. F0←front_wide, L0←cross_left, R0←cross_right 는 회전만 재투영, B0←rear_left+rear_right 는 지면+15 m 원통 프록시 접합. 아무 카메라도 못 보는 화소는 가장 가까운 가장자리로 채움 | 공식 VaVAM rectifier 와 같은 방식. 회전만으로 B0 를 접합하면 바로 뒤 차량이 이음매에서 사라지거나 반투명해진다(카메라 좌우 간격 1 m > 차폭 절반) |
| nuPlan 기준 자세 | `nuplan_rig.json` (NAVSIM trainval 1,310 로그 중앙값, 내부 파라미터는 97.8% 를 차지하는 기본값) | `extract_nuplan_rig.py` |
| 명령 | 공식 `determine_command_from_route`, 임계 3 m / lookahead 20 m (VaVAM·Alpamayo 공식 설정). dev 프리셋 경로는 40 m 앞부터 시작 | `calibrate_command.py`: NAVSIM 명령과 균형정확도 0.884, 최선 0.889 와 0.005 차 |
| ego | NAVSIM ego_status 4행(2 Hz), 속도는 DynamicState 그대로, **가속도만 ±10 m/s²(≈1 g) 경계** | 시뮬 첫 몇 상태에 유한차분 스파이크(롤아웃 44개 중 16개, 최대 −761 m/s²) → 모든 후보 충돌 logit −37~−147 → FP32 에서도 점수 전부 −∞ 로 정책 중단(clipgt-032b6f21). 그 뒤 MPC 는 \|ax\| ≤ 9.0 이라 정상 주행은 건드리지 않음. NAVSIM 범위(−5.2~3.4)로 자르면 28% 스텝이 바뀌어 기각 |
| 정밀도 | **FP32** (번들 기본 FP16 autocast 아님) | FP16 은 sigmoid 가 logit −16.6 아래에서 0 으로 underflow(FP32 는 ≈−104). 같은 가중치·입력에서 FP16 과 선택 궤적 차이는 4 s 지점 중앙값 0.01 m |
| 출력 | 선택 궤적 8점(0.5–4.0 s) + 현재 자세, 보간·평활·외삽 없음. 실패는 gRPC 오류(대체 궤적 없음) | 공식 드라이버 변환과 동일 |

## 실행

```bash
# 환경: conda env drivor-eval (py3.9, torch 2.1.0+cu121). nvidia-smi 이름용 링크 kjs-DrivoR-alpasim441
# 1) 오프라인 점검
$ENV/bin/python -m pytest e2e_challenge/drivor_eval/test_pai_views.py -q
$ENV/bin/python e2e_challenge/drivor_eval/render_check.py --clip-dir <asl_frames.py 출력> --out check.png
UV_OFFLINE=1 .venv/bin/python e2e_challenge/drivor_eval/replay_asl.py <rollout.asl> --address 127.0.0.1:<port>
# 2) 라이브 스모크 (장면 2개, 롤아웃 1)
.venv/bin/python e2e_challenge/drivor_eval/run_drivor441.py --model ours_hugsim_final --smoke 2 --gpus 4,5 --replicas 1
# 3) 본 평가 441×3 (ASL 은 채점 후 삭제)
.venv/bin/python e2e_challenge/drivor_eval/run_drivor441.py --model ours_hugsim_final --gpus 4,5,6,7 --replicas 6 --prune-asl
# 4) 적합
.venv/bin/python e2e_challenge/drivor_eval/fit_drivor.py --original runs/<original run> --ours runs/<ours run>
```

드라이버는 호스트 프로세스다(컨테이너 아님). 실행 디렉터리의 `drivers.pids` 로만 정리하며, 각 드라이버의
`sessions.jsonl` 에 세션별 영수증(forwards, errors, 명령 분포, 커버리지, 지연)이 남는다. 본 평가는 오류 0 이어야 적합에 들어간다.
