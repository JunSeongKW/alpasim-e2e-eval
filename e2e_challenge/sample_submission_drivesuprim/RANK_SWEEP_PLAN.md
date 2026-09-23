# 랭킹 지수·계수 탐색 실험 계획

작성 2026-09-04 · 대상 `nurec_stage3_ep24` (`epoch=24-step=18150.ckpt`)
목표: **458 클립(`nurec_val_clips.txt`)에서 최적의 지수·계수 조합을 찾는다.**

---

## 1. 무엇을 탐색하는가

플래너는 4,096 후보를 두 단계로 좁힙니다. 두 단계 모두 같은 형태의 점수식을 쓰지만
**독립적으로 조정할 수 있습니다.**

```
score = σ(NC)^e_nc · σ(DAC)^e_dac · σ(EP)^e_ep  +  w_imi · imi_normalised

coarse   4,096 → 256   (+ feasibility 게이트 강등)
fine       256 →   1
```

### 현재 값 (기준선)

| | coarse | fine |
|---|---|---|
| NC / DAC / EP 지수 | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |
| imi 계수 | 1.0 | **0.02** |
| 집계 헤드 계수 | 0.0 | 0.0 |
| imi 정규화 | on | on |

fine 의 0.02 는 [`REFINEMENT_IMI_WEIGHT_PATCH.md`](/home/kaist5/data/junseong/REFINEMENT_IMI_WEIGHT_PATCH.md)
적용분입니다. 그 외는 전부 기본값입니다.

### 탐색 축 8개

```
coarse   e_nc, e_dac, e_ep, w_imi
fine     e_nc, e_dac, e_ep, w_imi
```

---

## 2. 선행 작업 — fine 전용 지수 기능 추가

현재 코드에는 **단계별 지수가 없습니다.** `pdm_rank_product_exponents` 는 두 단계가 공유하고,
단계별로 갈리는 것은 imi 계수뿐입니다. 패치 문서 §6 이 `pdm_rank_product_exponents_refine` 를
"구현했으나 미채택" 으로 언급하므로, imi 와 같은 방식으로 넣습니다.

```python
# drivesuprim_config.py
pdm_rank_product_exponents_refine: Optional[Dict[str, float]] = None

# drivesuprim_model.py
def _rank_product(config, result, use_traffic_light, stage="coarse"):
    exps = (getattr(config, 'pdm_rank_product_exponents_refine', None)
            if stage == "refine" else None)
    if exps is None:
        exps = getattr(config, 'pdm_rank_product_exponents', None) or {}
```

기본값 `None` 이면 기존 동작 그대로라, 넣는 것만으로는 아무것도 바뀌지 않습니다.

---

## 3. 2단계 탐색 — 선별 후 확정

458 클립 한 번이 **약 3시간**입니다. 조합마다 전체를 돌리면 하루 8개가 한계입니다.

```
1단계  선별용 60클립   조합당 ~25분   하루 20+ 조합
2단계  상위 2~3개만 458클립 전체로 확정
```

### 60클립 선별 기준

458 개를 회전각·차량밀도·속도로 층화 추출해 전체 분포를 닮게 만듭니다.

```
458 클립 구성   회전각 중앙 6.9°, 45°+ 89개, 90°+ 13개
                차량밀도 중앙 5.0대,  고속(15 m/s+) 113개
```

**대표성 검증**: 지금 도는 458 실행이 끝나면, 같은 조합으로 60클립을 돌려 두 결과의
클립별 점수를 대조합니다. 어긋나면 선별을 다시 합니다. 이 검증 없이 60클립 결과를
믿으면 전체 탐색이 헛돕니다.

---

## 4. 탐색 순서 — 한 번에 한 축씩

8축 전수 탐색은 불가능합니다. 각 라운드는 **직전 라운드 승자 위에** 쌓습니다.

| 라운드 | 축 | 후보값 | 근거 |
|---|---|---|---|
| R1 | fine w_imi | 0, 0.01, 0.05, 0.1 | 패치 문서상 0~0.03 이 평탄, 0.05 위로 급락 |
| R2 | fine e_dac | 2, 3, 5 | DAC 가 실패의 최대 원인 |
| R3 | fine e_nc, e_ep | 각 2, 3 | 문서상 단독으로는 역효과 — 확인 필요 |
| R4 | coarse w_imi | 0.3, 2.0 | 미탐색 축 |
| R5 | coarse e_nc/e_dac/e_ep | 각 2, 3 | 미탐색 축 |
| R6 | 결합 | R1~R5 승자 조합 | 축 간 상호작용 확인 |

R6 이후에는 그때까지 결과를 보고 다음 라운드를 설계합니다. **중단 지시가 있을 때까지 반복합니다.**

---

## 5. 라운드마다 기록

매 실험 종료 시 `RANK_SWEEP_RESULTS.md` 에 한 항목씩 누적합니다.

```
### R1-c  fine w_imi = 0.05
  조합       coarse 1/1/1, imi 1.0  |  fine 1/1/1, imi 0.05
  결과       평균 0.xxxx   통과 xx/60
  판정       offroad x / 충돌 x / 회랑 x
  회전각별   <20° 0.xx   20~45° 0.xx   45°+ 0.xx
  직전 대비  +0.0xxx
  판단       채택 / 기각 — 한 줄 근거
```

기록 후 바로 다음 조합으로 넘어갑니다.

---

## 6. 주의 — 경계선 클립의 실행 간 편차

앞선 실험에서 같은 설정을 3회 돌렸을 때 **표준편차 0.36** 이 나온 사례가 있습니다
(단일 실행 1위가 3회 중 1회만 통과). 단일 실행으로 순위를 정하면 노이즈를 최적해로
오인합니다.

```
선별 단계   1회 실행으로 대략 거르기
확정 단계   상위 후보는 3회 반복해 평균·표준편차로 판단
```

---

## 7. 실행 환경 (고정)

```
체크포인트   epoch=24-step=18150.ckpt  (sha 1d9fa8d7…)
이미지       alpasim-e2e-drivesuprim-stage3:ep24-refimi2 계열
MPC          long 0.5 / lat 6.0 / idx_start 3      ← 고정, 탐색 대상 아님
워커         16        ← 이 장비의 상한. 32 은 두 방식 모두 실패
영상         off
```

**워커 16 을 넘기지 않습니다.** 32 워커(단일 스택)와 16×2 스택 모두 렌더러가 기동하지
못했고, 복구 과정에서 docker 메타데이터가 손상돼 데몬 재시작이 필요했습니다.

컨테이너 정리는 강제 종료를 반복하지 않고 정상 종료를 기다립니다. `RUN_DIR` 은 매번
새 이름을 씁니다 — 삭제 대기 중인 컨테이너와 이름이 겹치면 compose 가 실패합니다.

---

## 8. 산출물

```
runs/e2e_challenge_drivesuprim_debug/
├── val458_run2/              기준선 458클립 (진행 중)
└── rank_sweep/<라운드>/       각 조합의 실행 결과

e2e_challenge/sample_submission_drivesuprim/
├── RANK_SWEEP_PLAN.md        이 문서
├── RANK_SWEEP_RESULTS.md     라운드별 기록 (누적)
├── val60_clips.txt           선별용 60클립
└── assets/drivesuprim/stage3_ep24_eval/rank_variants/*.json
```
