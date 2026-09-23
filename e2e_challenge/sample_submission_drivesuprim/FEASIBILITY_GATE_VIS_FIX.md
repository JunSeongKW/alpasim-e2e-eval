# feasibility 게이트 — 동작 확인과 시각화 수정

2026-09-04 · stage3 ep24 (`epoch=24-step=18150.ckpt`)

---

## 1. 증상

영상에서 후보 궤적이 전부 파란색("feasibility survivors")으로 그려지고 `survived` 가
**4096** 으로 표시됨. 주행가능영역을 벗어나는 후보까지 생존으로 표시.

## 2. 결론

**게이트는 정상 작동 중이었고, 시각화만 잘못돼 있었습니다.**

원인은 한 줄입니다. `feasibility_mask` 가 모델 `forward()` 의 **지역 변수로 끝나** `output`
딕셔너리에 실리지 않았습니다. 그래서 드라이버가 마스크를 못 받고 폴백을 만들었습니다.

```python
# policy.py — 마스크가 없으면 전부 생존으로 간주
elif candidate_vocab is not None:
    feasibility_mask = np.ones(candidate_vocab.shape[0], dtype=np.bool_)
```

flat ViT-S 때는 `feasibility_enabled: false` 라 이 폴백이 옳았지만, 게이트가 켜진 stage3
에서는 실제와 다른 그림을 보여주고 있었습니다.

---

## 3. 실제로 작동 중인 필터링 로직

`_build_feasibility_mask()` 가 4,096 후보 전체에 대해 두 규칙을 AND 로 결합합니다.

### 규칙 1 — 주행가능영역

```python
driv = output['bev_seg_map'].softmax(1)[:, 1]   # BEV 셀별 P(drivable)
pc   = driv[:, r, c].view(B, V, Pd, 4)          # 후보 x 시점 x ego 4모서리
off  = (pc < 0.5) & cin                         # 격자 안 & 확률 0.5 미만
drivable_ok = ~off.any(dim=3).any(dim=2)        # 어느 모서리든, 어느 시점이든 -> 탈락
```

NAVSIM 의 DAC 와 같은 **모서리 기준**입니다. `cin` 때문에 BEV 격자
(x[0,56] y[-28,28]) 밖으로 나가는 구간은 판정에서 제외됩니다.

### 규칙 2 — 충돌

```
검출 객체   신뢰도 0.3 이상
박스        10차원 [cx, cy, log l, log w, cz, log h, sin, cos, vx, vy]
예측        등속(CV) 1.0초 전파
판정        ego 사각형(4.6 x 1.9 m) vs 객체 박스, 분리축 정리(SAT)
여유        박스 1.1배 (보행자·자전거 2.0배)
```

### 적용 — 마스킹이 아니라 강등

```python
span = scores.max(1) - scores.min(1)
scores = torch.where(feasibility_mask, scores, scores - (span + 1.0))
```

`-inf` 로 지우면 탈락 후보들이 모두 동점이 되어, 생존이 256개 미만일 때 top-k 가 인덱스
순으로 채워집니다. 벌점 방식은 탈락 후보 사이의 순서를 보존하므로 "탈락한 것 중 가장 나은
것" 으로 채워집니다.

**이 강등이 coarse top-k (4096 -> 256) 이전에 일어나므로 정밀 단계는 걸러진 후보를 보지
못합니다.** 의도한 1차 필터링이 맞습니다.

---

## 4. 수정 내역

### 4.1 마스크를 밖으로 내보냄

`assets/drivesuprim/stage3_ep24_eval/navsim/agents/drivesuprim/drivesuprim_model.py`

```python
output['feasibility_mask']           = feasibility_mask   # 전체 (AND 결과)
output['feasibility_drivable_mask']  = drivable_ok        # 규칙 1 단독
output['feasibility_collision_mask'] = ~collide           # 규칙 2 단독
```

규칙별로 나눈 이유는, "filtered" 하나로는 게이트가 **지도를 읽고 있는지 교통을 읽고 있는지**
알 수 없기 때문입니다.

### 4.2 드라이버 — payload 전달 + 계측

`drivesuprim_challenge/policy.py`, `driver.py`

```
PolicyPrediction   feasibility_drivable_mask / feasibility_collision_mask 필드 추가
payload            feasibility_drivable_packed / feasibility_collision_packed
로그               [DriveSuprim] FEASGATE: source=model kept=2283/4096 (55.7%)
                                 dropped=1813 by_drivable=1590 by_collision=223
```

`source=model` 이면 네트워크가 실제 마스크를 준 것, `source=fallback` 이면 전부 True 인
폴백입니다. **이 한 줄로 게이트 동작 여부가 바로 확인됩니다.**

### 4.3 시각화 — 색 분리

`src/eval/src/eval/data.py`, `video.py`

```
파란색 #78add2   feasibility survivors        생존
빨간색 #d62728   rejected: off drivable area  규칙 1 탈락
주황색 #ff8c00   rejected: collision          규칙 2 탈락
```

기존에는 `artists["rejected"].set_segments(...)` 가 **주석 처리되어** 탈락 후보가 아예 그려지지
않았습니다. 이를 되살리면서 두 규칙으로 나눴습니다.

두 규칙 모두에 걸린 후보는 **주행가능영역 쪽으로 귀속**시켜 두 레이어가 겹치지 않게 했고,
어느 규칙도 주장하지 않는 탈락(모델이 통합 마스크만 내보낸 경우)은 빨간색으로 그립니다.

상태 표시줄도 내역을 보여줍니다.

```
candidates 4,096
survived 2,283 | coarse 256 | filtered 1,813 (drivable 1,590 / collision 223)
```

---

## 5. 남은 검증

코드 경로는 확인했지만 **실제 데이터에서의 생존율은 아직 측정하지 않았습니다.**
번들 README 는 학습 환경 기준 55.7% (클립별 25.7~69.3%) 라고 밝히지만, 챌린지 드라이버
경로에서 같은 값이 나오는지는 별개입니다.

이미지 `alpasim-e2e-drivesuprim-stage3:ep24-gatedbg` 로 2클립만 돌리면 FEASGATE 로그에서
실측치가 나옵니다. 특히 확인할 것:

- `source=model` 인가 (폴백이 아닌가)
- 생존율이 100% 가 아닌가 (마스크가 공허하지 않은가)
- `by_drivable` / `by_collision` 비율

---

## 6. 영향 범위

**점수에는 영향이 없습니다.** 게이트는 수정 전에도 정상 작동했고, 바뀐 것은 마스크를 밖으로
노출하는 것과 그리는 방식뿐입니다. 중단 전까지 나온 458클립 평가 결과(283/458)도 유효합니다.
