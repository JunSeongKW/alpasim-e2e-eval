# 랭킹 지수·계수 탐색 결과

계획: [`RANK_SWEEP_PLAN.md`](RANK_SWEEP_PLAN.md) · 선별 클립 `val60_clips.txt` (458개에서 층화 추출)

### 기준선 — fine imi 0.02 (REFINEMENT_IMI_WEIGHT_PATCH 적용값)

```
coarse   exp 1/1/1   imi 1.0
fine     exp (coarse 와 동일)   imi 0.02
```

- **458 전체: 평균 0.6313   통과 319/458 (69.7%)**
- **60 부분집합: 평균 0.6274   통과 41/60** ← 이후 조합의 비교 기준
- 458 판정: 회랑이탈 58 · offroad 47 · 충돌 27 · 경로생성실패 2
- 회전각별(458): <20 0.785(306)   20~45 0.453(63)   45+ 0.227(89)
- 회전각별(60):  <20 0.788(40)    20~45 0.390(8)    45+ 0.250(12)

**60클립 대표성 검증**: 458 전체와 0.0038 차이. 60클립 결과를 458 성능의 대리로 써도
무방하다. (60클립 값은 458 실행에서 해당 클립만 추출한 것이라 별도 실행이 아니다.)

### R1a — fine imi 0.0

```
coarse   exp 1/1/1   imi 1.0
fine     exp (coarse 와 동일)   imi 0.0
```

- **평균 0.5688   통과 40/60**
- 판정: pass 40  left_corridor_laterall 8  collision_at_fault 5  offroad 5  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.768(40)   20~45 0.389(8)   45+ 0.026(12)
- 위치: `sweep_R1a_005440`

### R1b — fine imi 0.01

```
coarse   exp 1/1/1   imi 1.0
fine     exp (coarse 와 동일)   imi 0.01
```

- **평균 0.6082   통과 42/60**
- 판정: pass 42  collision_at_fault 7  left_corridor_laterall 6  offroad 3  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.792(40)   20~45 0.350(8)   45+ 0.168(12)
- 위치: `sweep_R1b_015335`

### R1c — fine imi 0.05

```
coarse   exp 1/1/1   imi 1.0
fine     exp (coarse 와 동일)   imi 0.05
```

- **평균 0.5733   통과 39/60**
- 판정: pass 39  offroad 7  left_corridor_laterall 6  collision_at_fault 6  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.767(40)   20~45 0.248(8)   45+ 0.144(12)
- 위치: `sweep_R1c_025956`

### R1d — fine imi 0.1

```
coarse   exp 1/1/1   imi 1.0
fine     exp (coarse 와 동일)   imi 0.1
```

- **평균 0.6266   통과 41/60**
- 판정: pass 41  left_corridor_laterall 7  offroad 7  collision_at_fault 3  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.777(40)   20~45 0.450(8)   45+ 0.243(12)
- 위치: `sweep_R1d_035817`

### R2a — fine DAC 지수 3

```
coarse   exp 1/1/1   imi 1.0
fine     exp DAC:3   imi 0.02
```

- **평균 0.6143   통과 43/60**
- 판정: pass 43  left_corridor_laterall 7  collision_at_fault 6  offroad 2  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.806(40)   20~45 0.322(8)   45+ 0.172(12)
- 위치: `sweep_R2a_050041`

### R2c — fine DAC 지수 0.5

```
coarse   exp 1/1/1   imi 1.0
fine     exp DAC:0.5   imi 0.02
```

- **평균 0.6699   통과 44/60**
- 판정: pass 44  left_corridor_laterall 6  collision_at_fault 4  offroad 4  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.845(40)   20~45 0.431(8)   45+ 0.245(12)
- 위치: `sweep_R2c_060301`

### R3a — fine EP 지수 2

```
coarse   exp 1/1/1   imi 1.0
fine     exp EP:2   imi 0.02
```

- **평균 0.6396   통과 43/60**
- 판정: pass 43  left_corridor_laterall 6  collision_at_fault 6  offroad 3  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.825(40)   20~45 0.419(8)   45+ 0.170(12)
- 위치: `sweep_R3a_065731`

### R2d — fine DAC 지수 0.3

```
coarse   exp 1/1/1   imi 1.0
fine     exp DAC:0.3   imi 0.02
```

- **평균 0.6146   통과 41/60**
- 판정: pass 41  left_corridor_laterall 7  collision_at_fault 5  offroad 5  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.778(40)   20~45 0.445(8)   45+ 0.182(12)
- 위치: `sweep_R2d_084921`

### R2e — fine DAC 0.15

```
coarse   exp 1/1/1   imi 1.0
fine     exp DAC:0.15   imi 0.02
```

- **평균 0.6225   통과 41/60**
- 판정: pass 41  offroad 8  collision_at_fault 5  left_corridor_laterall 4  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.816(40)   20~45 0.437(8)   45+ 0.103(12)
- 위치: `sweep_R2e_095016`

### R2f — fine DAC 0.0

```
coarse   exp 1/1/1   imi 1.0
fine     exp DAC:0   imi 0.02
```

- **평균 0.5986   통과 38/60**
- 판정: pass 38  offroad 11  left_corridor_laterall 6  collision_at_fault 3  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.758(40)   20~45 0.473(8)   45+ 0.150(12)
- 위치: `sweep_R2f_103655`

### R2g — fine DAC 0.7

```
coarse   exp 1/1/1   imi 1.0
fine     exp DAC:0.7   imi 0.02
```

- **평균 0.6319   통과 43/60**
- 판정: pass 43  left_corridor_laterall 6  collision_at_fault 5  offroad 4  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.769(40)   20~45 0.510(8)   45+ 0.256(12)
- 위치: `sweep_R2g_113202`

### R3b — fine DAC0.5 + EP2

```
coarse   exp 1/1/1   imi 1.0
fine     exp DAC:0.5 EP:2   imi 0.02
```

- **평균 0.6075   통과 41/60**
- 판정: pass 41  left_corridor_laterall 6  offroad 6  collision_at_fault 5  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.769(40)   20~45 0.446(8)   45+ 0.179(12)
- 위치: `sweep_R3b_122731`

### R3c — fine DAC0.5 + NC2

```
coarse   exp 1/1/1   imi 1.0
fine     exp DAC:0.5 NC:2   imi 0.02
```

- **평균 0.6168   통과 41/60**
- 판정: pass 41  left_corridor_laterall 8  collision_at_fault 5  offroad 4  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.793(40)   20~45 0.417(8)   45+ 0.162(12)
- 위치: `sweep_R3c_131845`

### R3d — fine DAC0.5 + NC0.5

```
coarse   exp 1/1/1   imi 1.0
fine     exp DAC:0.5 NC:0.5   imi 0.02
```

- **평균 0.6349   통과 43/60**
- 판정: pass 43  left_corridor_laterall 6  collision_at_fault 5  offroad 4  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.812(40)   20~45 0.438(8)   45+ 0.175(12)
- 위치: `sweep_R3d_141301`

### R3e — fine DAC0.5 + EP0.5

```
coarse   exp 1/1/1   imi 1.0
fine     exp DAC:0.5 EP:0.5   imi 0.02
```

- **평균 0.6296   통과 42/60**
- 판정: pass 42  collision_at_fault 7  left_corridor_laterall 5  offroad 4  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.794(40)   20~45 0.443(8)   45+ 0.207(12)
- 위치: `sweep_R3e_150635`

### R3f — fine DAC0.5 + NC0.5 + EP0.5

```
coarse   exp 1/1/1   imi 1.0
fine     exp DAC:0.5 EP:0.5 NC:0.5   imi 0.02
```

- **평균 0.6115   통과 41/60**
- 판정: pass 41  left_corridor_laterall 7  collision_at_fault 6  offroad 4  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.778(40)   20~45 0.434(8)   45+ 0.175(12)
- 위치: `sweep_R3f_155535`

### R4a — coarse imi 0.3 (fine DAC0.5 유지)

```
coarse   exp 1/1/1   imi 0.3
fine     exp DAC:0.5   imi 0.02
```

- **평균 0.6251   통과 42/60**
- 판정: pass 42  collision_at_fault 8  left_corridor_laterall 5  offroad 3  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.803(40)   20~45 0.411(8)   45+ 0.176(12)
- 위치: `sweep_R4a_163616`

### R4b — coarse imi 2.0 (fine DAC0.5 유지)

```
coarse   exp 1/1/1   imi 2.0
fine     exp DAC:0.5   imi 0.02
```

- **평균 0.6351   통과 42/60**
- 판정: pass 42  left_corridor_laterall 8  collision_at_fault 4  offroad 4  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.805(40)   20~45 0.371(8)   45+ 0.246(12)
- 위치: `sweep_R4b_171700`

### R5a — coarse DAC 0.5 (fine DAC0.5 유지)

```
coarse   exp DAC:0.5   imi 1.0
fine     exp DAC:0.5   imi 0.02
```

- **평균 0.6037   통과 40/60**
- 판정: pass 40  left_corridor_laterall 7  collision_at_fault 6  offroad 5  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.779(40)   20~45 0.391(8)   45+ 0.162(12)
- 위치: `sweep_R5a_180602`

### R5b — coarse DAC 2 (fine DAC0.5 유지)

```
coarse   exp DAC:2   imi 1.0
fine     exp DAC:0.5   imi 0.02
```

- **평균 0.6154   통과 40/60**
- 판정: pass 40  left_corridor_laterall 7  offroad 7  collision_at_fault 4  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.784(40)   20~45 0.448(8)   45+ 0.164(12)
- 위치: `sweep_R5b_185355`

### R5c — coarse EP 2 (fine DAC0.5 유지)

```
coarse   exp EP:2   imi 1.0
fine     exp DAC:0.5   imi 0.02
```

- **평균 0.5683   통과 38/60**
- 판정: pass 38  collision_at_fault 8  left_corridor_laterall 7  offroad 5  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.723(40)   20~45 0.397(8)   45+ 0.165(12)
- 위치: `sweep_R5c_194522`

### R5d — coarse NC 2 (fine DAC0.5 유지)

```
coarse   exp NC:2   imi 1.0
fine     exp DAC:0.5   imi 0.02
```

- **평균 0.6270   통과 42/60**
- 판정: pass 42  left_corridor_laterall 7  collision_at_fault 5  offroad 4  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.776(40)   20~45 0.441(8)   45+ 0.254(12)
- 위치: `sweep_R5d_202759`

### A1 — AlpaSim형: fine NC0.3 DAC0.3 EP2 (관문화 + 진행률 강화)

```
coarse   exp 1/1/1   imi 1.0
fine     exp DAC:0.3 EP:2 NC:0.3   imi 0.02
```

- **평균 0.6662   통과 44/60**
- 판정: pass 44  left_corridor_laterall 6  collision_at_fault 4  offroad 4  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.827(40)   20~45 0.546(8)   45+ 0.210(12)
- 위치: `sweep_A1_210658`

### A2 — AlpaSim형: fine NC0.5 DAC0.5 EP3

```
coarse   exp 1/1/1   imi 1.0
fine     exp DAC:0.5 EP:3 NC:0.5   imi 0.02
```

- **평균 0.6224   통과 42/60**
- 판정: pass 42  collision_at_fault 7  offroad 5  left_corridor_laterall 4  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.792(40)   20~45 0.437(8)   45+ 0.182(12)
- 위치: `sweep_A2_214548`

### A3 — AlpaSim형: fine NC0.2 DAC0.2 EP1 (관문화만)

```
coarse   exp 1/1/1   imi 1.0
fine     exp DAC:0.2 EP:1 NC:0.2   imi 0.02
```

- **평균 0.6233   통과 41/60**
- 판정: pass 41  collision_at_fault 7  offroad 6  left_corridor_laterall 4  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.772(40)   20~45 0.498(8)   45+ 0.211(12)
- 위치: `sweep_A3_222300`

### A4 — AlpaSim형: coarse·fine 양쪽 NC0.3 DAC0.3 EP2

```
coarse   exp DAC:0.3 EP:2 NC:0.3   imi 1.0
fine     exp DAC:0.3 EP:2 NC:0.3   imi 0.02
```

- **평균 0.6612   통과 43/60**
- 판정: pass 43  collision_at_fault 6  left_corridor_laterall 5  offroad 4  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.832(40)   20~45 0.487(8)   45+ 0.209(12)
- 위치: `sweep_A4_230135`

### R2c_r2 — 반복2: fine DAC 0.5

```
coarse   exp 1/1/1   imi 1.0
fine     exp DAC:0.5   imi 0.02
```

- **평균 0.6251   통과 41/60**
- 판정: pass 41  left_corridor_laterall 7  collision_at_fault 5  offroad 5  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.796(40)   20~45 0.465(8)   45+ 0.164(12)
- 위치: `sweep_R2c_r2_091933`

### R2c_r2 — 반복2: fine DAC 0.5

```
coarse   exp 1/1/1   imi 1.0
fine     exp DAC:0.5   imi 0.02
```

- **평균 0.6480   통과 43/60**
- 판정: pass 43  left_corridor_laterall 6  collision_at_fault 5  offroad 4  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.801(40)   20~45 0.417(8)   45+ 0.292(12)
- 위치: `sweep_R2c_r2_095430`

### A1_r2 — 반복2: fine NC.3 DAC.3 EP2

```
coarse   exp 1/1/1   imi 1.0
fine     exp DAC:0.3 EP:2 NC:0.3   imi 0.02
```

- **평균 0.6296   통과 41/60**
- 판정: pass 41  collision_at_fault 7  left_corridor_laterall 6  offroad 4  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.784(40)   20~45 0.525(8)   45+ 0.185(12)
- 위치: `sweep_A1_r2_103134`

### BASE_r2 — 반복2: 기준선 (전부 1.0)

```
coarse   exp 1/1/1   imi 1.0
fine     exp (coarse 와 동일)   imi 0.02
```

- **평균 0.6039   통과 41/60**
- 판정: pass 41  left_corridor_laterall 8  collision_at_fault 5  offroad 4  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.748(40)   20~45 0.504(8)   45+ 0.189(12)
- 위치: `sweep_BASE_r2_112838`

### R2c_r3 — 반복3: fine DAC 0.5

```
coarse   exp 1/1/1   imi 1.0
fine     exp DAC:0.5   imi 0.02
```

- **평균 0.6338   통과 42/60**
- 판정: pass 42  collision_at_fault 6  left_corridor_laterall 5  offroad 5  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.799(40)   20~45 0.450(8)   45+ 0.205(12)
- 위치: `sweep_R2c_r3_122446`

### A1_r3 — 반복3: fine NC.3 DAC.3 EP2

```
coarse   exp 1/1/1   imi 1.0
fine     exp DAC:0.3 EP:2 NC:0.3   imi 0.02
```

- **평균 0.6468   통과 42/60**
- 판정: pass 42  left_corridor_laterall 6  offroad 6  collision_at_fault 4  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.797(40)   20~45 0.553(8)   45+ 0.208(12)
- 위치: `sweep_A1_r3_133304`

### BASE_r3 — 반복3: 기준선

```
coarse   exp 1/1/1   imi 1.0
fine     exp (coarse 와 동일)   imi 0.02
```

- **평균 0.6742   통과 45/60**
- 판정: pass 45  left_corridor_laterall 5  offroad 5  collision_at_fault 3  Waypoint 53 fails sani 1  Waypoint 118 fails san 1
- 회전각별: <20 0.848(40)   20~45 0.452(8)   45+ 0.244(12)
- 위치: `sweep_BASE_r3_143312`

### V_base — 150클립 기준선 (전부 1.0)

```
coarse   exp 1/1/1   imi 1.0
fine     exp (coarse 와 동일)   imi 0.02
```

- **평균 0.6134   통과 99/150**
- 판정: pass 99  offroad 20  left_corridor_laterall 18  collision_at_fault 11  Waypoint 82 does not p 1  Waypoint 47 fails sani 1
- 회전각별: <20 0.780(100)   20~45 0.355(21)   45+ 0.225(29)
- 위치: `sweep_V_base_153641`

### V_dac5 — 150클립: fine DAC 0.5

```
coarse   exp 1/1/1   imi 1.0
fine     exp DAC:0.5   imi 0.02
```

- **평균 0.6021   통과 98/150**
- 판정: pass 98  offroad 22  left_corridor_laterall 17  collision_at_fault 11  Waypoint 82 does not p 1  Waypoint 47 fails sani 1
- 회전각별: <20 0.741(100)   20~45 0.483(21)   45+ 0.210(29)
- 위치: `sweep_V_dac5_165954`

### V_a1 — 150클립: fine NC.3 DAC.3 EP2

```
coarse   exp 1/1/1   imi 1.0
fine     exp DAC:0.3 EP:2 NC:0.3   imi 0.02
```

- **평균 0.6061   통과 97/150**
- 판정: pass 97  offroad 22  collision_at_fault 15  left_corridor_laterall 14  Waypoint 82 does not p 1  Waypoint 47 fails sani 1
- 회전각별: <20 0.756(100)   20~45 0.413(21)   45+ 0.229(29)
- 위치: `sweep_V_a1_181657`

### G1 — 자동제안: fine {'DAC': 0.3, 'EP': 2.0, 'NC': 0.4} coarse {'DAC': 0.3, 'EP': 2.0, 'NC': 0.3} imi c1.0/f0.02

```
coarse   exp DAC:0.3 EP:2 NC:0.3   imi 1.0
fine     exp DAC:0.3 EP:2 NC:0.4   imi 0.02
```

- **평균 0.5795   통과 92/150**
- 판정: pass 92  offroad 27  collision_at_fault 16  left_corridor_laterall 13  Waypoint 82 does not p 1  Waypoint 47 fails sani 1
- 회전각별: <20 0.724(100)   20~45 0.414(21)   45+ 0.201(29)
- 위치: `sweep_G1_192910`

### G2 — 자동제안: fine {'DAC': 0.3, 'EP': 0.85, 'NC': 0.3} coarse {'DAC': 0.3, 'EP': 2.0, 'NC': 0.3} imi c1.0/f0.02

```
coarse   exp DAC:0.3 EP:2 NC:0.3   imi 1.0
fine     exp DAC:0.3 EP:0.85 NC:0.3   imi 0.02
```

- **평균 0.5773   통과 93/150**
- 판정: pass 93  offroad 23  left_corridor_laterall 18  collision_at_fault 14  Waypoint 82 does not p 1  Waypoint 47 fails sani 1
- 회전각별: <20 0.725(100)   20~45 0.406(21)   45+ 0.193(29)
- 위치: `sweep_G2_204022`

### G5 — 자동제안: fine {'NC': 0.5, 'DAC': 1.3, 'EP': 0.3} coarse 1/1/1 imi c1.0/f0.02

```
coarse   exp 1/1/1   imi 1.0
fine     exp DAC:1.3 EP:0.3 NC:0.5   imi 0.02
```

- **평균 0.6086   통과 101/150**
- 판정: pass 101  offroad 19  left_corridor_laterall 17  collision_at_fault 11  Waypoint 82 does not p 1  Waypoint 47 fails sani 1
- 회전각별: <20 0.755(100)   20~45 0.445(21)   45+ 0.223(29)
- 위치: `sweep_G5_223625`

### G6 — 자동제안: fine {'NC': 1.7} coarse 1/1/1 imi c1.0/f0.02

```
coarse   exp 1/1/1   imi 1.0
fine     exp NC:1.7   imi 0.02
```

- **평균 0.5803   통과 95/150**
- 판정: pass 95  offroad 21  left_corridor_laterall 20  collision_at_fault 12  Waypoint 82 does not p 1  Waypoint 47 fails sani 1
- 회전각별: <20 0.742(100)   20~45 0.340(21)   45+ 0.198(29)
- 위치: `sweep_G6_234438`

### G7 — 자동제안: fine {'NC': 0.85, 'DAC': 1.7, 'EP': 1.3} coarse 1/1/1 imi c1.0/f0.02

```
coarse   exp 1/1/1   imi 1.0
fine     exp DAC:1.7 EP:1.3 NC:0.85   imi 0.02
```

- **평균 0.6033   통과 98/150**
- 판정: pass 98  offroad 20  left_corridor_laterall 18  collision_at_fault 12  Waypoint 82 does not p 1  Waypoint 47 fails sani 1
- 회전각별: <20 0.751(100)   20~45 0.448(21)   45+ 0.206(29)
- 위치: `sweep_G7_005729`

### G8 — 자동제안: fine {'NC': 3.0, 'DAC': 0.3, 'EP': 0.3} coarse 1/1/1 imi c1.0/f0.02

```
coarse   exp 1/1/1   imi 1.0
fine     exp DAC:0.3 EP:0.3 NC:3   imi 0.02
```

- **평균 0.5811   통과 97/150**
- 판정: pass 97  offroad 20  left_corridor_laterall 20  collision_at_fault 11  Waypoint 82 does not p 1  Waypoint 47 fails sani 1
- 회전각별: <20 0.726(100)   20~45 0.322(21)   45+ 0.270(29)
- 위치: `sweep_G8_020651`

### G9 — 자동제안: fine {'DAC': 0.3, 'EP': 0.85, 'NC': 0.5} coarse {'DAC': 0.3, 'EP': 2.0, 'NC': 0.3} imi c1.0/f0.02

```
coarse   exp DAC:0.3 EP:2 NC:0.3   imi 1.0
fine     exp DAC:0.3 EP:0.85 NC:0.5   imi 0.02
```

- **평균 0.5647   통과 90/150**
- 판정: pass 90  offroad 23  left_corridor_laterall 19  collision_at_fault 16  Waypoint 82 does not p 1  Waypoint 47 fails sani 1
- 회전각별: <20 0.691(100)   20~45 0.452(21)   45+ 0.210(29)
- 위치: `sweep_G9_031532`

### G10 — 자동제안: fine {'DAC': 0.3, 'EP': 2.0, 'NC': 0.7} coarse {'DAC': 0.3, 'EP': 2.0, 'NC': 0.3} imi c1.0/f0.02

```
coarse   exp DAC:0.3 EP:2 NC:0.3   imi 1.0
fine     exp DAC:0.3 EP:2 NC:0.7   imi 0.02
```

- **평균 0.5860   통과 93/150**
- 판정: pass 93  offroad 28  collision_at_fault 14  left_corridor_laterall 13  Waypoint 82 does not p 1  Waypoint 47 fails sani 1
- 회전각별: <20 0.741(100)   20~45 0.411(21)   45+ 0.179(29)
- 위치: `sweep_G10_042611`

### G11 — 자동제안: fine {'DAC': 0.3, 'EP': 2.0, 'NC': 3.0} coarse {'DAC': 0.3, 'EP': 2.0, 'NC': 0.3} imi c1.0/f0.02

```
coarse   exp DAC:0.3 EP:2 NC:0.3   imi 1.0
fine     exp DAC:0.3 EP:2 NC:3   imi 0.02
```

- **평균 0.6075   통과 98/150**
- 판정: pass 98  offroad 23  left_corridor_laterall 15  collision_at_fault 12  Waypoint 82 does not p 1  Waypoint 47 fails sani 1
- 회전각별: <20 0.763(100)   20~45 0.419(21)   45+ 0.206(29)
- 위치: `sweep_G11_053627`

### G12 — 자동제안: fine {'DAC': 0.3, 'EP': 2.0, 'NC': 2.0} coarse {'DAC': 0.3, 'EP': 2.0, 'NC': 0.3} imi c1.0/f0.02

```
coarse   exp DAC:0.3 EP:2 NC:0.3   imi 1.0
fine     exp DAC:0.3 EP:2 NC:2   imi 0.02
```

- **평균 0.6062   통과 98/150**
- 판정: pass 98  offroad 20  left_corridor_laterall 18  collision_at_fault 12  Waypoint 82 does not p 1  Waypoint 47 fails sani 1
- 회전각별: <20 0.770(100)   20~45 0.405(21)   45+ 0.188(29)
- 위치: `sweep_G12_064504`

### G13 — 자동제안: fine {'NC': 0.85, 'DAC': 2.5} coarse 1/1/1 imi c1.0/f0.02

```
coarse   exp 1/1/1   imi 1.0
fine     exp DAC:2.5 NC:0.85   imi 0.02
```

- **평균 0.6106   통과 101/150**
- 판정: pass 101  left_corridor_laterall 19  offroad 18  collision_at_fault 10  Waypoint 82 does not p 1  Waypoint 47 fails sani 1
- 회전각별: <20 0.770(100)   20~45 0.384(21)   45+ 0.226(29)
- 위치: `sweep_G13_075508`

### G14 — 자동제안: fine {'NC': 0.5, 'DAC': 1.7, 'EP': 0.3} coarse 1/1/1 imi c1.0/f0.02

```
coarse   exp 1/1/1   imi 1.0
fine     exp DAC:1.7 EP:0.3 NC:0.5   imi 0.02
```

- **평균 0.6144   통과 105/150**
- 판정: pass 105  left_corridor_laterall 17  offroad 15  collision_at_fault 11  Waypoint 82 does not p 1  Waypoint 47 fails sani 1
- 회전각별: <20 0.745(100)   20~45 0.474(21)   45+ 0.267(29)
- 위치: `sweep_G14_090544`
