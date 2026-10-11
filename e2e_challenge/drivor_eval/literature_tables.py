# SPDX-License-Identifier: Apache-2.0
"""DrivoR (ours / original) next to every published AlpaSim result, one table per metric convention.

    .venv/bin/python e2e_challenge/drivor_eval/literature_tables.py \
        --protocols runs/drivor-protocols-20261009 --out runs/drivor-literature-20261012

Our numbers come from our own 441x3 rollouts, recomputed under each paper's
definition (rescore_protocols.py, extra_protocols.py, and the AlpaSim default
aggregation in results-summary.json). Published numbers are copied verbatim from
the cited tables (checked against the paper text on 2026-10-12); percentages are
shown as %, fractions in the papers converted x100. Only OPTED and ECO use the
same 441 scenes; every other published row is on a different scene set, which
the "scenes" column says.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

# training category: Z zero-shot (no PAI-AV / NuRec / AlpaSim data), P trained on data of the
# evaluation vehicle platform (PhysicalAI-AV or NVIDIA internal data of that platform),
# C closed-loop fine-tuned inside AlpaSim on NuRec scenes, "?" not stated
OURS_TRAIN = "Z (NAVSIM navtrain)"
REF_LABEL = {"alpamayo1": "Alpamayo 1 (공식 기준 실행)", "vavam-linear": "VaVAM 선형 MPC (공식 기준 실행)",
             "vavam-nonlinear": "VaVAM 비선형 MPC (공식 기준 실행)"}
REF_TRAIN = {"alpamayo1": "P", "vavam-linear": "Z", "vavam-nonlinear": "Z"}

GROUNDACT = "GroundAct (arXiv 2610.07521) Tab. 2"
GROUNDACT_ROWS = [  # progress, at-fault coll, offroad, wrong lane (fractions), max dist GT (m)
    ("FSDrive† (PAI-AV 140.5K 미세조정)", "P", 0.604, 0.193, 0.271, 0.210, 2.577),
    ("Alpamayo-R1", "P", 0.682, 0.156, 0.189, 0.295, 2.781),
    ("AutoVLA† (PAI-AV 140.5K 미세조정)", "P", 0.706, 0.118, 0.185, 0.211, 2.544),
    ("LCDrive", "?", 0.639, 0.259, 0.263, 0.189, 2.825),
    ("Alpamayo 1.5", "P", 0.682, 0.166, 0.155, 0.297, 2.843),
    ("SpanVLA† (PAI-AV 140.5K 미세조정)", "P", 0.696, 0.156, 0.184, 0.207, 2.896),
    ("Alpamayo 2 Super", "P", 0.713, 0.084, 0.147, 0.220, 2.799),
    ("GroundAct", "P", 0.752, 0.105, 0.153, 0.207, 2.511),
]
COMPACT = "COMPACT-VA (arXiv 2606.07464) Tab. II"
COMPACT_ROWS = [  # progress, at-fault coll, offroad, wrong lane, km per at-fault incident
    ("Alpamayo 기반 baseline (2 s, 18 imgs)", "P", 0.73, 0.04, 0.28, 0.23, 0.49),
    ("COMPACT-VA (5 s, 40 imgs)", "P", 0.71, 0.05, 0.27, 0.23, 0.48),
]
# Alpamayo-R1 convention: CE all, CE at-fault, off-road, progress (%), score all, score at-fault (km)
AR1_ROWS = [
    ("Alpamayo-R1-10B", "P", "920 (NuRec)", None, 4.0, 16.0, None, None, 0.72, "Alpamayo-R1 (arXiv 2511.00088) Tab. 10, ±생략"),
    ("Alpamayo-R1-0.5B", "P", "920 (NuRec)", None, 9.0, 19.0, None, None, 0.35, "Alpamayo-R1 Tab. 10"),
    ("Alpamayo-R1 (재현)", "P", "916 (NuRec 26.02)", 19.0, 6.0, 17.0, 67.0, 0.36, 0.58, "Qwen-Drive-1.0 (arXiv 2609.00111) Tab. 7"),
    ("Alpamayo-1.5 (재현)", "P", "916 (NuRec 26.02)", 37.0, 11.0, 16.0, 59.0, 0.23, 0.45, "Qwen-Drive-1.0 Tab. 7"),
    ("DriveWAM", "?", "916 (NuRec 26.02)", 56.0, 5.0, 8.0, 35.0, 0.10, 0.53, "Qwen-Drive-1.0 Tab. 7"),
    ("SimWAM", "?", "916 (NuRec 26.02)", 35.0, 22.0, 19.0, 62.0, 0.22, 0.30, "Qwen-Drive-1.0 Tab. 7"),
    ("Qwen-Drive-1.0-SFT", "P", "916 (NuRec 26.02)", 38.0, 12.0, 24.0, 54.0, 0.16, 0.27, "Qwen-Drive-1.0 Tab. 7"),
    ("Qwen-Drive-1.0-RL", "P", "916 (NuRec 26.02)", 41.0, 11.0, 12.0, 48.0, 0.16, 0.37, "Qwen-Drive-1.0 Tab. 7"),
    ("Qwen-RobotNav-4B", "Z (nuScenes·OpenScene)", "920 (NuRec)", None, 22.0, 34.0, None, None, 0.15,
     "Qwen-RobotNav (arXiv 2606.18112) Tab. 9"),
    ("Qwen-RobotNav-8B", "Z (nuScenes·OpenScene)", "920 (NuRec)", None, 22.0, 27.0, None, None, 0.17, "Qwen-RobotNav Tab. 9"),
    ("DriveHeron 2B", "Z (일본 사내 데이터)", "909 (public_2507 추정)", None, 12.0, 10.0, None, None, 0.75,
     "Turing 블로그(zenn.dev, 2026-05-27), 논문 아님"),
    ("Alpamayo-1 10B (재현)", "P", "909", None, 7.0, 30.0, None, None, 0.44, "Turing 블로그"),
    ("Alpamayo-1.5 10B (재현)", "P", "909", None, 3.0, 26.0, None, None, 0.51, "Turing 블로그"),
    ("Alpamayo 1.5 (모델 카드)", "P", "913", None, None, None, None, None, 1.37, "HF 모델 카드, 전체/책임 구분 미기재(08-07 전 0.81)"),
    ("Alpamayo 2 Super (모델 카드)", "P", "913", None, None, None, None, None, 1.50, "HF 모델 카드, 전체/책임 구분 미기재"),
]
ROAD = "RoaD (arXiv 2512.01993) Tab. 3"
ROAD_ROWS = [  # driving score (km), collision, offroad (fractions), distance (m)
    ("RoaD 미세조정", "P+C (AlpaSim 8,251 장면)", 0.6300, 0.0239, 0.2098, 147.2),
    ("재렌더링 전문가 궤적 미세조정", "P+C", 0.4985, 0.0464, 0.2583, 151.9),
    ("BC 추가 학습", "P", 0.4215, 0.0627, 0.2783, 143.7),
    ("기본 모델 (BC 사전학습, 2만 h)", "P", 0.4443, 0.0525, 0.2833, 149.0),
]
COLLISION_ROWS = [  # model, train, scenes, collision, front, lateral, rear, offroad (%), source
    ("Alpamayo 1.5", "P", "574 (NuRec, WAM 학습 장면 제외)", 6.9, 1.0, 0.6, 5.3, None, "OmniDreams (arXiv 2606.03159) §7.2 본문"),
    ("OmniDreams 미세조정 정책 (~2B)", "P", "574", 4.2, 0.9, 0.4, 3.0, None, "OmniDreams §7.2 본문"),
    ("Alpamayo 1.5 (4 cam)", "P", "140 (내부 추정), 6회", 14, None, None, None, 7, "Instant NuRec (arXiv 2607.14203) Fig. 8 막대값"),
    ("Alpamayo-R1", "P", "140, 6회", 20, None, None, None, 11, "Instant NuRec Fig. 8"),
    ("VaVAM", "Z", "140, 6회", 40, None, None, None, 12, "Instant NuRec Fig. 8"),
]
FLASH = "FlashDrive (arXiv 2608.12932) Tab. 3"
FLASH_ROWS = [  # D_traj, D_loc (m), P_rel, collision, offroad, wrong lane (fractions), plan dev
    ("Alpamayo 1.5", "P", 20.0, 35.2, 0.85, 0.19, 0.41, 0.45, 0.24),
    ("Alpamayo 1.5 + FlashDrive", "P", 22.4, 32.8, 0.85, 0.15, 0.32, 0.51, 0.16),
]
NAVSAFE_ROWS = [  # DS, SR (%) — NavSafe-∞ Tab. 7 v2, mean±std over 3 seeds
    ("PDM-Closed (특권 정보, 학습 없음)", "90.17±1.00", "81.31±1.03"),
    ("GTRS-Dense–V2-99 (SimScale)", "73.54±0.84", "61.79±1.56"),
    ("SparseDriveV2", "65.26±1.72", "48.81±3.04"),
    ("**DrivoR (공개 체크포인트)**", "**64.74±1.60**", "**50.71±0.71**"),
    ("GTRS-Dense–V2-99", "64.27±0.76", "49.52±1.15"),
    ("DrivoR (SimScale, 134k)", "57.66±1.32", "42.50±0.36"),
    ("LTF", "53.04±1.92", "27.74±1.76"),
    ("DiffusionDrive", "44.77±0.75", "20.95±0.55"),
]


NOTES = """## 8. 공정한 비교를 위한 참고 사항

### 8.1 학습 조건: 제로샷·퓨샷·도메인 내 학습

| 범주 | 뜻 | 해당 행 | 비교할 때 |
|---|---|---|---|
| **Z 제로샷** | 평가 도메인 데이터(PAI 차량 영상, NuRec 장면, AlpaSim 폐루프)를 학습에 쓰지 않음 | **Ours·DrivoR original**(NAVSIM navtrain), VaVAM(OpenDV·nuPlan·nuScenes), LTFv6(nuPlan·CARLA), DiffusionDrive(NAVSIM, 체크포인트 미기재), Qwen-RobotNav(nuScenes·OpenScene = NAVSIM 원천 데이터), DriveHeron(일본 사내 데이터), NavSafe-∞의 모든 정책 | 우리와 같은 선의 비교 대상. 같은 441 장면에선 1절의 VaVAM·LTFv6·DiffusionDrive, NAVSIM 계열 제로샷은 3절 Qwen-RobotNav(장면 세트 다름) |
| **Z + 학습 없는 후처리** | 제로샷 모델에 궤적 후처리만 더함 | ECO 행(후처리 가중치는 HUGSIM에서 선택) | 우리 ours와 가장 비슷한 설정(ours도 모델 선택을 HUGSIM에서 했고 441로는 아무것도 조정하지 않음) |
| **P 같은 차량 플랫폼 데이터** | PhysicalAI-AV(평가 차량과 같은 플랫폼) 또는 그 플랫폼의 NVIDIA 내부 데이터로 학습 | Alpamayo R1·1.5·2(8만~11.5만 h; HF 카드 데이터셋 태그에 PAI-AV·NuRec가 있으나 학습 사용 여부는 본문 미기재), Qwen-Drive(PAI-AV 138만 샘플 + RL 1.5만 장면, 개루프), GroundAct·FSDrive†·AutoVLA†·SpanVLA†(PAI-AV 14.05만 샘플), FlashDrive(PAI-AV 4천 클립), COMPACT-VA, RoaD 기본 모델(2만 h, 공개 PAI 1,700 h 포함) | 카메라 배치·차량·도시 분포가 평가와 같아 도메인 이점이 크다. 제로샷과 같은 선에서 비교하지 않는다 |
| **C 폐루프 미세조정** | AlpaSim 안에서 NuRec 장면 롤아웃으로 미세조정 | OPTED·BC·RoaD-recovery 행(NuRec 1,669 장면, 441과 분리, 2 epoch), RoaD(3DGS 8,251 장면) | 평가 시뮬레이터의 렌더링·제어·채점 자체에 적응한 결과 |
| **퓨샷** | 평가 도메인 샘플을 소수만 사용 | AlpaSim 결과를 낸 논문 중 명시적 퓨샷 설정은 없다. 가장 가까운 것은 LaPla(2만 프레임 추가 미세조정, AlpaSim 수집인지 불명확, 60 장면 평가) | 퓨샷 비교가 필요하면 설정(샘플 수·장면 분리)을 직접 정해야 한다 |
| **학습·평가 장면 겹침** | 평가 장면이 학습에 들어갔을 가능성 | OneFixer(겹침 명시), Qwen-Drive(916 장면 겹침 미기재, 개루프 644 분할은 겹침 인정), Alpamayo 카드(NuRec 데이터셋 태그), 대회 참가 모델(441 curated_val을 학습에 쓰는 경우 많음 — 우리 axe 계열 포함, DrivoR는 해당 없음) | 겹침이 있거나 확인되지 않은 행은 일반화 성능으로 해석하지 않는다 |

### 8.2 평가 조건 차이

| 항목 | 우리 (DrivoR) | 논문들 | 영향 |
|---|---|---|---|
| 장면 세트 | 441 curated_val(대회 공개 분할) | OPTED·ECO 441; GroundAct·Alpamayo 카드 913(public_2601); Qwen-Drive 916(26.02); Alpamayo-R1·RoaD·Qwen-RobotNav 920; COMPACT-VA 910; 그 밖에 50~574 부분집합 | 난이도가 달라 절대값을 비교할 수 없다. 같은 Alpamayo-R1의 책임 AlpaSim score가 0.72(920, 원논문) / 0.58(916, Qwen-Drive 재현) / 0.44(909, Turing 재현)로 출처마다 다르다 |
| 장면당 롤아웃 | 3 | OPTED·RoaD 3, GroundAct 1, OpenLongTail 2, Instant NuRec 6, 다수 미기재 | 1회 평가는 분산이 크다(axe-v9 재실행: 440 클립 중 38개의 통과/실패가 뒤바뀜, 평균은 0.001 차이) |
| 제어기 | 기본 비선형 MPC(대회 기본값) | 대부분 기본 MPC. ECO·공식 기준 일부는 선형 MPC(대회에 없는 옵션). OPTED 일부 마찰원 제약 MPC. RoaD는 자체 제어기(200 ms 지연·자세 잡음). NavSafe-∞는 LQR | 제어기만 바꿔도 점수가 크게 바뀐다(공식 VaVAM: 비선형 11.0 → 선형 31.4) |
| 집계 규칙 | 각 절 설명대로 같은 롤아웃을 재채점 | 후방 추돌 포함 여부, 4 m 이탈이 실패인지 그 뒤 사건만 제외인지, 측면 접촉 포함 여부(RoaD 본문·부록 불일치)가 논문마다 다르다 | 같은 롤아웃도 충돌률이 크게 달라진다(ours: 책임 11.1%, 후방 포함 36.4%, 원지표 44.3%) |
| 렌더러·초기 구간 | dev 프리셋: harmonizer 켬, 처음 1.7 s 강제 GT 동안에도 드라이버 호출 | 공식 리더보드(ec2)는 harmonizer 끔·강제 GT 동안 호출 생략. 논문 대부분 미기재. NuRec 버전(25.07/26.01/26.02/26.04)도 다름 | 렌더링 품질과 초기 상태가 달라진다 |
| 카메라 | PAI 6대 → nuPlan 4뷰 변환(전·좌·우는 회전 재투영, 후방은 두 카메라를 지면+15 m 원통 프록시로 접합) | Alpamayo 4대(전방 광각·망원, 좌우 교차), Alpamayo 2 Super 6대, VaVAM 전방 1대, RoaD 2대 | 학습 때와 다른 카메라 배치·위치는 제로샷 모델에 불리하다 |
| 추론 지연 | 동기식 시뮬레이터라 점수에 반영되지 않음(ours 전처리 ~410 ms + 모델 ~68 ms, FP32) | Qwen-Drive 1181 ms, Alpamayo 1.5 940 ms(OPTED), FlashDrive 1150 → 463 ms | 실시간(10 Hz) 가능 여부는 점수와 별도로 적어야 한다 |
| 우리 어댑터 | FP32 추론, ego 가속도 ±10 m/s² 경계(시뮬 시작 스파이크 대응), 명령은 공식 규칙(3 m / 20 m), 441로 아무것도 조정하지 않음 | — | ours·original에 똑같이 적용돼 둘 사이 비교는 공정하다 |

### 8.3 논문에 쓸 수 있는 비교

1. **같은 조건 직접 비교**: 1절의 우리 두 모델 ↔ 공식 기준 실행(Alpamayo 1, VaVAM). 같은 dev 계약, 같은 441×3이다. OPTED·ECO 행은 장면·지표가 같지만 모델 연결 방식·렌더러 설정이 다를 수 있어 참고값이다(같은 VaVAM이 OPTED 3.9%, ECO·공식 11.0%).
2. **제로샷끼리(1절, 기본 MPC)**: ours 33.4 > DrivoR 29.7 > DiffusionDrive 28.9(ECO) > LTFv6 26.5(OPTED) > VaVAM 11.0. 학습 없는 후처리를 더한 DiffusionDrive + ECO는 35.2. 논문 행은 장면별 값이 없어 유의성 검정은 하지 못했다. ours − original은 우리 데이터로 +3.7점 [+1.2, +6.3].
3. **장면 세트가 다른 2·3절**의 913·916·920 장면 수치와 직접 비교하려면 DrivoR를 같은 장면 세트에서 다시 돌려야 한다(public_2601 913 장면 기준, 두 모델 동시·GPU 8장으로 1회 약 11시간, 3회 약 32시간 예상).
4. **원지표(6절)**는 기록 재생 뒤차의 관통 현상 때문에 느린 정책의 충돌률이 부풀려지므로, 쓸 경우 각주가 필요하다.

### 8.4 수치는 있지만 표에서 뺀 출처 (정의 불명확·장면 세트 상이)

- OpenLongTail (arXiv 2607.09655): 53개 롱테일 사건, AlpaSim score·충돌률 정의 미기재. Alpamayo R1 평균 AS 0.534 / CR 58.8%.
- LaPla (arXiv 2609.04070): 60 장면, 책임 충돌·도로 이탈·경로 완료·성공률. LaPla SR 61.67%.
- TRUST (arXiv 2610.00601): public_2601 중 888 장면, 충돌·근접(0.5 m 이내)·도로 이탈, 폐루프 ADE. Alpamayo 1.5 충돌 16.22%.
- Latency Analysis of Alpamayo 1 (arXiv 2605.08975): 첫 실패까지 거리(DTF) 131.6~137.1 m, 폐루프 장면 수 미기재.
- OneFixer (arXiv 2609.32175): 사내 60 장면×5 seed, 학습 장면과 겹침 명시.
- ReasonBreak (arXiv 2605.29114): 50 클립 적대적 공격 성공률.
- OpenReview 초록만 확인(표 미확인): STAR-Drive(913 장면, Alpamayo 1.5 Score_AF 59.63→65.70), DriveVLA-Cache(913, 70.8→73.4), COSP(장면 점수 0.750→0.828), TacitVLA(“PAI 공개 분할” 장면 점수 52.5%, 441로 추정).
- 원본 DrivoR를 AlpaSim에서 평가한 논문은 찾지 못했다(2026-10-12 기준).
"""


def f(x, d=1, scale=1.0):
    if x is None:
        return "–"
    return x if isinstance(x, str) else f"{x * scale:.{d}f}"   # published values may carry their own ±


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocols", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    prot = json.loads((a.protocols / "protocols.json").read_text())["tables"]
    extra = json.loads((a.protocols / "extra_protocols.json").read_text())
    default = json.loads((a.protocols / "alpasim_default_aggregation.json").read_text())
    pub441 = json.loads((HERE / "published_alpasim441.json").read_text())["rows"]
    ours = {"ours": "**Ours (DrivoR + Z_safe)**", "original": "**DrivoR (original V1)**"}
    S441 = "441 (curated_val), 3회"
    md = ["# AlpaSim 폐루프 결과: 평가 기준별 비교 (DrivoR ours / original 포함)", "",
          "우리 두 모델은 같은 441×3 롤아웃을 각 논문의 정의대로 다시 채점했다. 논문 행은 표에 보고된 값을 그대로 옮겼다(직접 실행 아님). "
          "**같은 441 장면을 쓴 논문은 OPTED·ECO뿐**이고, 나머지 논문 행은 장면 세트가 달라 수치를 직접 비교하면 안 된다(‘장면’ 열). "
          "학습 열: **Z** 제로샷(평가 도메인 PAI-AV·NuRec·AlpaSim 데이터 미사용), **P** 평가 차량 플랫폼 데이터(PhysicalAI-AV 또는 같은 플랫폼의 NVIDIA 내부 데이터)로 학습, "
          "**C** AlpaSim 안에서 NuRec 장면으로 폐루프 미세조정, **?** 미기재.", ""]

    # 1. challenge
    t = prot["challenge"]
    md += ["## 1. 대회 방식 — 장면 점수 (AlpaSim e2e challenge · OPTED · ECO) — 유일하게 같은 441 장면", "",
           "첫 충돌(후방 포함)·도로 이탈·GT 4 m 이탈에서 집계 중단. 책임 충돌(정면·측면)·도로 이탈·옆으로 4 m 이탈이면 0점, 아니면 min(진행률/0.8, 1).", "",
           "| 모델 | 학습 | 장면·롤아웃 | 장면 점수(%)↑ | 책임 충돌(%)↓ | 도로 이탈(%)↓ | 4 m 이탈(%)↓ | 진행률(%)↑ | 책임 사고당 km↑ | 출처 |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for k, label in ours.items():
        s = t[k]
        md.append(f"| {label} | {OURS_TRAIN} | {S441} | {f(s['scene_score'])} | {f(s['collision_at_fault'])} | {f(s['offroad'])} | "
                  f"{f(s['corridor_exit'])} | {f(s['progress'])} | {f(s['km_per_event_at_fault'], 2)} | 직접 평가 |")
    for k, label in REF_LABEL.items():
        s = t[label.replace(" (공식 기준 실행)", " (기준)")]
        md.append(f"| {label} | {REF_TRAIN[k]} | {S441} | {f(s['scene_score'])} | {f(s['collision_at_fault'])} | {f(s['offroad'])} | "
                  f"{f(s['corridor_exit'])} | {f(s['progress'])} | {f(s['km_per_event_at_fault'], 2)} | NVlabs/alpasim 공개 결과 파일 |")
    code = {"제로샷": "Z", "AlpaSim 미세조정": "Z+C", "PAI-AV": "P", "같은 차량 플랫폼": "P", "학습 없는 후처리": "Z"}
    for r in pub441:
        train = next((v for key, v in code.items() if key in r["train_domain"]), "?")
        name = r["model"] + (" — 선형 MPC(대회에 없는 옵션)" if r["controller"] == "linear MPC" else "")
        md.append(f"| {name} | {train} | 441, {('3회' if r['source'] == 'OPTED' else '미기재')} | {f(r['scene_score_pct'])} | "
                  f"{f(r['atfault_collision_pct'])} | {f(r['offroad_pct'])} | {f(r['corridor_exit_pct'])} | {f(r['progress_pct'])} | "
                  f"{f(r['km_per_incident'], 2)} | {r['source']} |")

    # 2. AlpaSim default aggregation
    md += ["", "## 2. AlpaSim 기본 집계 지표 (GroundAct · COMPACT-VA)", "",
           "첫 충돌(종류 무관)·도로 이탈·GT 4 m 이탈에서 집계 중단(AlpaSim 기본 프로토콜). 진행률은 최종 위치를 GT 경로에 투영한 비율, "
           "책임 충돌은 정면·측면, 역주행은 차로 방향과 120° 이상 차이, GT 거리는 GT 궤적까지 최대 거리의 평균.", "",
           "| 모델 | 학습 | 장면·롤아웃 | 진행률(%)↑ | 책임 충돌(%)↓ | 도로 이탈(%)↓ | 역주행 차로(%)↓ | GT 최대 거리(m) | 책임 사고당 km↑ | 출처 |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for k, label in {**ours, **{r: REF_LABEL[r] for r in REF_LABEL}}.items():
        s = default[k]
        md.append(f"| {label} | {OURS_TRAIN if k in ours else REF_TRAIN[k]} | {S441} | {f(s['progress'], 1, 100)} | "
                  f"{f(s['collision_at_fault'], 1, 100)} | {f(s['offroad'], 1, 100)} | {f(s['wrong_lane'], 1, 100)} | "
                  f"{f(s['dist_to_gt_trajectory'], 2)} | {f(s['avg_dist_between_incidents_at_fault'], 2)} | "
                  f"{'직접 평가' if k in ours else '공식 결과 파일'} |")
    for name, tr, p, c, o, w, d in GROUNDACT_ROWS:
        md.append(f"| {name} | {tr} | 913 (public_2601), 1회 | {f(p, 1, 100)} | {f(c, 1, 100)} | {f(o, 1, 100)} | {f(w, 1, 100)} | "
                  f"{f(d, 2)} | – | {GROUNDACT} |")
    for name, tr, p, c, o, w, km in COMPACT_ROWS:
        md.append(f"| {name} | {tr} | 910 (NuRec), 미기재 | {f(p, 0, 100)} | {f(c, 0, 100)} | {f(o, 0, 100)} | {f(w, 0, 100)} | – | "
                  f"{f(km, 2)} | {COMPACT} |")

    # 3. Alpamayo-R1
    t = prot["alpamayo"]
    md += ["", "## 3. Alpamayo-R1 방식 — 근접충돌·AlpaSim score (Alpamayo-R1 · Qwen-Drive · Qwen-RobotNav)", "",
           "첫 근접충돌(후방 포함) 또는 도로 이탈에서 종료, GT에서 4 m 넘게 벗어난 뒤의 사건은 제외. ‘책임’은 후방 근접충돌 제외. "
           "AlpaSim score = 사건 사이 평균 주행 거리(km). 근접충돌은 시뮬레이터 충돌로 대응시켰다(우리 행).", "",
           "| 모델 | 학습 | 장면 | 근접충돌 전체(%)↓ | 근접충돌 책임(%)↓ | 도로 이탈(%)↓ | 진행률(%)↑ | Score 전체(km)↑ | Score 책임(km)↑ | 출처 |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for k, label in ours.items():
        s = t[k]
        md.append(f"| {label} | {OURS_TRAIN} | {S441} | {f(s['collision_any'])} | {f(s['collision_at_fault'])} | {f(s['offroad'])} | "
                  f"{f(s['progress'])} | {f(s['km_per_event_all'], 2)} | {f(s['km_per_event_at_fault'], 2)} | 직접 평가 |")
    for k, label in REF_LABEL.items():
        s = t[label.replace(" (공식 기준 실행)", " (기준)")]
        md.append(f"| {label} | {REF_TRAIN[k]} | {S441} | {f(s['collision_any'])} | {f(s['collision_at_fault'])} | {f(s['offroad'])} | "
                  f"{f(s['progress'])} | {f(s['km_per_event_all'], 2)} | {f(s['km_per_event_at_fault'], 2)} | 공식 결과 파일, 우리가 재채점 |")
    for name, tr, sc, ce, cef, o, p, sa, sf, src in AR1_ROWS:
        md.append(f"| {name} | {tr} | {sc} | {f(ce)} | {f(cef)} | {f(o)} | {f(p)} | {f(sa, 2)} | {f(sf, 2)} | {src} |")

    # 4. RoaD
    t, e = prot["road"], extra["road_main"]
    md += ["", "## 4. RoaD 방식 — 후방 추돌 무시 (RoaD)", "",
           "뒤차가 일으킨 후방 추돌은 사고가 아님. 사고(책임 충돌·도로 이탈) 또는 GT 4 m 이탈에서 집계 중단, driving score = 사고 사이 km. "
           "RoaD는 측면 접촉을 본문에선 제외, 부록 B.1에선 포함해 정의가 엇갈려 우리 행은 두 가지로 계산했다. "
           "후방으로 시작된 접촉 구간 전체를 무시(기록 재생 뒤차가 ego를 관통해 같은 접촉이 정면으로 바뀌는 현상 대응).", "",
           "| 모델 | 학습 | 장면 | Driving score(km)↑ | 충돌(%)↓ | 도로 이탈(%)↓ | 주행 거리(m)↑ | 출처 |",
           "|---|---|---|---:|---:|---:|---:|---|"]
    for k, label in ours.items():
        s = t[k]
        md.append(f"| {label} — 측면 포함(부록 B.1) | {OURS_TRAIN} | {S441} | {f(s['km_per_event_at_fault'], 2)} | "
                  f"{f(s['collision_at_fault'])} | {f(s['offroad'])} | {f(s['dist_m'])} | 직접 평가 |")
        s = e[k]
        md.append(f"| {label} — 측면 제외(본문) | {OURS_TRAIN} | {S441} | {f(s['km_per_incident'], 2)} | {f(s['collision'])} | "
                  f"{f(s['offroad'])} | {f(s['dist_m'])} | 직접 평가 |")
    for name, tr, ds, c, o, d in ROAD_ROWS:
        md.append(f"| {name} | {tr} | 920 (NuRec), 3회 | {f(ds, 2)} | {f(c, 1, 100)} | {f(o, 1, 100)} | {f(d)} | {ROAD} |")
    md.append("")
    md.append("공식 기준 실행(Alpamayo 1, VaVAM)은 시점별 지표가 공개되지 않아 4·5·6절 방식으로 다시 계산할 수 없다.")

    # 5. collision breakdown
    e = extra["omnidreams"]
    md += ["", "## 5. 충돌 유형별 (OmniDreams · Instant NuRec) — 후방 추돌 포함", "",
           "GT 궤적 4 m 이내에서 일어난 충돌·도로 이탈만 집계, 20 s 전체(첫 사건 후에도 계속). 충돌은 정면·측면·후방의 합. "
           "우리 행은 롤아웃별 첫 접촉의 유형으로 분류해 세 유형의 합이 충돌률과 같다.", "",
           "| 모델 | 학습 | 장면 | 충돌(%)↓ | 정면 | 측면 | 후방 | 도로 이탈(%)↓ | 출처 |", "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for k, label in ours.items():
        s = e[k]
        md.append(f"| {label} | {OURS_TRAIN} | {S441} | {f(s['collision'])} | {f(s['front'])} | {f(s['lateral'])} | {f(s['rear'])} | "
                  f"{f(s['offroad'])} | 직접 평가 |")
    for name, tr, sc, c, fr, la, re, o, src in COLLISION_ROWS:
        md.append(f"| {name} | {tr} | {sc} | {f(c)} | {f(fr)} | {f(la)} | {f(re)} | {f(o)} | {src} |")

    # 6. raw
    e = extra["raw"]
    md += ["", "## 6. 집계 중단 없는 원지표 (FlashDrive)", "",
           "AlpaSim 지표별 시간 집계(사건은 최대, 진행률은 최소, 거리는 최대, 계획 편차는 평균)를 롤아웃 전체에 적용, 사건 뒤에도 계속 집계·4 m 규칙 없음. "
           "**주의**: 집계를 끊지 않으므로 후방 추돌 뒤 기록 재생 뒤차가 ego를 관통하며 정면·측면 충돌로 다시 잡힌다. 느린 정책일수록 충돌률이 부풀려진다.", "",
           "| 모델 | 학습 | 장면 | GT 궤적 거리(m)↓ | GT 위치 거리(m)↓ | 상대 진행률↑ | 충돌(%)↓ | 도로 이탈(%)↓ | 역주행 차로(%)↓ | 계획 편차↓ | 출처 |",
           "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for k, label in ours.items():
        s = e[k]
        md.append(f"| {label} | {OURS_TRAIN} | {S441} | {f(s['dist_to_gt_trajectory'])} | {f(s['dist_to_gt_location'])} | "
                  f"{f(s['progress_rel'], 2)} | {f(s['collision_any'])} | {f(s['offroad'])} | {f(s['wrong_lane'])} | "
                  f"{f(s['plan_deviation'], 2)} | 직접 평가 |")
    for name, tr, dt, dl, pr, c, o, w, pd_ in FLASH_ROWS:
        md.append(f"| {name} | {tr} | 무작위 100 클립, 1회 | {f(dt)} | {f(dl)} | {f(pr, 2)} | {f(c, 0, 100)} | {f(o, 0, 100)} | "
                  f"{f(w, 0, 100)} | {f(pd_, 2)} | {FLASH} |")

    # 7. other simulator
    md += ["", "## 7. (참고) 다른 시뮬레이터에서의 DrivoR — NavSafe-∞", "",
           "AlpaSim이 아니다: nuPlan 테스트 로그 280개 사건 장면을 NuRec로 재구성해 Isaac Sim에서 실행(LQR 제어, 반응형 IDM 뒤차, harmonizer 사용, "
           "seed 3개). 모든 정책이 NAVSIM 공개 가중치의 제로샷. 원본 DrivoR만 평가됐고 우리 모델(ours)은 없다. DS는 CARLA 방식 driving score, SR은 성공률.", "",
           "| 모델 | DS↑ | SR(%)↑ |", "|---|---:|---:|"]
    md += [f"| {n} | {ds} | {sr} |" for n, ds, sr in NAVSAFE_ROWS]
    md += ["", "출처: NavSafe-∞ (arXiv 2609.26618 v2) Tab. 7, 전체 22행 중 발췌.", "", NOTES]
    (a.out).mkdir(parents=True, exist_ok=True)
    body = "\n".join(md) + "\n"
    (a.out / "tables.md").write_text(body)
    print(body)


if __name__ == "__main__":
    main()
