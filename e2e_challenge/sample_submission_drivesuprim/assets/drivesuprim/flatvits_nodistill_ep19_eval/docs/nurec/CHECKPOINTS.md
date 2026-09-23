# 체크포인트 대조표

파일명이 여러 번 바뀌었고 같은 가중치가 두세 이름으로 존재한다. 어느 것이 어느
결과를 낸 것인지는 **파일 크기**로 대조하는 것이 가장 확실하다 — 이름은 복사하며
바뀌지만 바이트 수는 안 바뀐다.

`inspect_checkpoint.py` 는 텐서 모양만 보므로 100% 로드되어도 "이게 그 결과를 낸
체크포인트인가" 는 답해주지 못한다. 그것이 이 표가 있는 이유다.

---

## NAVSIM navtest 결과와 체크포인트

| 모델 | navtest EPDMS | 크기 (B) | 원본 파일명 | assets 사본 | agent config |
|---|---:|---:|---|---|---|
| ResNet34 | — | 593,591,742 | — | `drivesuprim_r34.ckpt` | `drivesuprim_agent_r34` |
| V2-99 (VoVNet) | — | 1,370,514,646 | — | `drivesuprim_vov.ckpt` | `drivesuprim_agent_vov` |
| ViT-L | — | 5,127,157,698 | — | `drivesuprim_vit.ckpt` | `drivesuprim_agent_vit` |
| **ConvNeXt V2-Tiny + Filter** | **0.7905 (79.1)** | 895,096,136 | `epoch=01-step=5320.ckpt` | `epoch=01-step=5320.ckpt` | `..._vov_v2_cnx_stage3` |
| **ResNet50 + Filter** | **0.7989 (79.9)** | 835,041,665 | `epoch=03-step=2435.ckpt` | `drivesuprim_v2_resnet50_stage3_epoch03.ckpt` | `..._vov_v2_r50_stage3` |
| **ViT-s PlanOnly** | — | 711,401,335 | `epoch=03-step=2656.ckpt` | (워크스페이스 최상위) | `..._vov_v2_vits_planonly_scratch` |

### 함정 두 가지

**`drivesuprim_v2_convnextv2_stage3_epoch04.ckpt` (895,097,413 B) 는 79.1 을 낸
파일이 아니다.** 크기가 1,277 B 다르고, 79.1 은 stage3 **epoch 1** 인
`epoch=01-step=5320.ckpt` 가 냈다. 드라이버의 `checkpoint_default` 매핑이
epoch04 를 가리키고 있어 혼동하기 쉽다.

근거: `migration_package/DriveSuprim/exp_v2/evaluation/v2cnx_s3ep1_CTRL_GATEON/`
의 CSV 가 `NC=0.9618 2FC=0.5452 score=0.7905` 로 정확히 일치하고, 그
`overrides.yaml` 이 `epoch=01-step=5320.ckpt` 를 가리킨다.

**`drivesuprim_v2_resnet50_stage3_epoch03.ckpt` 는 `epoch=03-step=2435.ckpt` 와
같은 파일이다.** 둘 다 835,041,665 B. 이름만 바꿔 복사한 것.

---

## NAVSIM navtest 참조 수치 (79.1 / 79.9)

| | ConvNeXt V2-Tiny + Filter | ResNet50 + Filter |
|---|---:|---:|
| EPDMS | 79.1 | 79.9 |
| NC | 0.9618 | 0.9583 |
| DAC | 0.9480 | 0.9528 |
| DDC | 0.9866 | 0.9887 |
| TL | 0.9877 | 0.9896 |
| EP | 0.9498 | 0.9326 |
| TTC | 0.9555 | 0.9524 |
| LK | 0.9612 | 0.9588 |
| HC | 0.9761 | 0.9802 |
| 2FC | 0.5452 | 0.6425 |

NuRec 에서 낸 수치와 직접 비교하면 안 된다. 데이터셋도 컨트롤러도 다르다.

---

## 체크포인트마다 박혀 있는 것

| | vocab | 카메라 K | 컨트롤러 |
|---|---|---|---|
| r34 / vov / vit | **8192** (`test_8192_kmeans.npy`) | NAVSIM | LQR |
| cnx / r50 / vits | **4096** (`test_4096_kmeans.npy`) | NAVSIM 412/366 | LQR |
| 2026-08-26 이후 NuRec 학습분 | **4096** (`nurec_train_kmeans_4096x40x3.npy`) | 377 | MPC |

`NUREC_VOCAB=navsim` 이 앞의 둘을, `nurec` 이 마지막을 고른다. vocab 크기가
8192 인 것들은 `NUREC_VOCAB_PATH` 를 직접 지정해야 한다 — 스위치는 4096 두 개만
구분한다. 자세한 것은 `PAIRING.md`.

---

## 확인 방법

```bash
# 크기로 대조 (가장 확실)
stat -c %s <ckpt>

# 구조가 맞는지
source setup_nurec.sh
NUREC_VOCAB_PATH=<vocab> $PYTHON_BIN scripts/nurec/inspect_checkpoint.py \
  --agent <config> --checkpoint <ckpt>
# loaded 100% / missing 0 / mismatched 0 이어야 한다
```

`mismatched` 가 2개 뜨면 대개 vocab 크기가 틀린 것이다 (8192 모델에 4096 을 준 경우).
