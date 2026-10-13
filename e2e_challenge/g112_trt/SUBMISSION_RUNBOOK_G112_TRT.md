# vits512_g112_kv28 + TensorRT 제출 런북

작성 2026-10-13. **이미지 빌드·로컬 검증까지 끝났고, push와 submit은 하지 않았다.**
이 문서는 제출을 실행하는 사람이 읽는다. 경로는 저장소 루트
(`/home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim`) 기준이다.

> **제출 전 선행 조건: 주최측 TensorRT 허가.** 대회 규정에 TensorRT 언급이 없어
> NVlabs/alpasim#205 로 문의했고, 2026-10-13 기준 답변이 없다(댓글 0). 허가 전 제출은
> 규정 해석 위험을 진다. 허가가 안 나면 같은 이미지를 `DRIVESUPRIM_TRT_BACKBONE=off` 로
> 다시 굽는 대신, 아래 "TensorRT 없이 내야 한다면"을 따른다.

---

## 0. 무엇을 제출하는가

| 항목 | 값 |
|---|---|
| 로컬 태그 | `alpasim-e2e-drivesuprim-stage3:g112-kv28-trt` |
| 이미지 ID | `sha256:2b0c5e0067d6a8001ebc7f89136b5563db81dc1f7e1d82d8e523ac79c7f02e40` |
| 크기 | 11.0 GB (한도 40 GiB). 베이스 axe-v9 레이어 27개 + 새 레이어 3개 |
| 베이스 | axe-v9 (`teams/axe@sha256:85134c1e…`), 공식 채점을 이미 통과한 이미지 |
| 모델 | vits512_g112_kv28 (`../models/stage3_vits512_g112_kv28_best_20261010`) |
| 체크포인트 sha256 | `f048fe9c5bc1aaf069d92a023cacb668f2c5e74b2496f107b26e75d347357843` |
| TensorRT | 10.13.0.35 (CUDA 11), ViT 이미지 백본만 FP16 엔진 |
| 게인 파일 | `e2e_challenge/axe_local_eval/controller_gains_axe_v9.json` (lat 1.0 / lon 0.25 / idx 3) |
| 트랙 | `pai` |

로컬 441 점수(PyTorch, 두 번 실행): 0.629 / 0.633. axe-v9 은 같은 조건에서 0.701 이다.
curated_val 441 은 학습에 들어간 클립이라 공식 점수의 예측치가 아니다.

### TensorRT가 하는 일과 안전장치

이미지 백본(ViT-S, 매 Drive 호출마다 3회)만 TensorRT 엔진으로 바꾼다. 나머지는 g112 의
441 평가 때와 바이트 단위로 같다(코드 파일 422개 일치, `policy.py` 에 4줄 hook 추가).

- 드라이버가 뜰 때 공식 GPU 위에서 엔진 출력을 PyTorch 백본과 비교하고(상대 오차 ≤ 5%),
  통과한 경우에만 교체한다. 실패 사유가 무엇이든 PyTorch 로 뜬다. 즉 **TensorRT 문제로
  드라이버가 죽는 경로는 없다.** 로그에 `[DriveSuprim] TRT backbone: ...` 한 줄로 결과가 남는다.
- 엔진 2개: `exact`(H100, cc 9.0 전용)와 `ampere_plus`(cc 8.0 이상 아무 GPU: A100, L40S, A10G
  등). 공식 GPU 종류는 공개되지 않았으므로 둘 다 넣었다.
- 실행 중 TensorRT 호출이 예외를 내면 그 호출부터 PyTorch 로 계속한다.

---

## 1. 로컬 검증 결과 (2026-10-13, `e2e_challenge/g112_trt/validate.sh`)

| # | 항목 | 결과 |
|---|---|---|
| 1 | 이미지·태그·axe-v9 베이스 위에 빌드 | 통과 |
| 2 | entrypoint/cmd/workdir/user, axe-v9 환경변수 67개 유지(경로 2개만 변경) + TRT 스위치 2개 | 통과 |
| 3 | 크기 11 GB ≤ 40 GiB | 통과 |
| 4 | 체크포인트·설정·엔진 해시, TensorRT/torch 버전, 코드가 441 평가 마운트와 동일 | 통과 |
| 5 | 게인 파일을 실제 CLI 검증 함수로 확인 | 통과 |
| 6 | 공식 환경만(변수 4개, read-only, tmpfs, cap-drop, 네트워크 없음): 24초에 로드, `active engine=exact`, gRPC get_version 응답, VRAM 4.1 GB | 통과 |
| 7 | ampere_plus 엔진(H100 이외 GPU 용)이 이 서버에서 역직렬화·검증 통과 후 활성 | 통과 |
| 8 | fallback 4종: 엔진 폴더 없음 / 스위치 off / tensorrt import 불가 / 검증이 모든 엔진 거부 → 전부 PyTorch 로 로드·get_version 응답 | 통과 |
| 9 | 프로세스 내: 엔진 실제 사용, PyTorch 와 같은 후보·궤적 차 0.0 m, TRT 예외 시 PyTorch 전환 후 PyTorch 와 동일 | 통과 |
| 9 | 프로세스 내: 엔진과 다른 shape 입력을 PyTorch 가 처리 | 기능은 동작, 테스트가 PyTorch 두 번 호출의 비트 일치를 요구해 실패 → 상대 오차 비교로 수정, **재실행 전** |

엔진 단독: 백본 1회 7.27 ms → 0.99 ms(exact) / 1.09 ms(ampere_plus), 상대 오차 1.1%.
모델 전체 forward 는 PyTorch 68.2 ms(axe-v9 60.4 ms)에서 약 51 ms 로 줄 것으로 추정한다
(백본 3회분 ≈ 17 ms 절감; 이 서버 H100 단독 측정 기준).

## 2. 검증하지 않은 것

- **폐루프 점수 동등성.** 이 이미지로 441 을 돌려 PyTorch 결과(0.629/0.633)와 비교하는
  실행은 사용자 지시로 보류했다. 스크립트는 `e2e_challenge/g112_trt/run_parity.sh`
  (GPU 4–7, 약 4.5 시간, 결과는 `compare_parity.py` 가 run-to-run 잡음과 비교).
- **공식 ec2 토폴로지(16 레플리카 × 동시 2 롤아웃) 실행.** 렌더러가 GPU 0–3 을 써야 해서
  돌리지 않았다. 드라이버 코드·동시성 처리(`_inference_lock` 으로 추론 직렬화)는 공식 통과한
  axe-v9 과 같다.
- **H100 이외 GPU 에서의 실제 실행.** ampere_plus 엔진은 이 서버 H100 에서만 검증했다.
  다른 GPU 에서 거부되면 PyTorch 로 뜨고, 그 경우 속도는 g112 PyTorch 와 같다.

---

## 3. 등록·약관·잔여 횟수 (quota 소모 없음)

```bash
uv run e2e_challenge/competitor_cli/alpasim_challenge.py me            # "registered": true 확인
uv run e2e_challenge/competitor_cli/alpasim_challenge.py terms status  # 본인·캡틴 모두 최신
uv run e2e_challenge/competitor_cli/alpasim_challenge.py limits        # 남은 횟수
```

토큰 파일 `~/.alpasim/challenge.json` 은 이 계정을 쓰는 연구원 모두가 공유한다. 덮어쓰지
않으려면 `ALPASIM_CONFIG_DIR=~/.alpasim-<이름>` 을 붙인다.

## 4. 태그·push (quota 소모 없음)

```bash
TAG=<새 태그, 예: axe-g112-trt>          # latest 금지, 기존 태그(axe-v9, axe-v10) 재사용 금지
ECR_URI=696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:$TAG

uv run e2e_challenge/competitor_cli/alpasim_challenge.py ecr-login
docker tag sha256:2b0c5e0067d6a8001ebc7f89136b5563db81dc1f7e1d82d8e523ac79c7f02e40 "$ECR_URI"
GPU=4 ECR_URI="$ECR_URI" e2e_challenge/g112_trt/validate.sh   # [10] 이 이미지 ID 동일성을 확인
docker push "$ECR_URI"
docker manifest inspect "$ECR_URI"
```

**태그는 반드시 위 이미지 ID 로 만든다.** 2026-09-12 제출은 다른 원본에서 만든 태그가
올라가 그대로 채점됐다. `validate.sh` 의 [10] 만 이것을 잡는다(`docker manifest inspect`
는 못 잡는다). 베이스 레이어는 ECR `teams/axe` 에 이미 있으므로 실제 업로드는 새
레이어 약 1.1 GB 다.

## 5. 제출 (되돌릴 수 없음)

```bash
uv run e2e_challenge/competitor_cli/alpasim_challenge.py submit "$ECR_URI" --track pai \
  --controller-gains e2e_challenge/axe_local_eval/controller_gains_axe_v9.json
```

**`--controller-gains` 를 빠뜨리지 말 것.** 게인은 이미지에 없고 이 인자로만 전달된다.
빠뜨리면 기본값(lat 1.0 / lon 2.0 / idx 10)으로 채점되는데, g112 를 그 값으로 잰 적이 없다.

## 6. 결과 확인

```bash
uv run e2e_challenge/competitor_cli/alpasim_challenge.py status <submission_id>
```

채점 상태의 throughput wall time 을 한도와 함께 본다(axe-v8: 9,649초 / 한도 14,400초).

---

## TensorRT 없이 내야 한다면

허가가 안 나면 TensorRT 를 이미지에 넣지 않는 편이 안전하다(스위치만 꺼 두는 것은
"TensorRT 를 포함한 이미지"라는 점이 남는다). `e2e_challenge/g112_trt/Dockerfile` 에서
`COPY site/` 와 TRT 환경변수를 빼고, `build.sh` 의 3단계에서 hook 과 엔진 복사를 생략해
다시 빌드한다. 그 이미지는 g112 의 441 평가 구성 그대로다.

## 다시 빌드해야 한다면

```bash
GPU=4 e2e_challenge/g112_trt/build.sh      # 엔진은 runs/g112-trt-build/engines 재사용, 이미지만 재빌드
GPU=4 e2e_challenge/g112_trt/validate.sh
```

재빌드하면 이미지 ID 가 바뀐다. 이 문서의 ID 를 새 값으로 고치고 검증을 다시 통과시킨다.
엔진을 새로 만들려면 `runs/g112-trt-build/engines/` 를 비운다(GPU 1장, 약 5분).
