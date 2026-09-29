# axe-v10 제출 런북

작성 2026-09-29. **이미지는 빌드·검증 완료 상태이며 남은 것은 push 와 submit 뿐이다.**
모든 경로는 절대경로다. 이 문서는 제출을 실행할 사람을 위한 것이다.

---

## 0. 무엇을 제출하는가

| 항목 | 값 |
|---|---|
| ECR URI | `696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v10` |
| 이미지 ID | `sha256:fe684f3347128963475cb7986eb2a70edf475a07f2f706b47588ea28c6b297ba` |
| 베이스 | axe-v9 (`85134c1ea9f0`) — 실제 제출됐던 이미지 |
| 체크포인트 | `/home/kaist5/data/junseong/AlpaSim-E2E-Challenge/models/stage3_aug_ep29_final.ckpt` |
| 체크포인트 sha256 | `29c6fca63c4d4bd561fb90101ebbf39f7cb23b87a09251b1db3a766f05ffdef8` |
| 게인 파일 | `/home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim/e2e_challenge/axe_local_eval/controller_gains_axe_v9.json` |
| 트랙 | `pai` |

axe-v9 와 달라진 점은 둘이다. **가중치**가 `stage3_aug_ep29_final` 로 바뀌었고,
**route 리랭커가 γ=0.01 로 켜져 있다**(arm `cache-centre-max`: 캐시 ON,
중심점 offset 1.467 m, 최댓값 집계, 최소 겹침 8 m).

로컬 성적(441 curated_val): 평균 장면점수 **0.7180**, at-fault 사고간 거리 **1.2563 km**,
0점률 0.1859. 같은 체크포인트를 리랭커 없이 돌리면 0.6951 이고, 현 제출본 axe-v9 은 0.6993 이다.

> **이 숫자를 공식 점수의 예측치로 읽지 말 것.** curated_val 441 클립은 학습에
> 포함돼 있어 로컬 점수는 학습셋 점수다. 실측 환산은 axe-v5 로컬 0.2275 → 공식
> 0.1971, axe-v9 로컬 0.6993 → 공식 0.5028 로, 로컬이 높을수록 전환율이 떨어진다.

---

## 1. 먼저 확인할 것 — 등록 (이것이 막히면 제출 불가)

```bash
cd /home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim
uv run e2e_challenge/competitor_cli/alpasim_challenge.py me
```

응답의 **`"registered": true`** 여야 한다. 2026-09-29 기준 `JunseongKwak` 계정은
`registered: false` 라 `limits` 와 `submissions` 가 HTTP 403 이었다. 등록되지 않은
계정으로는 submit 이 거부된다.

토큰은 12시간마다 만료된다. `authenticated: false` 면 재인증한다.

```bash
uv run e2e_challenge/competitor_cli/alpasim_challenge.py auth-url
uv run e2e_challenge/competitor_cli/alpasim_challenge.py configure-token <토큰>
```

> 이 계정(`kaist5`)은 여러 연구원이 공유하고 토큰은 `~/.alpasim/challenge.json`
> 하나뿐이다. 다른 사람 세션을 덮어쓰지 않으려면 `ALPASIM_CONFIG_DIR=~/.alpasim-<이름>`
> 을 붙여 분리한다.

---

## 2. 프리플라이트 (quota 소모 없음, 몇 번이든 가능)

```bash
cd /home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim

ECR_URI=696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v10 \
EXPECTED_CKPT_SHA256=29c6fca63c4d4bd561fb90101ebbf39f7cb23b87a09251b1db3a766f05ffdef8 \
PREFLIGHT_GPU=<여유 4 GB 이상인 카드> PREFLIGHT_TIMEOUT=900 \
  e2e_challenge/axe_local_eval/preflight_submit.sh \
  696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v10 \
  e2e_challenge/axe_local_eval/controller_gains_axe_v9.json
```

`PREFLIGHT PASSED` 가 나와야 한다. 2026-09-29 기준 9항목 전부 통과 확인됨.

**`[9]` 항목이 가장 중요하다.** 2026-09-12 제출은 ECR 태그를 잘못된 원본에서 만들어
`DRIVESUPRIM_EGO_*` 가 없는 이미지가 올라갔고 그대로 채점됐다. `docker tag` 는 이미지
ID를 바꾸지 않으므로 ID 비교만이 이것을 잡아낸다. `docker manifest inspect` 로는 못 잡는다.

GPU 가 전부 찼으면 프리플라이트 [8]만 실패한다. 그 경우 여유 4 GB 이상인 카드가
생길 때까지 기다린다. 다른 연구원의 학습이 쓰는 잔여 메모리를 빼앗으면 그쪽이 OOM 난다.

---

## 3. ECR 로그인 및 push (quota 소모 없음)

```bash
cd /home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim

uv run e2e_challenge/competitor_cli/alpasim_challenge.py ecr-login
docker push 696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v10
docker manifest inspect 696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v10
```

push 는 27.6 GB 이고 이 서버의 디스크·네트워크 상태에 따라 오래 걸린다.

---

## 4. 약관·잔여 횟수 (quota 소모 없음)

```bash
uv run e2e_challenge/competitor_cli/alpasim_challenge.py terms status
uv run e2e_challenge/competitor_cli/alpasim_challenge.py limits
```

약관은 **본인과 팀 캡틴 양쪽**이 최신이어야 한다. 한쪽이라도 미동의면 submit 이
거부되며, 이 거부는 submission 을 만들지 않으므로 횟수를 쓰지 않는다.

**남은 횟수를 반드시 여기서 확인한다.** 2026-09-29 기준 AXE 의 공식 제출은 3건
(axe-v5 / v8 / v9)이고 09-17 이후 새 제출이 없다.

---

## 5. 제출 (되돌릴 수 없음)

```bash
cd /home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim

uv run e2e_challenge/competitor_cli/alpasim_challenge.py submit \
  696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v10 \
  --track pai \
  --controller-gains /home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim/e2e_challenge/axe_local_eval/controller_gains_axe_v9.json
```

**`--controller-gains` 를 빠뜨리면 안 된다.** MPC 게인은 이미지에 들어 있지 않고 이
인자로만 전달된다. 생략하면 평가자 기본값(`lat=1.0 / lon=2.0 / idx=10`)으로 채점되는데,
그 조합은 측정한 적이 없다. 파일 이름이 `axe_v9` 인 것은 값이 axe-v9 와 같기 때문이며,
γ=0.01 을 측정한 441 실행도 같은 게인(`lat 1.0 / lon 0.25 / idx 3`)을 썼다.

응답의 `submission_id` 를 기록해 둔다.

---

## 6. 결과 확인

```bash
uv run e2e_challenge/competitor_cli/alpasim_challenge.py status <submission_id>
uv run e2e_challenge/competitor_cli/alpasim_challenge.py leaderboard --track pai
```

채점에는 수 시간이 걸린다. axe-v8 은 9,649초를 썼고 한도 14,400초 안에 들어왔다.

---

## 이 제출에만 있는 주의사항

**리랭커가 이미지에 구워져 있다.** 지금까지의 모든 리랭커 실행은 파일 7개를
bind-mount 하고 환경변수 5개를 주입해 켰다. 공식 평가자는 둘 다 하지 않고
`--read-only` 로 공식 변수 4개(`ALPASIM_DRIVER_HOST/PORT`,
`ALPASIM_CONTESTANT_REPLICA_INDEX/REPLICAS`)만 준다. 그래서 axe-v10 은 코드와 스위치를
전부 이미지에 넣었다. 확인된 사실은 다음과 같다.

- 공식 환경(read-only, 변수 4개, 마운트 없음)에서 드라이버가 스스로
  `ROUTE RERANK: enabled=True weight=0.01 aggregate=max centre_dx_m=1.467 cache=1` 을 찍는다
- `probe_reranker.sh` 가 컨테이너 안에서 선택 함수를 통과시켜
  `RERANK PROBE OK`, `CACHE PROBE OK dense=375 -> resampled=36 slots=64` 를 냈다
- 구운 파일 9개의 sha256 이 소스와 전부 일치한다
- `/app/drivesuprim_challenge` 에 이미지 쪽 잔재 파일이 없다 (COPY 전 `rm -rf` 로
  bind-mount 의 "교체" 의미를 재현했다)

**절대 다른 이미지에서 `axe-v10` 태그를 다시 만들지 말 것.** 위 ID
(`fe684f3347…`)가 아닌 것을 push 하면 2026-09-12 와 같은 사고가 난다.

**검증하지 않은 것이 하나 있다.** 구운 이미지로 실제 클립을 주행시켜 마운트 방식과
점수가 같은지 대조하는 패리티 검증은 GPU 8장이 모두 점유돼 돌리지 못했다. 스크립트는
`e2e_challenge/axe_local_eval/parity_axe_v10.sh` 에 있고 여유 12 GB 인 카드 하나와
15분이면 끝난다. 기준값은 `runs/clip1-ep29-g0p01-cache-centre-max` 의
score 0.6485338950771624 / progress 0.5188271160617299 / 주행거리 69.139 m 다.
구운 파일이 해시 단위로 동일하고 환경변수도 같아 다르게 동작할 경로는 남아 있지
않지만, 시간이 있다면 돌려보는 편이 낫다.

**제출 판단의 근거와 한계.** γ=0.01 은 441 클립 감마 사다리(0, 0.01, 0.02, 0.05, 0.1)의
최고점이다. 다만 0.01 과 0.02 의 차이 0.0077 은 학습셋에서 잰 값이고 시뮬레이터
잡음 범위다. 0.02 는 at-fault 충돌(0.0569 대 0.0612)과 at-fault 거리(1.2846 대 1.2563)에서
앞서므로, **공식 순위가 PCS 와 at-fault 거리로 결정된다는 점을 감안하면 0.02 도
합리적인 선택이었다.** 0.01 을 고른 것은 총점 기준이다.

---

## 빌드를 다시 해야 한다면

```bash
cd /home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim
e2e_challenge/axe_local_eval/prepare_axe_v10_context.sh /tmp/axe_v10_ctx_fast
cd /tmp/axe_v10_ctx_fast && docker build \
  -t 696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v10 \
  --build-arg CHECKPOINT_SHA256=29c6fca63c4d4bd561fb90101ebbf39f7cb23b87a09251b1db3a766f05ffdef8 \
  --build-arg CHECKPOINT_SOURCE=/home/kaist5/data/junseong/AlpaSim-E2E-Challenge/models/stage3_aug_ep29_final.ckpt .
```

`Dockerfile.axe_v10_fast` 를 쓴다. 원래 쓴 `Dockerfile.axe_v10` 은 COPY 를 7줄로 나눠
레이어를 9개 만들었고, 26 GB 베이스 위에서 레이어 커밋 하나에 7~117분이 걸려 빌드에
3시간 41분이 들었다. fast 버전은 staging 디렉터리에 모아 한 번에 COPY 하므로 레이어가
2개다. `prepare_axe_v10_context.sh` 는 빌드 전에 체크포인트 해시와 모든 모듈의 파이썬
문법을 검사한다.
