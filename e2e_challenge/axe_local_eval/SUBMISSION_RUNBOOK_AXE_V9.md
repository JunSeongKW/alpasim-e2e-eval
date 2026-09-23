# axe-v9 제출 런북

작성 2026-09-17. 이미지는 준비·검증 완료 상태이며, 남은 것은 **push와 submit** 뿐이다.
모든 경로는 절대경로다.

---

## 0. 무엇을 제출하는가

| 항목 | 값 |
|---|---|
| ECR URI | `696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v9` |
| 이미지 ID | `sha256:85134c1ea9f09d140be610ca063c50ec60e99980c392e5bf81fe6563af503765` |
| 체크포인트 | `/home/kaist5/data/junseong/models/stage3_merged_route_ep30.ckpt` |
| 체크포인트 sha256 | `364c801e397e9d6e36ea1f9027dc4ca804ad2e429bb588cf4e84185ca851de3d` |
| 게인 파일 | `/home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim/e2e_challenge/axe_local_eval/controller_gains_axe_v9.json` |
| 트랙 | `pai` |

제출 이미지는 441클립 로컬 평가(`/home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim/runs/leaderboard-merged-route-ep30`)에
쓴 이미지와 **레이어 단위로 동일하다**. 제출용으로 레이어를 덧붙이지 않았다 — 필요한
`DRIVESUPRIM_*` 환경변수가 이미 전부 구워져 있기 때문이다. 평가한 것과 제출하는 것이
같은 이미지여야 로컬 점수가 의미를 가진다.

로컬 성적: avg scene score **0.6993**, at-fault 거리 **1.2982 km**, PCS 2649
(자체 모델 9개 + 레퍼런스 8개 중 rank 구간 1–4).

---

## 1. 제출 전 최종 확인 (quota 소모 없음)

```bash
cd /home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim

ECR_URI=696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v9 \
PREFLIGHT_GPU=4 PREFLIGHT_TIMEOUT=900 \
  /home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim/e2e_challenge/axe_local_eval/preflight_submit.sh \
  alpasim-e2e-drivesuprim-stage3:merged-route-ep30 \
  /home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim/e2e_challenge/axe_local_eval/controller_gains_axe_v9.json
```

`PREFLIGHT PASSED` 가 나와야 한다. 2026-09-17 기준 9개 항목 전부 통과 확인됨.
GPU 4가 사용 중이면 `PREFLIGHT_GPU` 를 비어 있는 번호로 바꾼다.

**`[9]` 항목이 가장 중요하다.** 2026-09-12 제출은 ECR 태그를 잘못된 원본에서 만들어
`DRIVESUPRIM_EGO_*` 가 없는 이미지가 올라갔고, 그대로 채점됐다. `docker tag` 는 이미지
ID를 바꾸지 않으므로 ID 비교만이 이것을 잡아낸다. `docker manifest inspect` 로는 못 잡는다.

---

## 2. ECR 로그인 및 push

```bash
cd /home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim

uv run python e2e_challenge/competitor_cli/alpasim_challenge.py ecr-login

docker push 696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v9
```

push 후 원격에 올라간 것이 로컬과 같은지 확인한다.

```bash
docker manifest inspect 696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v9
```

---

## 3. 약관·잔여 횟수 확인 (quota 소모 없음)

```bash
cd /home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim

uv run python e2e_challenge/competitor_cli/alpasim_challenge.py terms status
uv run python e2e_challenge/competitor_cli/alpasim_challenge.py limits
```

약관은 **본인과 팀 캡틴 양쪽**이 최신이어야 한다. 한쪽이라도 미동의면 submit이 거부되며,
이 거부는 submission을 만들지 않으므로 횟수를 쓰지 않는다.

---

## 4. 제출

```bash
cd /home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim

uv run python e2e_challenge/competitor_cli/alpasim_challenge.py submit \
  696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe:axe-v9 \
  --track pai \
  --controller-gains /home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim/e2e_challenge/axe_local_eval/controller_gains_axe_v9.json
```

**`--controller-gains` 를 빠뜨리면 안 된다.** MPC 게인은 이미지에 들어 있지 않고 이
인자로만 전달된다. 생략하면 평가자 기본값(`lat=1.0 / lon=2.0 / idx=10`)으로 채점되는데,
그 조합은 측정한 적이 없다.

응답의 `submission_id` 를 기록해 둔다.

---

## 5. 결과 확인

```bash
cd /home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim

uv run python e2e_challenge/competitor_cli/alpasim_challenge.py status <submission_id>
uv run python e2e_challenge/competitor_cli/alpasim_challenge.py leaderboard --track pai
```

채점에는 수 시간이 걸린다. 이전 제출(axe-v8)은 9,649초를 썼고 한도 14,400초 안에
들어왔다. 리더보드 반영까지 시간이 걸리는 것은 정상이다.

---

## 주의사항

**게인은 이미지가 아니라 제출 인자다.** 4절의 `--controller-gains` 가 유일한 전달
경로다. 값은 441클립 평가에 실제 적용된 것과 동일하며, 그 사실은
`/home/kaist5/data/junseong/AlpaSim-E2E-Challenge/alpasim/runs/leaderboard-merged-route-ep30/controller-config.yaml`
와 대조해 확인했다.

**`latest` 태그는 거부된다.** `axe-v9` 는 문제없다.

**제출은 되돌릴 수 없다.** 이번 달 남은 횟수를 3절에서 반드시 먼저 확인한다.

**로컬 상위권은 통계적으로 구분되지 않는다.** 자체 모델 상위 4개가 rank 구간 1–4로
전부 겹친다. axe-v9 를 고른 근거는 확정된 우위가 아니라, 최신 체크포인트이면서
`dist_to_gt_trajectory` 1.9594로 GT 추종이 가장 정확하다는 점이다.

**공식 순위 지표는 로컬 PCS와 다르다.** 공개 리더보드의 `legacy_score` 는 at-fault
사고 간 거리이고, 그 지표만 보면 로컬에서는 `stage3_new_ep30`(1.6000)이 axe-v9(1.2982)보다
높다. PCS/`rank_hi` 를 노리는 선택임을 인지하고 제출한다.
