# HANDOFF — 이 파일 하나로 다음 에이전트가 이어받는다

마지막 갱신: 2026-09-23 21:26 KST (Claude Code)

에이전트(Claude Code, Codex 등)는 세션을 **시작할 때 이 파일과 `git log -10` 을 읽고**,
**끝낼 때 이 파일을 갱신하고 커밋**한다. 대화 원문은 옮기지 않는다. 규칙은 `AGENTS.md`.

## 1. 실행 중인 작업

| 작업 | 시작 | 확인 방법 | 결과 위치 | 종료 예정 |
|---|---|---|---|---|
| ep29 체크포인트 441클립 평가 (GPU 0-3, 드라이버 16 / 렌더러 16) | 19:23 기동, 19:35 주행 시작 | `tail runs/leaderboard-260923-ep29-step30330.progress.log` | `runs/leaderboard-260923-ep29-step30330/aggregate/results-summary.json` | 01:30~02:00 |
| 리랭커 10클립 `before` arm (GPU 0-3, 4워커, 영상 ON) | 19:58 | `tail runs/rerank10-before.progress.log` | `runs/rerank10-before/` (aggregate/, rollouts/<clip>/<session>/*.mp4) | 21:45~22:00 |
| 리랭커 10클립 `rerank` arm | 19:58 | `tail runs/rerank10-rerank.progress.log` | `runs/rerank10-rerank/`, 프레임별 판단은 종료 시 `runs/rerank10-rerank.driver-*.log` | 21:45~22:00 |

- ep29 런처: `e2e_challenge/axe_local_eval/run_260923_ep29_step30330.sh` (nohup, 로그 `260923_ep29_attempt5.nohup.log`). 성공 시 rollout 원본을 스스로 지운다.
- 10클립 런처: `e2e_challenge/route_reranker/run_10clips_reranker.sh` (`ARM=before|rerank`). 종료 후 비교:
  `.venv/bin/python e2e_challenge/axe_local_eval/compare_on_clips.py before=runs/rerank10-before/aggregate/results-summary.json rerank=runs/rerank10-rerank/aggregate/results-summary.json`
- 드라이버 컨테이너: `axe-lb-260923-e29s30330-g*` (16), `axe-rr-rerank10-{before,rerank}-g*` (4+4). 시뮬 스택: compose 프로젝트 `leaderboard-260923-ep29-step30330`, `rerank10-before`, `rerank10-rerank`.
- GPU 2 에는 junhyeok 의 `run_route_pilot.py` (17.7 GB) 가 함께 올라가 있다. 건드리지 않는다.
- 정리 대기: `.trash-260923/` (2.2 GB, 복구 가능). 세 평가가 끝난 뒤 `rm -rf .trash-260923`.

## 2. 최근 결과

- 기준선 axe-v9 (merged-route-ep30, 441클립): `runs/leaderboard-merged-route-ep30/aggregate/results-summary.json`.
- route 캐시 corridor 필터 441 (중심점 판정): 음성 결과. `runs/routecache-val441-center1s/aggregate/`.
- 리랭커 10클립 중간 집계 (첫 4클립 486프레임): 승자가 8 m 게이트 통과 135, 선택 변경 **1** 프레임. 번들의 오프라인 결과(점수 변화 0)와 일치하는 방향.
- 기동 시간: `.pyc` 쓰기 폭주 + 렌더러 2.8 GB 동시 다운로드가 원인. 수정 후 16+16 스택이 12분 만에 주행 시작(이전 4회 실패). `e2e_challenge/axe_local_eval/README.md` "Startup time".

## 3. 마지막 커밋 이후 바뀐 것

- HANDOFF 7절에 GitHub 사본의 LFS 처리 방침(포인터만, clone 시 SKIP_SMUDGE) 추가.
- AGENTS.md 종료 루틴에 `git push mine` 추가 (원격: GitHub JunSeongKW, deploy key `~/.ssh/id_ed25519_junseong*`, ssh 별칭 `github-junseong`, `github-junseong-safedrive`).
- HANDOFF.md 7절(다른 서버에서 재구성) 추가.
- tools/handoff-commit.sh 추가: 세션 종료를 한 명령으로(HANDOFF 시각 갱신 + 스테이징 + 커밋, junseong/* 브랜치만). AGENTS.md 종료 루틴이 이를 가리킨다.

## 4. 다음 단계

1. ep29 441 완료 → 기준선과 나란히 보고 (`compare_on_clips.py ep29=... baseline=runs/leaderboard-merged-route-ep30/aggregate/results-summary.json`).
2. 리랭커 10클립 완료 → before/rerank 클립별 점수·corridor 지표·영상 위치 보고. 변경 프레임이 거의 없으면 "근거리 route 부재로 작동 조건 미충족" 으로 정리하고, 캐시된 route 를 리랭커 입력으로 넣는 변형은 사용자 결정 사항으로 남긴다.
3. 세 평가 종료 후 `.trash-260923` 삭제, `runs/leaderboard-260923-ep29-step30330/rollouts` 가 자동 삭제됐는지 확인.
4. `e2e_challenge/EXPERIMENTS.md` 에 두 실험 행 추가.

## 5. 미결 질문 (사용자 결정 필요)

- 리랭커에 route 캐시(근거리 복원)를 결합한 arm 을 돌릴지.
- corridor 필터 후속: 하드 게이트 대신 감점(demotion) 방식으로 갈지.

## 6. 고정 경로

- 체크포인트: `../models/` (ep29: `../models/260923_epoch=29-step=30330.ckpt`, 이미지 `alpasim-e2e-axe-v9:260923-ep29-step30330`).
- 데이터셋: `data/nre-artifacts -> /home/kaist5/Dataset/alpasim/data/nre-artifacts` (심볼릭 링크, 지우지 말 것).
- 렌더러 공유 캐시: `.cache/renderer-shared/` (비면 `warm_renderer_cache.sh` 가 자동으로 채움).
- 10클립 목록: `e2e_challenge/route_cache_filter/clips10.txt` (선정 근거 `clips10.md`).

## 7. 다른 서버에서 재구성 (git 으로 오지 않는 것)

| 항목 | 이 서버 위치 | 옮기는 방법 |
|---|---|---|
| 데이터셋 (NuRec 아티팩트, 장면) | `data/nre-artifacts -> /home/kaist5/Dataset/alpasim/data/nre-artifacts` | 대상 서버의 데이터 경로로 심볼릭 링크 재생성 |
| 도커 이미지 | `alpasim-base:0.89.0`, `nvcr.io/nvidia/nre/nre-ga:26.04`, 드라이버 이미지 `alpasim-e2e-*` | 베이스는 pull, 드라이버 이미지는 `e2e_challenge/sample_submission_drivesuprim` 의 Dockerfile 로 재빌드(체크포인트는 ../models 에서) |
| 체크포인트 | `../models/*.ckpt` | rsync |
| 렌더러 공유 캐시 2.7 GB | `.cache/renderer-shared/` | rsync 하거나 첫 실행 때 `warm_renderer_cache.sh` 가 자동 생성 |
| host venv | `.venv/` (uv) | `uv sync` (`UV_OFFLINE` 은 캐시가 생긴 뒤부터) |
| 실행 결과 | `runs/<run>/aggregate/results-summary.json` | 요약 JSON 만 rsync 하면 충분 |

git 으로 오는 것: 런처·드라이버 패키지·오버레이 코드·문서·실험 원장.

### GitHub 사본(`mine`)과 Git LFS

NVlabs 원본은 테스트 픽스처(usdz/asl/ply, 약 234 MB)를 LFS 로 관리한다. 개인 사본
`JunSeongKW/alpasim-e2e-eval` 에는 LFS 객체를 올리지 않았다(포인터 파일만 있음). 따라서:

- 다른 서버에서 받을 때: `GIT_LFS_SKIP_SMUDGE=1 git clone -b junseong/e2e-eval git@github-junseong:JunSeongKW/alpasim-e2e-eval.git alpasim`
- 테스트 픽스처가 필요하면 원본을 두 번째 remote 로 두고 가져온다: `git remote add origin https://github.com/NVlabs/alpasim.git && git lfs fetch origin`
- 첫 push 만 `--no-verify` 가 필요했다. 이후 커밋에는 LFS 파일이 없으므로 `git push mine` 그대로 된다.
