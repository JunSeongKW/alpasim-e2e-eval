# HANDOFF — 이 파일 하나로 다음 에이전트가 이어받는다

마지막 갱신: 2026-09-23 21:44 KST (Codex)

에이전트(Claude Code, Codex 등)는 세션을 **시작할 때 이 파일과 `git log -10` 을 읽고**,
**끝낼 때 이 파일을 갱신하고 커밋**한다. 대화 원문은 옮기지 않는다. 규칙은 `AGENTS.md`.

## 1. 실행 중인 작업

| 작업 | 시작 | 확인 방법 | 결과 위치 | 종료 예정 |
|---|---|---|---|---|
| ep29 체크포인트 441클립 평가 (GPU 0-3, 드라이버 16 / 렌더러 16) | 19:23 기동, 19:35 주행 시작 | `tail runs/leaderboard-260923-ep29-step30330.progress.log` | `runs/leaderboard-260923-ep29-step30330/aggregate/results-summary.json` | 01:30~02:00 |
| 리랭커 10클립 두 arm 의 로그 수집·컨테이너 정리 | 점수·영상 완료, 21:39 wizard 종료 | `tail e2e_challenge/route_reranker/10clips_rerank10_{before,rerank}.log`; `docker ps --filter name=axe-rr-rerank10` | `runs/rerank10-{before,rerank}/` | 런처 정리 확인 필요 |

- ep29 런처: `e2e_challenge/axe_local_eval/run_260923_ep29_step30330.sh` (nohup, 로그 `260923_ep29_attempt5.nohup.log`). 성공 시 rollout 원본을 스스로 지운다.
- 10클립 런처: `e2e_challenge/route_reranker/run_10clips_reranker.sh` (`ARM=before|rerank`). 종료 후 비교:
  `.venv/bin/python e2e_challenge/axe_local_eval/compare_on_clips.py before=runs/rerank10-before/aggregate/results-summary.json rerank=runs/rerank10-rerank/aggregate/results-summary.json`
- 드라이버 컨테이너: `axe-lb-260923-e29s30330-g*` (16), 정리 대기 중이면 `axe-rr-rerank10-{before,rerank}-g*` (4+4). 시뮬 스택: compose 프로젝트 `leaderboard-260923-ep29-step30330`, `rerank10-before`, `rerank10-rerank`.
- GPU 2 에는 junhyeok 의 `run_route_pilot.py` (17.7 GB) 가 함께 올라가 있다. 건드리지 않는다.
- 21:42 기준 별도 Claude 세션이 `axe-rr-rerank10-*`와 `rerank10-*` 컨테이너를 이름으로 한정해 정리 중이다. 중복 정리하지 말고 해당 작업 종료 후 남은 대상을 확인한다.
- 정리 대기: `.trash-260923/` (2.2 GB, 복구 가능). ep29 평가가 끝난 뒤 삭제.

## 2. 최근 결과

- 기준선 axe-v9 (merged-route-ep30, 441클립): `runs/leaderboard-merged-route-ep30/aggregate/results-summary.json`.
- route 캐시 corridor 필터 441 (중심점 판정): 음성 결과. `runs/routecache-val441-center1s/aggregate/`.
- 리랭커 10클립 점수 완료: before 0.2362 / rerank 0.2324, 둘 다 0점 7/10, corridor 이탈 6/10 → 5/10. 선택 변경 **1/1,896프레임**; 비교 가능한 후보 0개인 프레임 1,277개. 상세 클립별 점수·영상 경로: `e2e_challenge/route_reranker/RESULTS_10CLIPS.md`. `before`는 점수·영상 저장 뒤 Docker 종료 오류로 wizard exit 1.
- ep29는 21:38 기준 148/441 rollout.asl. 완료 전 기준선과 전체 점수를 비교하지 않는다.
- 기동 시간: `.pyc` 쓰기 폭주 + 렌더러 2.8 GB 동시 다운로드가 원인. 수정 후 16+16 스택이 12분 만에 주행 시작(이전 4회 실패). `e2e_challenge/axe_local_eval/README.md` "Startup time".

## 3. 마지막 커밋 이후 바뀐 것

- 10클립 두 arm 결과를 같은 10클립으로 비교하고 `e2e_challenge/route_reranker/RESULTS_10CLIPS.md`에 클립별 점수·corridor·영상·프레임별 선택을 기록했다. far-only route 에서는 게이트 성립이 드물어 연구 질문의 조건이 충분히 형성되지 않았다.
- `e2e_challenge/EXPERIMENTS.md`에 두 arm 행을 추가했다. `before`의 Docker 종료 오류는 결과 생성 뒤에 발생했음을 기록했다.
- ep29 진행률을 확인했다. 평가가 실행 중이므로 점수 결론이나 대량 정리는 보류했다.

## 4. 다음 단계

1. ep29 441 완료 → 기준선과 나란히 보고 (`compare_on_clips.py ep29=... baseline=runs/leaderboard-merged-route-ep30/aggregate/results-summary.json`).
2. 리랭커 두 arm 런처의 로그 수집·컨테이너 정리 완료 확인. `before` wizard 는 결과 저장 뒤 Docker 종료 오류가 났다. 남은 대상이 있으면 이름 필터로 확인 후 정리.
3. ep29 종료 후 `.trash-260923` 삭제, `runs/leaderboard-260923-ep29-step30330/rollouts` 가 자동 삭제됐는지 확인.
4. 사용자가 원하면 근거리 route 캐시를 리랭커 입력에 결합한 arm 을 설계. 현재 10클립 결과는 게이트 조건이 거의 성립하지 않아 리랭커 자체의 유효성을 판별하지 못한다.

## 5. 미결 질문 (사용자 결정 필요)

- 10클립에서 선택 변경이 1/1,896프레임에 그친 만큼 리랭커에 route 캐시(근거리 복원)를 결합한 arm 을 돌릴지.
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

NVlabs 원본은 테스트 픽스처(usdz/asl/ply 등 20개, 약 225 MB)를 LFS 로 관리한다. GitHub 는
LFS 포인터가 가리키는 객체가 없는 push 를 거부하므로(GH008), 개인 사본에는 LFS 객체까지
올려 두었다(`git lfs fetch --all origin` 으로 NVlabs 에서 받은 뒤 `git lfs push --all mine`).
무료 LFS 용량 1 GB 중 약 225 MB 를 쓴다.

- 다른 서버에서 받을 때 대역폭을 아끼려면: `GIT_LFS_SKIP_SMUDGE=1 git clone -b junseong/e2e-eval git@github-junseong:JunSeongKW/alpasim-e2e-eval.git alpasim`
  (평가에는 픽스처가 필요 없다. pytest 를 돌릴 때만 `git lfs pull`.)
- 이후 커밋에는 LFS 파일이 없으므로 종료 루틴의 `git push mine` 그대로 된다.
