# HANDOFF — 이 파일 하나로 다음 에이전트가 이어받는다

마지막 갱신: 2026-09-24 23:00 KST (Claude Code)

에이전트(Claude Code, Codex 등)는 세션을 **시작할 때 이 파일과 `git log -10` 을 읽고**,
**끝낼 때 이 파일을 갱신하고 커밋**한다. 대화 원문은 옮기지 않는다. 규칙은 `AGENTS.md`.

## 1. 실행 중인 작업

없음. γ 튜닝 파이프라인 5단계 전부 완료(23:00). GPU 0-3 반납됨.

- 정리 대기: `.trash-260923/` 2.2 GB — 이제 실행 중인 평가가 없으므로 `rm -rf` 해도 안전.
- 미완: 원격 push (`git push mine`) 는 에이전트 권한으로 막혀 있어 사용자가 직접 해야 함.

## 2. 최근 결과

### ★ 리랭커 441 최종 (2026-09-24)

| 지표 | γ=0.02 | γ=0.1 | axe-v9 |
|---|---:|---:|---:|
| 평균 점수 | **0.7299** | 0.7242 | 0.6993 |
| 0점률 | 0.1905 | **0.1701** | 0.2200 |
| corridor 이탈 | 0.0726 | **0.0635** | 0.0909 |
| 도로 이탈 | 0.0884 | **0.0658** | 0.0886 |
| 전체 충돌 | **0.2245** | 0.2766 | 0.2364 |
| at-fault 충돌 | **0.0363** | 0.0454 | 0.0386 |
| 주행거리 m | **165.4** | 153.6 | 165.2 |
| GT 오차 m | 1.856 | **1.717** | 1.959 |

- **γ=0.02 가 제출 후보**: axe-v9 대비 +0.0306, 모든 지표 개선(차선위반 +0.002 만 미세 악화), trade-off 없음.
- γ=0.1 은 route 준수 전문: 0점률·이탈·GT오차 최저지만 충돌 +0.040, 주행거리 −12 m. 점수는 0.02 와 사실상 동급이나 at-fault 거리 민감한 리더보드에서는 불리.
- 설정: cache=1, centre_dx=1.467, aggregate=max, min_overlap=8. 실행 `runs/rr441-g0p02-cache-centre-max/`, `runs/rr441-g0p1-cache-centre-max/`.
- 방법론: 무작위 층화 40클립 검증이 정확(+0.0398 예측 vs +0.0306 실측). 실패군 편향 38클립의 가중 추정은 방향만 맞고 크기 1/3 과소.

### γ 스윕 핵심 (2026-09-24)

- 번들 원본(캐시 없음): 1,892프레임 중 교체 1건. 승자 적격 프레임 16.9%. 작동 조건 미충족.
- 캐시+중심점+최댓값(cache-centre-max), 1클립(eaba3f6a): γ 0.0005→교체 0, 0.05→58건·점수 0→1.0.
- 38클립(γ=0 짝지은 기준): 해결 0.02→5/26, 0.05→9/26, 0.1→11/26; 대조군 파손 0/1/2; 충돌 0.079→0.053/0.105/0.158.
- **441 추정(실패 9.1% 가중)**: 0.02 +0.0097 > 0.05 +0.0047 > 0.1 +0.0028 — 38클립 순위와 반대. 대조군이 0.95+ 클립뿐이라 0점 클립 97개의 반응은 미측정 → ②검증셋의 존재 이유.
- 시뮬레이터 비결정성: 같은 코드·체크포인트에서 워커 16→4 만으로 38클립 중 8개 판정이 뒤집힘. 비교는 반드시 같은 워커 수의 짝지은 기준선과.
- ep29 441: 0.6886 vs axe-v9 0.6993 (at-fault 충돌 2배, 도로이탈 −1.6%p). 제출 후보 아님.

- 기준선 axe-v9 (merged-route-ep30, 441클립): `runs/leaderboard-merged-route-ep30/aggregate/results-summary.json`.
- route 캐시 corridor 필터 441 (중심점 판정): 음성 결과. `runs/routecache-val441-center1s/aggregate/`.
- 리랭커 10클립 점수 완료: before 0.2362 / rerank 0.2324, 둘 다 0점 7/10, corridor 이탈 6/10 → 5/10. 선택 변경 **1/1,896프레임**; 비교 가능한 후보 0개인 프레임 1,277개. 상세 클립별 점수·영상 경로: `e2e_challenge/route_reranker/RESULTS_10CLIPS.md`. `before`는 점수·영상 저장 뒤 Docker 종료 오류로 wizard exit 1.
- 리랭커 후처리 완료: 두 arm 모두 영상 11개와 요약 JSON 보존. `before`의 wizard exit 1은 Docker 종료 단계에서 발생했고 이름 한정 컨테이너 정리는 끝났다.
- ep29는 21:48 기준 161/441 rollout.asl. 완료 전 기준선과 전체 점수를 비교하지 않는다.
- 기동 시간: `.pyc` 쓰기 폭주 + 렌더러 2.8 GB 동시 다운로드가 원인. 수정 후 16+16 스택이 12분 만에 주행 시작(이전 4회 실패). `e2e_challenge/axe_local_eval/README.md` "Startup time".

## 3. 마지막 커밋 이후 바뀐 것

- γ=0.02 / γ=0.1 441 실행 완료, EXPERIMENTS.md 3행 추가, HANDOFF 1·2절 갱신.
- run_10clips_reranker.sh: RENDER_VIDEO 가 하드코딩 true 였음 → 환경변수 존중으로 수정(38클립 스윕 5 arm 이 영상 39개씩 렌더링했음; 441 전에 반드시 필요). start_drivers_reranker.sh: 재사용 조회 3회 재시도(도커 지연 시 빈 결과로 드라이버를 지우던 문제). 둘 다 임시파일+mv 로 교체해 실행 중 인스턴스에 영향 없음.
- ② 재개 검증: 런타임 autoresume 은 장면별 `_complete` 개수만큼 빼고 배분(`simulate/__main__.py`). g0 arm: 16 건너뜀, 24 jobs. '완료분 재실행' 은 append 된 옛 로그를 잘못 읽은 것.
- run_10clips_reranker.sh 에 RESUME/FULL_SET 스위치, decide_gamma.py, chain_validate_then_441.sh(nohup 체인). ① 클립별 표: 80도 이상 회전 실패는 γ≥0.02 에서 안정적으로 해결(2c263e19·98694f91·ddc3e8df), γ=0.1 은 멀쩡한 회전을 깸(e904e9c0 0.99→0), γ=0.01 은 중간 함정(dc1966f4 0.70→0), 직진 실패는 어느 γ 로도 안 풀림.
- 리랭커: route 캐시 결합, 변형(centre/max), 시작 게이트·감시자, γ 스윕·검증·441 런처, 리포트 스크립트 일체. 세 차례 조용한 실패(64칸 초과, import 경로, 변수 범위) 수정.
- 두 10클립 arm 의 런처 종료와 영상·요약 JSON을 확인하고, 남아 있던 Docker 컨테이너를 해당 실험 이름으로 한정해 정리했다. 21:48 기준 대상 컨테이너가 없다.
- `e2e_challenge/EXPERIMENTS.md`의 실행 상태 문구를 완료 상태로 고쳤다. 점수는 이전 커밋의 `route_reranker/RESULTS_10CLIPS.md`에 있다.
- ep29는 계속 평가 중이므로 `.trash-260923`와 rollout 원본을 그대로 둔다. 완료 후 441클립 기준선 비교가 다음 작업이다.

## 4. 다음 단계

1. ep29 441 완료 → 기준선과 나란히 보고 (`compare_on_clips.py ep29=... baseline=runs/leaderboard-merged-route-ep30/aggregate/results-summary.json`).
2. ep29 종료 후 `.trash-260923` 삭제, `runs/leaderboard-260923-ep29-step30330/rollouts` 가 자동 삭제됐는지 확인.
3. 근거리 route 캐시를 리랭커 입력에 결합하는 arm 의 필요성을 결정. 현재 10클립 결과는 게이트 조건이 거의 성립하지 않아 리랭커 자체의 유효성을 판별하지 못한다.

## 5. 미결 질문 (사용자 결정 필요)

- 10클립에서 선택 변경이 1/1,896프레임에 그쳤다. 근거리 route 캐시를 결합해 리랭커의 작동 조건을 형성한 arm 을 돌릴지.
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
