# 이 저장소에서 에이전트가 지킬 것 (junseong 작업 규칙)

Claude Code 는 `CLAUDE.md`(= `@AGENTS.md`)로, Codex 는 이 파일로 같은 내용을 읽는다.
프로젝트의 **현재 상태**는 `HANDOFF.md` 에만 있다. 이 파일에는 바뀌지 않는 규칙만 둔다.

## 세션 시작 루틴

```bash
git status --short --branch && git log --oneline -10
cat HANDOFF.md
docker ps --filter name=axe- --format '{{.Names}}\t{{.Status}}' ; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
tail -n 3 runs/*.progress.log 2>/dev/null
```

## 세션 종료 루틴

1. `HANDOFF.md` 의 1(실행 중)·2(결과)·3(변경)·4(다음)·5(미결) 절을 갱신한다.
2. 커밋한다: `tools/handoff-commit.sh "<에이전트 이름>" "[eval] 제목 한 줄"`. 그리고 `git push mine` (원격 `mine` = GitHub JunSeongKW, 다른 서버가 같은 상태를 보도록). HANDOFF.md 3절(마지막 커밋 이후 바뀐 것)이
   커밋 본문이 되므로 원인·조치·결과 경로·다음 할 일을 그 절에 먼저 적는다. 제목 접두어는 `[eval]` `[feature]` `[infra]`.
   실험을 띄운 커밋에는 `git tag run/<run-name>` 를 단다. 커밋은 `junseong/*` 브랜치에만 한다(스크립트가 거부함).
3. 장시간 작업은 nohup 으로 띄우고 진행 로그를 `runs/<run>.progress.log` 에 남긴다. 세션 임시 폴더에 두지 않는다.

## 바뀌지 않는 규칙

- **GPU**: 이 계정의 카드는 4~7 이다. 0~3 은 사용자가 명시적으로 허락한 작업에만 쓴다. util 0% 는 비어 있다는 뜻이 아니다. 쓰기 전에 `nvidia-smi` 로 여유 메모리를 본다.
- **공유 머신**: 컨테이너·프로세스 정리는 반드시 이름 필터로 범위를 좁히고 대상을 먼저 출력한다. `docker rm -f $(docker ps -aq)`, `docker container prune`, `--filter ancestor=`, `pkill -f` 금지. 다른 연구원(junhyeok, hanbin, dogun, uisung)의 컨테이너·프로세스·폴더는 건드리지 않는다.
- **평가 중 대량 삭제 금지**: 수십 GB 삭제는 ext4 저널을 막아 실행 중인 평가를 죽인다. 정리는 `.trash-*/` 로 옮겨 두고 평가가 끝난 뒤 지운다.
- **평가 계약 불변**: 16 드라이버 / 16 워커 / 16 렌더러 / 441 curated_val / dev preset / gains lat 1.0 · lon 0.25 · idx 3. 기동 속도 관련 변경(`FAST_STARTUP`, 캐시, `PYTHONDONTWRITEBYTECODE`)은 점수에 영향이 없는 것만 허용한다.
- **실행 방식**: 확인 질문으로 멈추지 말고 합리적 기본값으로 진행한 뒤 가정을 결과와 함께 보고한다. 되돌릴 수 없는 삭제만 예외.
- **연구 명제에 고정**: 모든 실험 제안은 "planning 에 필요한 정보는 상황마다 다르고 사람이 미리 정하면 안 된다" 는 명제의 어느 부분에 답하는지 한 줄로 밝힌다. 효과 크기가 큰 곁가지보다 원래 질문을 겨냥한 실험이 우선이다.
- **결과 보고**: 성능 향상 크기가 아니라 논문 기여가 되는지를 기준으로 판단해 말한다. 실험 결과는 차수별로 쪼개지 말고 한 표로 모은다.
- **환경**: 새 의존성은 새 venv 에. `uv run` 은 `UV_OFFLINE=1`. 새 데이터셋은 `/home/kaist5/Dataset/` 관례를 따른다. 토큰·키는 화면·로그·채팅에 절대 내지 않는다.
- **경로**: `data/nre-artifacts` 는 공용 데이터셋으로 가는 심볼릭 링크다(지우지 말 것). 체크포인트는 `../models/`. 결과는 `runs/<run>/aggregate/results-summary.json`. 실험 원장은 `e2e_challenge/EXPERIMENTS.md`.

---

# Repository Guidelines

## Overview

Alpasim is a lightweight, data-driven research simulator for autonomous vehicle testing using a microservice architecture with gRPC communication. The runtime orchestrates physics simulation, traffic, neural rendering, and ego vehicle policy evaluation.

## Documentation

- **User docs** → [docs/ONBOARDING.md](docs/ONBOARDING.md), [docs/TUTORIAL.md](docs/TUTORIAL.md), [docs/OPERATIONS.md](docs/OPERATIONS.md)
- **Design and data** → [docs/DESIGN.md](docs/DESIGN.md), [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md)
- **Coordinate frames, coding style, contributing** → [CONTRIBUTING.md](CONTRIBUTING.md)

## Build and run (quick reference)

- **Environment**: `source setup_local_env.sh` (or `./setup_local_env.sh`). The project uses **uv** for dependencies and scripts; use `uv run` for commands (e.g. `uv run pytest`, `uv run alpasim_wizard ...`).
- **After changing `.proto` files**: `cd src/grpc && uv run compile-protos`
- **Run simulation locally** (from repo root or `src/wizard`): `uv run alpasim_wizard deploy=local topology=1gpu driver=vavam wizard.log_dir=./my_run` (deploy configs live in `src/wizard/configs/deploy/`.)
- **Tests**: `uv run pytest` (e.g. `uv run pytest src/runtime/tests`)
- **Static checks**: `pre-commit run --all-files`

## Environment and external repos

Environment variables and external repository URLs are project- or environment-specific; see CONTRIBUTING and your setup for local development.

## Commit and pull request guidelines

- Keep commits focused and imperative. Rebase onto `main` before submitting; force-pushes are expected after rebases.
- Pipelines auto-bump versions for touched packages; allow the bot-generated commit to land and re-trigger CI if needed.
- PRs should explain scenario impact, reference issue IDs, and attach logs/screens for wizard/runtime regressions. Confirm tests and `pre-commit` pass before requesting review.
- When pushing to a branch that has the auto-bump commit "Alpasim automatic version bump", force push over it if that's the only commit you'd overwrite. Do not manually update docker container versions; that is done by the CI pipeline.

## Other conventions

- Coordinate frame conventions: [CONTRIBUTING.md](CONTRIBUTING.md)

## MCP Servers

When asked to access any of the following services, check if you have access to the corresponding MCP server:

- Linar
- Gitlab (especially relevant for MRs)
