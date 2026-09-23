# Route 리랭커 10클립 평가 (2026-09-23)

질문: 상황에 따라 planning 에 필요한 route 정보가 달라질 때, 현재 제공되는 far-only route 로 후보 선택이 실제 바뀌는가?

`rerank10-before`와 `rerank10-rerank`는 같은 10클립·이미지·체크포인트·MPC 설정을 사용하고 리랭커 스위치만 다르다. 두 arm 모두 `results-summary.json`에 10개 점수가 있고, 각 arm에 클립 영상 10개와 합본 영상 1개가 있다. 영상은 `runs/rerank10-{before,rerank}/rollouts/<clip>/<session>/*.mp4`, 합본은 각 `aggregate/videos/all/00_all_clips_fast.mp4`.

| clip 접두어 | before 점수 | rerank 점수 | before corridor | rerank corridor |
|---|---:|---:|---:|---:|
| 0feb3afa | 0.000 | 0.000 | 1 | 1 |
| 2466266e | 0.749 | 0.773 | 0 | 0 |
| 37d660b1 | 0.000 | 0.000 | 0 | 0 |
| 61ef954c | 0.000 | 0.000 | 1 | 1 |
| 98694f91 | 0.000 | 0.000 | 1 | 1 |
| 9ea70552 | 0.000 | 0.000 | 1 | 1 |
| dc1966f4 | 0.698 | 0.698 | 0 | 0 |
| ddc3e8df | 0.000 | 0.000 | 1 | 1 |
| e904e9c0 | 0.915 | 0.853 | 0 | 0 |
| eaba3f6a | 0.000 | 0.000 | 1 | 0 |
| **평균 / 비율** | **0.2362** | **0.2324** | **0.60** | **0.50** |

두 arm 모두 0점 7/10. `eaba3f6a`는 rerank에서 corridor 이탈이 사라졌으나 at-fault 전방 충돌이 생겨 여전히 0점이다. 작은 표본의 arm 간 차이를 리랭커의 인과 효과로 해석하지 않는다.

`runs/rerank10-rerank.driver-*.log`의 프레임별 기록에서 선택 변경은 **1/1,896프레임**이다. 비교 가능한 후보가 0개인 프레임이 1,277개이고, 원래 승자가 8 m 게이트를 통과한 프레임은 319개다. 현재 far-only route 는 대부분의 프레임에서 게이트 조건을 충족하지 못한다. 캐시된 근거리 route 를 입력에 결합한 arm 은 별도 설계·결정이 필요하다.

`rerank` wizard 는 exit 0. `before` runtime 은 결과와 영상을 저장한 뒤 정상 종료했지만 Docker Compose 가 renderer 컨테이너를 멈추는 과정에서 daemon 오류를 내어 wizard 는 exit 1. 런처의 로그 수집·컨테이너 정리가 완료됐는지 다음 세션에서 확인한다.
