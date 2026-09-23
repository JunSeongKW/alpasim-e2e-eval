# 텐서 크기 레퍼런스

`drivesuprim_agent_bevformer_vov_v2_vits_stage3` + `use_route=True` 기준.
**토큰 수**와 **채널**을 헷갈리지 않게 분리해서 적는다. 채널은 거의 전부 256이다
(`tf_d_model = bev_embed_dims = 256`, 그래야 `cat` 이 된다).

## 0. 스칼라 설정값

| 이름 | 값 | 출처 |
|---|---|---|
| `tf_d_model` | 256 | config:587 |
| `tf_d_ffn` | 1024 | config:588 |
| `tf_num_head` | 8 | config:590 |
| `bev_embed_dims` | 256 | yaml (= tf_d_model 이어야 함) |
| `bev_h` × `bev_w` | 56 × 56 | vits yaml |
| `point_cloud_range` | x[0, 56] y[-28, 28] z[-3, 5] m | vits yaml (기본값 ±32 를 덮음) |
| → BEV 해상도 | **1.0 m / cell** | 56 m / 56 cell |
| `bev_keyval_grid` | 28 | vits yaml |
| `bev_num_cameras` | 3 | CAM_L0 / F0 / R0 |
| `bev_img_height` × `width` | 256 × 512 | config:257-258 |
| `bev_seq_len` | 3 | 현재 + 과거 2 |
| `bev_num_encoder_layers` | 3 | vits yaml |
| `vocab_size` | 4096 | vits yaml |
| vocab 파일 shape | **(4096, 40, 3)** | 4 s / 0.1 s = 40 pose, (x,y,yaw) |
| `vadv2_head_nlayers` | 3 | stage 1 디코더 층 수 |
| `refinement.stage_layers` | "3" | refinement 층 수 |
| `refinement.topks` | "256" | 후보 256 개로 압축 |
| `num_ego_status` | 1 | 현재 프레임만 |
| route 슬롯 | **20** | 데이터 실측, (20, 3) 중 앞 2 만 사용 |
| `route_norm_m` | 80.0 | x 최대 81.2 m |
| `route_hidden_dim` | 64 | RouteEncoder / RouteTokens 내부 |

## 1. 원시 입력

| 텐서 | shape | 비고 |
|---|---|---|
| `bev_imgs` | `[bs, 3, 3, 3, 256, 512]` | (seq 3, cam 3, RGB 3, H, W) |
| `lidar2img` | `[bs, 3, 3, 4, 4]` | seq × cam × 4×4 |
| `status_feature` | `[bs, 8]` | cmd(4, **use_route 면 전부 0**) + vel(2) + acc(2) |
| `route_feature` | `[bs, 20, 2]` | /80 정규화, 패딩 0 |
| `route_mask` | `[bs, 20]` | 1=유효. 프레임당 유효 median 10 |
| `vocab` | `[4096, 40, 3]` | 학습 안 됨 (`requires_grad=False`) |

## 2. BEV 인코더 출력

| 텐서 | 토큰 수 | 채널 | 비고 |
|---|---:|---:|---|
| `bev_feature` | 56 × 56 = **3,136** | 256 | 1 m/cell, x 0~56 m |

## 3. Stage 1 — 4096 후보 채점

| 텐서 | 토큰 수 | 채널 | 비고 |
|---|---:|---:|---|
| `embedded_vocab` (query) | **4,096** | 256 | vocab 40×3=120 → pos_embed |
| `keyval` (image memory) | 28 × 28 = **784** | 256 | `bev_keyval_pool` 로 3136 → 784 |
| `route_tokens` | **20** | 256 | `RouteTokens` |
| `memory` = cat | **804** | 256 | route 비중 **2.6 %** |
| attention score | `[bs, 4096, 804]` | — | softmax 는 804 축 |
| `status_encoding` | 1 (broadcast) | 256 | 트랜스포머 **뒤**에 더해짐 |
| `dist_status` | 4,096 | 256 | `tr_out + status_encoding` |
| `scores` | 4,096 | 1 | 9 개 head 가중합 |

## 4. Refinement — 256 후보 재순위

**주의: 후보 개수 256 과 채널 256 이 우연히 같다. 헷갈리기 쉬운 지점.**

| 텐서 | 토큰 수 | 채널 | 비고 |
|---|---:|---:|---|
| `trajs_status` (query) | **256** ← 후보 수 | 256 ← 채널 | `dist_status` 에서 top-k gather |
| `_img_feat_fg` (image memory) | 56 × 56 = **3,136** | 256 | **풀링 없음** |
| `route_tokens` | **20** | 256 | 별개 인스턴스 |
| `memory` = cat | **3,156** | 256 | route 비중 **0.63 %** |
| attention score | `[bs, 256, 3156]` | — | softmax 는 3156 축 |

## 5. 후보 깔때기

```
4096 (vocab)  --top-k-->  256  --argmax-->  1
```

## route 비중 요약

| | 이미지 토큰 | route 토큰 | route 비중 |
|---|---:|---:|---:|
| stage 1 | 784 | 20 | **2.6 %** |
| refinement | 3,136 | 20 | **0.63 %** |

`bev_keyval_grid: 28` 은 메모리 절약이 목적이지만, **부수효과로 stage 1 의
route 비중을 4 배 높인다.** 28 → 56 으로 올리면 2.6 % → 0.63 % 로 떨어진다.
두 사안은 반대 방향으로 당기므로 같이 결정해야 한다.

## route 와 BEV 박스의 겹침 (실측, 12,287 프레임)

`point_cloud_range` 가 x[0,56] y[-28,28] 이므로 route 가 BEV 밖이라는 것은 사실이 아니다.

```
전체 유효 waypoint 중 박스 안  : 38.5 %
프레임당 안에 든 개수          : median 4  (p10 3, p90 4)
1 개 이상 든 프레임            : 95.3 %
하나도 없는 프레임             :  4.7 %
route x 범위                   : min -43.8,  p50 58.0,  max 81.2 m
```

SafeDrive 식 trajectory-as-reference-point 를 route 에 적용하는 것이 원리적으로
막혀 있지 않다. 다만 60 % 는 박스 밖이라 `padding_mode='border'` 에 걸리므로,
안에 든 것만 reference point 로 쓰고 나머지는 토큰으로 두는 혼합이 현실적이다.
