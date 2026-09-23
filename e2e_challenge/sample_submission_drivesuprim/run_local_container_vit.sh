  #!/usr/bin/env bash
  set -euo pipefail

  IMAGE="${IMAGE:-alpasim-e2e-drivesuprim-vit-driver:latest}"

  HOST_PORT="${ALPASIM_DRIVER_PORT:-6789}"
  CONTAINER_PORT="${ALPASIM_DRIVER_CONTAINER_PORT:-6789}"

  # AlpaSim은 GPU 0, DriveSuprim은 GPU 1
  GPU_INDEX="${ALPASIM_GPU_INDEX:-1}"

  # EC2에서 전달되는 submission 환경변수
  REPLICA_INDEX="${ALPASIM_CONTESTANT_REPLICA_INDEX:-0}"
  REPLICAS="${ALPASIM_CONTESTANT_REPLICAS:-1}"

  MAX_BATCH_SIZE="${DRIVESUPRIM_MAX_BATCH_SIZE:-2}"
  BATCH_WAIT_MS="${DRIVESUPRIM_BATCH_WAIT_MS:-5}"
  INFERENCE_INTERVAL_US="${DRIVESUPRIM_INFERENCE_INTERVAL_US:-0}"

  args=(
    docker run
    --rm
    --init
    --name drivesuprim-vit-jh

    --gpus "device=${GPU_INDEX}"

    --cap-drop ALL
    --security-opt no-new-privileges:true
    --read-only
    --pids-limit 1024
    --memory 32g
    --cpus 8

    # 공식 제출 환경과 동일한 writable scratch 크기
    --tmpfs /tmp:rw,nosuid,nodev,size=2g
    --tmpfs /run:rw,nosuid,nodev,size=64m

    -p "127.0.0.1:${HOST_PORT}:${CONTAINER_PORT}"

    -e "ALPASIM_DRIVER_HOST=0.0.0.0"
    -e "ALPASIM_DRIVER_PORT=${CONTAINER_PORT}"

    -e "ALPASIM_CONTESTANT_REPLICA_INDEX=${REPLICA_INDEX}"
    -e "ALPASIM_CONTESTANT_REPLICAS=${REPLICAS}"

    ## DriveSuprim VoV
    -e "DRIVESUPRIM_BACKBONE_TYPE=bevformer_m"
    -e "DRIVESUPRIM_CHECKPOINT_PATH=/app/assets/drivesuprim/drivesuprim_vit.ckpt"
    -e "DRIVESUPRIM_BACKBONE_PATH=/app/assets/drivesuprim/da_vitl16.pth"
    -e "DRIVESUPRIM_VOCAB_PATH=/app/assets/drivesuprim/test_8192_kmeans.npy"

    # GetVersion은 즉시 응답하고 StartSession에서 모델 로딩을 기다림
    -e "DRIVESUPRIM_REQUIRE_POLICY_FOR_VERSION=0"
    -e "DRIVESUPRIM_SESSION_STARTUP_TIMEOUT_S=600"

    -e "DRIVESUPRIM_MAX_BATCH_SIZE=${MAX_BATCH_SIZE}"
    -e "DRIVESUPRIM_BATCH_WAIT_MS=${BATCH_WAIT_MS}"
    -e "DRIVESUPRIM_INFERENCE_INTERVAL_US=${INFERENCE_INTERVAL_US}"

    # -e "DRIVESUPRIM_USE_FP16=1"
    # -e "DRIVESUPRIM_NATIVE_FP16=0"

    -e "ALPASIM_DRIVER_GRPC_WORKERS=8"
  )

  args+=("${IMAGE}")

  exec "${args[@]}"