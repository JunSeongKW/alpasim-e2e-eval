# Retain the exact axe-v9 inference dependencies, controller-facing adapter and
# precision. Overlay only the checkpoint's source, config and five-camera bridge.
FROM 696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe@sha256:85134c1ea9f09d140be610ca063c50ec60e99980c392e5bf81fe6563af503765
COPY app/ /app/
ARG CHECKPOINT_SHA256
ARG SOURCE_MANIFEST_SHA256
LABEL org.alpasim.checkpoint.sha256=$CHECKPOINT_SHA256 \
      org.alpasim.source_manifest.sha256=$SOURCE_MANIFEST_SHA256 \
      org.alpasim.input.cameras="CAM_L1,CAM_L0,CAM_F0,CAM_R0,CAM_R1" \
      org.alpasim.eval.subject="stage3-5cam-ep05"
ENV DRIVESUPRIM_CHECKPOINT_PATH=/app/assets/drivesuprim/stage3_5cam_ep05.ckpt \
    DRIVESUPRIM_CONFIG_PATH=/app/assets/drivesuprim/stage3_5cam_config.json \
    DRIVESUPRIM_VOCAB_PATH=/app/assets/drivesuprim/nurec_train_kmeans_4096x40x3.npy \
    DRIVESUPRIM_BACKBONE_TYPE=bevformer_m \
    DRIVESUPRIM_REQUIRE_EXACT_CHECKPOINT=1 \
    DRIVESUPRIM_PINHOLE_PROFILE=K377_512X256 \
    DRIVESUPRIM_USE_FP16=0 \
    DRIVESUPRIM_EGO_FOOTPRINT_FROM_API=1 \
    DRIVESUPRIM_EGO_CENTER_OFFSET=1 \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
