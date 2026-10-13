# vits512_g112_kv28 submission image, image backbone on a TensorRT FP16 engine.
#
# Base: the exact axe-v9 image the official evaluator has already run (runtime,
# CUDA 11.8 / torch 2.0.1, entrypoint, workdir, user and every baked DRIVESUPRIM_*
# flag unchanged). On top, exactly what the 441 evaluation of this model mounted:
# the bundle's navsim + nurec_pf, the pkg3cam adapter, checkpoint and composed
# config -- the replaced directories are removed first so nothing of axe-v9's
# copies is left behind, as with the read-only mounts. Then the TensorRT 10.13
# (CUDA 11) runtime libraries, the engines and the small module that installs them.
# Building TensorRT engines needs none of this; only the runtime is shipped.
FROM 696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe@sha256:85134c1ea9f09d140be610ca063c50ec60e99980c392e5bf81fe6563af503765
RUN rm -rf /app/navsim /app/nurec_pf /app/drivesuprim_challenge
COPY app/ /app/
COPY site/ /usr/local/lib/python3.10/dist-packages/
ARG CHECKPOINT_SHA256
ARG SOURCE_MANIFEST_SHA256
ARG CONFIG_SHA256
ARG ENGINES_MANIFEST_SHA256
LABEL org.alpasim.checkpoint.sha256=$CHECKPOINT_SHA256 \
      org.alpasim.source_manifest.sha256=$SOURCE_MANIFEST_SHA256 \
      org.alpasim.config.sha256=$CONFIG_SHA256 \
      org.alpasim.trt_manifest.sha256=$ENGINES_MANIFEST_SHA256 \
      org.alpasim.input.cameras="CAM_L0,CAM_F0,CAM_R0" \
      org.alpasim.eval.subject="stage3-vits512-g112-kv28" \
      org.alpasim.trt="tensorrt 10.13 cu11, image backbone FP16, PyTorch fallback"
ENV DRIVESUPRIM_CHECKPOINT_PATH=/app/assets/drivesuprim/pkg_stage3_best.ckpt \
    DRIVESUPRIM_CONFIG_PATH=/app/assets/drivesuprim/pkg_config.json \
    DRIVESUPRIM_PINHOLE_PROFILE=K377_512X256 \
    DRIVESUPRIM_TRT_BACKBONE=auto \
    DRIVESUPRIM_TRT_DIR=/app/assets/drivesuprim/trt \
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
