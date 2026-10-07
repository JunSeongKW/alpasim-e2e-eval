# Per-model settings for the three-camera AXEv1.0-NuRec stage-3 bundles of
# 2026-10-08. Sourced by prepare.sh / smoke.sh; run_pkg3cam.py reads the same
# values back from the prepare directory's model.json.
#
# The agent config is the bundle's own stage-3 YAML whose name matches the
# bundle (verified against the checkpoint's tensor shapes):
#   vits512_6layer   -> ..._stage3_nurec_bev6          (6 BEV encoder layers)
#   vits768_3layer   -> ..._stage3_nurec_img768        (768x384, raw_fisheye)
#   vitl512_3layer   -> ..._stage3_nurec_vitl_img512_lr1 (DINOv2 ViT-L)
case "${MODEL:?MODEL=vits512_6layer|vits768_3layer|vitl512_3layer_lr1}" in
    vits512_6layer)
        BUNDLE_NAME=stage3_vits512_6layer_best_20261008
        AGENT=drivesuprim_agent_bevformer_vov_v2_vits_stage3_nurec_bev6
        SUBJECT=stage3-vits512-6layer
        PINHOLE_PROFILE=K377_512X256
        PORT=7600; WIZARD_PORT=24000 ;;
    vits768_3layer)
        BUNDLE_NAME=stage3_vits768_3layer_best_20261008
        AGENT=drivesuprim_agent_bevformer_vov_v2_vits_stage3_nurec_img768
        SUBJECT=stage3-vits768-3layer
        PINHOLE_PROFILE=K565P5_768X384
        PORT=7700; WIZARD_PORT=25000 ;;
    vitl512_3layer_lr1)
        BUNDLE_NAME=stage3_vitl512_3layer_lr1_best_20261008
        AGENT=drivesuprim_agent_bevformer_vov_v2_vits_stage3_nurec_vitl_img512_lr1
        SUBJECT=stage3-vitl512-3layer-lr1
        PINHOLE_PROFILE=K377_512X256
        PORT=7800; WIZARD_PORT=26000 ;;
    *) echo "unknown MODEL=$MODEL" >&2; return 2 2>/dev/null || exit 2 ;;
esac
BASE_IMAGE='696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/axe@sha256:85134c1ea9f09d140be610ca063c50ec60e99980c392e5bf81fe6563af503765'
RUN_NAME="leaderboard-${SUBJECT}-20261008"
