#!/usr/bin/env bash
# Prepare one AXEv1.0-NuRec three-camera bundle for the 441 evaluation.
#
#   MODEL=vits512_6layer e2e_challenge/pkg3cam_eval/prepare.sh
#
# CPU only, no image build. The evaluation runs the immutable axe-v9 image with
# the bundle's source, the adapter and the checkpoint bind-mounted read-only
# (run_pkg3cam.py), so nothing here needs a GPU or a 26 GB image layer.
#
# Writes runs/prepare-<subject>-20261008/:
#   drivesuprim_challenge/  axe-v9's adapter package + the 5cam_eval adapter
#                           changes (per-session ego_geom, axe-v9 MSDA setting,
#                           camera order from camera_layout) + 3-camera layout
#   assets/config.json      the bundle's stage-3 YAML composed for inference
#   model.json              everything run_pkg3cam.py needs
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
source "$HERE/models.sh"
BUNDLE="$ROOT/../models/$BUNDLE_NAME"
SRC="$BUNDLE/AXEv1.0-NuRec"
CKPT="$SRC/checkpoints/stage3_best.ckpt"
PREP="$ROOT/runs/prepare-${SUBJECT}-20261008"
log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }

[[ -f "$BUNDLE/SHA256SUMS" ]] || { log "no SHA256SUMS in $BUNDLE (incomplete archive?)"; exit 2; }
(cd "$BUNDLE" && sha256sum -c --quiet SHA256SUMS)
log "bundle checksums OK: $(wc -l < "$BUNDLE/SHA256SUMS") files"
SHA="$(sha256sum "$CKPT" | cut -d' ' -f1)"

rm -rf "$PREP/drivesuprim_challenge" "$PREP/assets"
mkdir -p "$PREP/drivesuprim_challenge" "$PREP/assets"
# ADAPTER_BASE_DIR: a copy of the base image's /app/drivesuprim_challenge taken
# earlier (same digest). Used when the docker daemon is too slow to create a
# container -- on 10-08 a near-full disk kept `docker create` hanging >10 min.
if [[ -n "${ADAPTER_BASE_DIR:-}" ]]; then
    cp -a "$ADAPTER_BASE_DIR/." "$PREP/drivesuprim_challenge/"
else
    cid="$(docker create "$BASE_IMAGE" true)"
    docker cp "$cid:/app/drivesuprim_challenge/." "$PREP/drivesuprim_challenge/"
    docker rm "$cid" >/dev/null
fi
cp "$ROOT/e2e_challenge/5cam_eval/drivesuprim_challenge/"{driver,policy,cnx_bev_bridge}.py "$PREP/drivesuprim_challenge/"
cp "$HERE/camera_layout.py" "$PREP/drivesuprim_challenge/"
sed -i 's/f"5-camera driver requires bev_num_cameras=/f"This driver requires bev_num_cameras=/' "$PREP/drivesuprim_challenge/policy.py"
if [[ "$PINHOLE_PROFILE" != K377_512X256 ]]; then
    "$ROOT/.venv/bin/python" "$HERE/add_pinhole_profile.py" "$PREP/drivesuprim_challenge/driver.py" "$PINHOLE_PROFILE"
fi
rm -rf "$PREP/drivesuprim_challenge/__pycache__"

# COMPOSE_ON_HOST=1 composes with the host venv's hydra instead of the image's;
# checked to produce a byte-identical config.json for vits512_6layer.
if [[ "${COMPOSE_ON_HOST:-0}" == 1 ]]; then
    "$ROOT/.venv/bin/python" "$HERE/compose_config.py" --source "$SRC" --agent "$AGENT" \
        --checkpoint /app/assets/drivesuprim/pkg_stage3_best.ckpt --output "$PREP/assets/config.json"
else
    docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp --entrypoint python \
        -v "$SRC:/bundle:ro" -v "$HERE:/tools:ro" -v "$PREP/assets:/output" "$BASE_IMAGE" \
        /tools/compose_config.py --source /bundle --agent "$AGENT" \
        --checkpoint /app/assets/drivesuprim/pkg_stage3_best.ckpt --output /output/config.json
fi

"$ROOT/.venv/bin/python" - "$PREP" <<PY
import json, sys, subprocess
from pathlib import Path
prep = Path(sys.argv[1])
cfg = json.loads((prep / "assets/config.json").read_text())
model = {
    "model": "$MODEL", "subject": "$SUBJECT", "run_name": "$RUN_NAME",
    "bundle": "$BUNDLE", "source": "$SRC", "agent": "$AGENT",
    "checkpoint": "$CKPT", "checkpoint_sha256": "$SHA",
    "bundle_manifest_sha256": subprocess.check_output(["sha256sum", "$BUNDLE/SHA256SUMS"], text=True).split()[0],
    "pinhole_profile": "$PINHOLE_PROFILE", "port": $PORT, "wizard_port": $WIZARD_PORT,
    "base_image": "$BASE_IMAGE",
    "config_summary": {k: cfg.get(k) for k in (
        "bev_num_cameras", "bev_img_width", "bev_img_height", "bev_ori_image_mode",
        "bev_num_encoder_layers", "bev_history_encoder_layers", "bev_h", "bev_w",
        "bevformer_img_backbone_type", "bevformer_vit_impl", "bevformer_vit_name",
        "bev_num_feature_levels", "feasibility_drivable_thresh",
        "feasibility_agent_conf_thresh", "use_route", "bev_pitch_correct")},
}
(prep / "model.json").write_text(json.dumps(model, indent=2) + "\n")
print(json.dumps(model["config_summary"], indent=1))
PY
log "prepared $PREP (checkpoint sha256 ${SHA:0:16}...)"
