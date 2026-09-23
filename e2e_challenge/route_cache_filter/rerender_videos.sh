#!/usr/bin/env bash
# Re-render an arm's videos from its saved ASL logs, without re-simulating.
#
# KEEP_ROLLOUTS=1 leaves rollout.asl next to each clip, and that log carries the
# driver's debug payload -- the cached route, the route mask and the rejected
# trajectories included. So a change to how the overlay is *drawn* costs a
# re-render rather than another two hours of closed-loop simulation, which is
# what makes iterating on the picture affordable at all.
#
# The saved eval-config.yaml already carries the video settings the run used
# (render_video, parse_unstructured_debug_info, the panel layout), so nothing
# needs re-specifying here.
#
# Note what this cannot do: the `before` arm ran with the cache off, so its logs
# contain no route payload and re-rendering it will still show no overlay. That
# is the arm being a baseline, not the drawing failing.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ARM="${1:-after}"
RUN_DIR="$ROOT/runs/${RUN_TAG:-routecache}-$ARM"

[[ -d "$RUN_DIR" ]] || { echo "no such run: $RUN_DIR" >&2; exit 2; }
n_asl="$(find "$RUN_DIR/rollouts" -name 'rollout.asl' 2>/dev/null | wc -l)"
[[ "$n_asl" -gt 0 ]] || { echo "no rollout.asl under $RUN_DIR" >&2; exit 2; }
echo "re-rendering $n_asl rollout(s) in $RUN_DIR"

# Rendering is skipped when the mp4 is already there ("Delete it to re-render"),
# which is right for resuming a run and wrong for iterating on the drawing.
echo "removing $n_asl existing per-clip video(s) so they are redrawn"
find "$RUN_DIR/rollouts" -name '*.mp4' -delete
rm -rf "$RUN_DIR/aggregate/videos"

# The wizard runs aggregation inside the runtime image, which carries ffmpeg;
# reeval runs it on the host, which does not. The venv already ships a static
# build via imageio-ffmpeg, so put that on PATH rather than asking for a system
# package -- `_concatenate_videos` shells out to a bare "ffmpeg".
FFMPEG_EXE="$("$ROOT/.venv/bin/python3" -c \
    'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())' 2>/dev/null || true)"
if [[ -x "${FFMPEG_EXE:-}" ]] && ! command -v ffmpeg >/dev/null 2>&1; then
    shim="$(mktemp -d)"
    ln -s "$FFMPEG_EXE" "$shim/ffmpeg"
    export PATH="$shim:$PATH"
    echo "using bundled ffmpeg: $FFMPEG_EXE"
fi

cd "$ROOT"
exec uv run alpasim-reeval "$RUN_DIR"
