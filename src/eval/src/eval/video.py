# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025-2026 NVIDIA Corporation

import logging
import os
import traceback

import matplotlib as mpl
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.style as mplstyle
import matplotlib.transforms as transforms
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Polygon
import numpy as np
import polars as pl
from tqdm import tqdm

from eval.aggregation import processing
from eval.aggregation.processing import ProcessedMetricDFs
from eval.data import CameraProjector, ScenarioEvalInput, SimulationResult
from eval.schema import EgoLoc, EvalConfig, MapElements, VideoLayout
from eval.video_data import ShapelyMap
from eval.video_reasoning_overlay_utils import render_reasoning_overlay_style_video

logger = logging.getLogger("alpasim.eval.video")

mpl.use("Agg")
mplstyle.use("fast")

VIDEO_FILE_NAME_FORMAT = "{clipgt_id}_{rollout_id}_{camera_id}_{layout_id}.mp4"


def render_and_save_video(
    simulation_result: SimulationResult,
    processed_metric_dfs: ProcessedMetricDFs,
    output_dir: str,
    cfg: EvalConfig,
    clipgt_id: str,
    rollout_id: str,
) -> None:
    """
    Render and save video for a simulation result.

    This is the unified video rendering function that takes SimulationResult directly.

    Args:
        simulation_result: The simulation result to render.
        processed_metric_dfs: Processed metrics for display in the video.
        output_dir: Output directory for the video.
        cfg: Evaluation configuration.
        clipgt_id: Clip/ground truth identifier.
        rollout_id: Rollout identifier.
    """
    logger.info(
        "Rendering video for %s/%s",
        clipgt_id,
        rollout_id,
    )

    os.makedirs(output_dir, exist_ok=True)

    for video_layout in cfg.video.video_layouts:
        output_path = os.path.join(
            output_dir,
            VIDEO_FILE_NAME_FORMAT.format(
                clipgt_id=clipgt_id,
                rollout_id=rollout_id,
                camera_id=cfg.video.camera_id_to_render,
                layout_id=video_layout,
            ),
        )

        if os.path.exists(output_path):
            logger.info(
                "Video already exists, skipping %s. Delete it to re-render.",
                output_path,
            )
            continue

        if video_layout == VideoLayout.REASONING_OVERLAY:
            # Use reasoning overlay style rendering (camera, reasoning text overlay, trajectory chart)
            logger.info("Using reasoning overlay style video rendering")
            render_reasoning_overlay_style_video(
                simulation_result,
                processed_metric_dfs,
                output_path,
                cfg,
            )
        elif video_layout == VideoLayout.DEFAULT:
            # Use the default debug view rendering (bev map, camera, metrics)
            anim, fps = create_video_animation(
                processed_metric_dfs,
                simulation_result,
                cfg,
                clipgt_id=clipgt_id,
                rollout_id=rollout_id,
            )
            anim.save(
                output_path,
                fps=fps,
                dpi=100,
                writer="ffmpeg",
            )
            plt.close(anim._fig)
        else:
            raise ValueError(f"Unknown video layout: {video_layout}")


def render_video_from_eval_result(
    scenario_input: ScenarioEvalInput,
    metrics_df: pl.DataFrame | None,
    cfg: EvalConfig,
    output_dir: str,
    clipgt_id: str,
    rollout_id: str,
) -> bool:
    """
    Render video from evaluation result with full error handling.

    This is a convenience function that handles the full video rendering workflow:
    - Creates SimulationResult from ScenarioEvalInput
    - Processes metrics for video display
    - Renders and saves the video

    Args:
        scenario_input: The scenario evaluation input data.
        metrics_df: The metrics DataFrame from evaluation (can be None).
        cfg: Evaluation configuration.
        output_dir: Directory to save the video.
        clipgt_id: Clip/ground truth identifier.
        rollout_id: Rollout identifier.

    Returns:
        True if video was rendered successfully, False otherwise.
    """
    try:
        logger.info("Rendering video for %s/%s", clipgt_id, rollout_id)

        # Get SimulationResult for video rendering
        simulation_result = SimulationResult.from_scenario_input(scenario_input, cfg)

        # Process metrics for video (need ProcessedMetricDFs for video rendering)
        unprocessed_metrics = processing.UnprocessedMetricsDFs(metrics_df)
        processed_metrics = unprocessed_metrics.process()

        render_and_save_video(
            simulation_result=simulation_result,
            processed_metric_dfs=processed_metrics,
            output_dir=output_dir,
            cfg=cfg,
            clipgt_id=clipgt_id,
            rollout_id=rollout_id,
        )

        logger.info("Video saved to %s/videos/", output_dir)
        return True
    except Exception as e:
        logger.error("Error rendering video: %s", e)
        logger.error("Stacktrace: %s", traceback.format_exc())
        return False


def _setup_fig() -> tuple[plt.Figure, dict[str, plt.Axes]]:
    fig = plt.figure(figsize=(14, 10))
    fig.subplots_adjust(
        left=0.01, right=0.99, bottom=0.01, top=0.97, wspace=0.03, hspace=0.03
    )

    gs = gridspec.GridSpec(
        nrows=2,
        ncols=3,
        figure=fig,
        width_ratios=[1, 1, 0.55],
        height_ratios=[1, 1],
    )
    axs = {}
    axs["map"] = fig.add_subplot(gs[0, 0])
    axs["drivesuprim_bev"] = fig.add_subplot(gs[0, 1])
    axs["table"] = fig.add_subplot(gs[0, 2])
    axs["image"] = fig.add_subplot(gs[1, 0:2])
    # Ranking decomposition: which term of the score actually selected the
    # trajectory being driven. Shares the bottom row with the camera view.
    axs["rank"] = fig.add_subplot(gs[1, 2])
    # axs["plans"] = fig.add_subplot(gs[1, 2])
    axs["map"].set_xticks([])
    axs["map"].set_yticks([])
    axs["drivesuprim_bev"].set_xticks([])
    axs["drivesuprim_bev"].set_yticks([])
    axs["table"].set_xticks([])
    axs["table"].set_yticks([])
    axs["image"].set_xticks([])
    axs["image"].set_yticks([])
    axs["rank"].set_xticks([])
    axs["rank"].set_yticks([])

    axs["map"].set_aspect("equal")
    axs["drivesuprim_bev"].set_aspect("equal")
    # axs["plans"].set_aspect("equal")
    return fig, axs


_RANK_TERM_LABELS = {
    "imi": "il (imitation)",
    "no_at_fault_collisions": "nc (collision)",
    "drivable_area_compliance": "dac (drivable)",
    "gt_compliance": "gt (GT match)",
    "ego_progress": "ep (progress)",
    "pdm_score": "agg (aggregate)",
}
_RANK_TERM_ORDER = (
    "imi",
    "no_at_fault_collisions",
    "drivable_area_compliance",
    "gt_compliance",
    "ego_progress",
    "pdm_score",
)


def _setup_rank_panel(ax) -> dict:
    """Bars for each ranking term's influence on this frame's choice.

    Influence is the term's spread over the candidates as a share of all terms'
    spread. A term with no spread cannot reorder candidates no matter how large
    its value, so spread -- not magnitude -- is what selects a trajectory.
    """

    ax.set_title("ranking term influence (spread share)", fontsize=9)
    ax.set_xlabel("% of total spread across candidates", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.5, len(_RANK_TERM_ORDER) - 0.5)
    ax.set_yticks(range(len(_RANK_TERM_ORDER)))
    ax.set_yticklabels(
        [_RANK_TERM_LABELS[name] for name in _RANK_TERM_ORDER], fontsize=7
    )
    ax.invert_yaxis()
    bars = ax.barh(
        range(len(_RANK_TERM_ORDER)),
        [0.0] * len(_RANK_TERM_ORDER),
        color="#4a7fd4",
        height=0.6,
    )
    labels = [
        ax.text(
            1.0,
            index,
            "",
            va="center",
            fontsize=6,
            color="#222222",
        )
        for index in range(len(_RANK_TERM_ORDER))
    ]
    caption = ax.text(
        0.5,
        -0.16,
        "",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.5,
    )
    return {"ax": ax, "bars": bars, "labels": labels, "caption": caption}


def _update_rank_panel(artists, driver_response):
    """Redraw the influence bars for the current frame."""

    if artists is None:
        return []

    stats = getattr(driver_response, "drivesuprim_rank_stats", None) or {}
    selected = getattr(driver_response, "drivesuprim_rank_selected", None)
    changed = list(artists["bars"]) + list(artists["labels"])
    changed.append(artists["caption"])

    dominant = None
    for index, name in enumerate(_RANK_TERM_ORDER):
        entry = stats.get(name) or {}
        influence = float(entry.get("influence", 0.0) or 0.0)
        artists["bars"][index].set_width(influence)
        # A term that flips the choice when removed is decisive for THIS frame,
        # which is a stronger statement than a large share.
        flips = bool(entry.get("loo_flip", False))
        artists["bars"][index].set_color("#d4574a" if flips else "#4a7fd4")
        if entry:
            artists["labels"][index].set_text(
                f"{influence:5.1f}%  val={float(entry.get('value', 0.0)):+.2f}"
                f"  solo_rank={int(entry.get('solo_rank', 0))}"
                + ("  FLIPS" if flips else "")
            )
        else:
            artists["labels"][index].set_text("")
        if dominant is None or influence > dominant[1]:
            dominant = (name, influence)

    if stats and dominant is not None:
        flippers = [
            _RANK_TERM_LABELS[name].split()[0]
            for name in _RANK_TERM_ORDER
            if (stats.get(name) or {}).get("loo_flip")
        ]
        artists["caption"].set_text(
            f"driven candidate #{selected}   dominant: "
            f"{_RANK_TERM_LABELS[dominant[0]].split()[0]} "
            f"({dominant[1]:.0f}%)   flips: {', '.join(flippers) or 'none'}"
        )
    else:
        artists["caption"].set_text("no ranking debug payload for this frame")

    return changed


def _setup_drivesuprim_bev(
    ax: plt.Axes,
    cfg: EvalConfig,
) -> dict[str, object]:
    """Create reusable artists for model BEV perception and candidates."""
    radius_m = float(cfg.video.map_video.map_radius_m)
    if cfg.video.map_video.ego_loc == EgoLoc.BOTTOM_CENTER:
        forward_limits = (-0.25 * radius_m, 1.75 * radius_m)
    else:
        forward_limits = (-radius_m, radius_m)
    lateral_limits = (radius_m, -radius_m)
    ax.set_title("DriveSuprim perception + candidate BEV", fontsize=9)
    ax.set_facecolor("#f5f5f5")
    ax.set_xlabel("left  ←  lateral y (m)  →  right", fontsize=7)
    ax.set_ylabel("forward x (m)", fontsize=7)
    ax.grid(True, color="white", linewidth=0.5, alpha=0.8)
    drivable = ax.imshow(
        np.zeros((2, 2), dtype=np.float32),
        extent=[-32.0, 32.0, -5.0, 64.0],
        origin="lower",
        cmap=ListedColormap(["#ffffff", "#b8b8b8"]),
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
        alpha=0.68,
        zorder=1,
    )
    # Rejections are split by the rule that made them, because "filtered" on its
    # own does not say whether the gate was reading the map or the traffic.
    # Red vs orange read as one colour at this line width and opacity, so the two
    # rejection layers differ in hue AND dash pattern: solid red for the map,
    # dashed violet for traffic. Violet also stays clear of the pink obstacle
    # boxes and the yellow coarse layer.
    rejected_drivable = LineCollection(
        [], colors="#ff2d2d", linewidths=0.55, alpha=0.28, zorder=2
    )
    rejected_collision = LineCollection(
        [], colors="#a855f7", linewidths=0.65, alpha=0.40, zorder=2,
        linestyles=(0, (2.5, 2.0)),
    )
    survived = LineCollection(
        [], colors="#78add2", linewidths=0.5, alpha=0.30, zorder=3
    )
    # The route-corridor gate. Its rejections get their own layer rather than
    # joining the drivable/collision ones: those two come from the network, this
    # one is applied afterwards against remembered route, and reading whether it
    # fired is the entire point of the comparison it was built for. Cyan is the
    # only hue left that neither the map (red), the traffic (violet), the coarse
    # set (yellow) nor the driven path (green) is already using.
    # Rejected candidates sit ABOVE the surviving fan and the coarse set, not
    # below them: a rejection drawn under two thousand pale blue lines is
    # invisible however bright it is, and the whole reason this layer exists is
    # to be read off the frame. Magenta because every other hue in the panel is
    # already spoken for -- red is the drivable rule, violet the collision one,
    # yellow the coarse set, green the driven path, cyan the route itself.
    rejected_route = LineCollection(
        [], colors="#ff2bb5", linewidths=1.1, alpha=0.55, zorder=5.5,
        linestyles=(0, (5, 5)),
    )
    # Only this leading segment was evaluated by the route gate. Draw it in a
    # separate, solid colour so a viewer cannot mistake the magenta 1--4 s
    # continuation for evidence that the gate inspected those future poses.
    rejected_route_checked = LineCollection(
        [], colors="#ff7a00", linewidths=1.35, alpha=0.72, zorder=5.7,
    )
    # The corridor the gate actually measures against, drawn as a band rather
    # than described in the legend. Without it the viewer sees lines being
    # thrown away and has to take the 4 m on trust.
    route_corridor = Polygon(
        np.zeros((3, 2)), closed=True, facecolor="#00e5ff", edgecolor="none",
        alpha=0.13, zorder=1.5,
    )
    ax.add_patch(route_corridor)
    cached_route = ax.plot(
        [], [], color="#00e5ff", linewidth=3.2, alpha=1.0, zorder=7.5,
        linestyle=(0, (7, 3)),
    )[0]
    # Route reranker. The far route message (42-80 m ahead) is what the mean-L2
    # cost is measured against; drawn dotted so it cannot be confused with the
    # cached route above. The model's winner BEFORE reranking is drawn only on
    # frames where the reranker replaced it, in orange over the green final
    # path, so a change is visible as two lines and a kept frame as one.
    route_message = ax.plot(
        [], [], color="#00e5ff", linewidth=2.0, alpha=0.95, zorder=7.4,
        linestyle=(0, (1.5, 2.0)), marker="o", markersize=2.5,
    )[0]
    rerank_original = ax.plot(
        [], [], color="#ff7a00", linewidth=3.0, alpha=0.95, zorder=6.4,
        linestyle=(0, (6, 3)),
    )[0]
    coarse = LineCollection(
        [], colors="#ffd400", linewidths=0.75, alpha=0.34, zorder=5
    )
    obstacles = PolyCollection(
        [], facecolors="#f5a3a3", edgecolors="#111111",
        linewidths=1.0, alpha=0.55, zorder=4,
    )
    ax.add_collection(rejected_drivable)
    ax.add_collection(rejected_collision)
    ax.add_collection(rejected_route)
    ax.add_collection(rejected_route_checked)
    ax.add_collection(survived)
    ax.add_collection(obstacles)
    ax.add_collection(coarse)
    final = ax.plot([], [], color="#00a63c", linewidth=3.0, alpha=1.0, zorder=6)[0]
    # What the person actually drove over the same 4 s, in the model's own ego
    # frame, so the chosen candidate and the recording are read off one picture.
    # This is the trajectory the imitation head was trained against; the map
    # panel shows the same thing in world coordinates.
    gt = ax.plot(
        [], [], color="#ffffff", linewidth=2.2, alpha=0.95,
        linestyle=(0, (4, 2)), zorder=8,
    )[0]
    ego = ax.plot(0.0, 0.0, marker="^", markersize=8, color="black", zorder=7)[0]
    status = ax.text(
        0.02,
        0.98,
        "No DriveSuprim debug data",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8),
        zorder=8,
    )
    ax.legend(
        handles=[
            Line2D([0], [0], color="#00a63c", lw=3, label="final refinement"),
            Line2D([0], [0], color="#ffffff", lw=2.2, linestyle=(0, (4, 2)),
                   label="recorded human (GT)"),
            Line2D([0], [0], color="#ffd400", lw=2, label="coarse top-256"),
            Line2D(
                [0], [0], color="#78add2", lw=1, alpha=0.8,
                label="feasibility survivors",
            ),
            Line2D([0], [0], color="#00e5ff", lw=3.2, linestyle=(0, (7, 3)),
                   label="cached route"),
            Patch(facecolor="#00e5ff", edgecolor="none", alpha=0.2,
                  label="route corridor +-4 m"),
            Line2D([0], [0], color="#ff7a00", lw=1.7,
                   label="route-gate checked segment"),
            Line2D([0], [0], color="#ff2bb5", lw=1.7, linestyle=(0, (4, 2)),
                   label="rejected continuation (unchecked)"),
            Line2D([0], [0], color="#00e5ff", lw=2.0, linestyle=(0, (1.5, 2.0)),
                   marker="o", markersize=3, label="route message (42-80 m)"),
            Line2D([0], [0], color="#ff7a00", lw=3.0, linestyle=(0, (6, 3)),
                   label="model winner before rerank"),
            Patch(
                facecolor="#f5a3a3", edgecolor="#111111", alpha=0.55,
                label="detected object",
            ),
        ],
        loc="lower right",
        fontsize=6,
        framealpha=0.8,
    )
    ax.set_xlim(*lateral_limits)
    ax.set_ylim(*forward_limits)
    return {
        "drivable": drivable,
        "gt": gt,
        "rejected_drivable": rejected_drivable,
        "rejected_collision": rejected_collision,
        "survived": survived,
        "rejected_route": rejected_route,
        "rejected_route_checked": rejected_route_checked,
        "route_corridor": route_corridor,
        "cached_route": cached_route,
        "route_message": route_message,
        "rerank_original": rerank_original,
        "obstacles": obstacles,
        "coarse": coarse,
        "final": final,
        "ego": ego,
        "status": status,
        "last_response": None,
        "lateral_limits": lateral_limits,
        "forward_limits": forward_limits,
    }


# The scorer's corridor half-width. Hard-coded rather than read from the eval
# config because this is the driver-side filter's threshold being visualised,
# and the two are deliberately the same number for a reason the picture should
# show; if they ever diverge, the band must follow the driver, not the scorer.
ROUTE_CORRIDOR_HALF_WIDTH_M = 4.0

# How many rejected candidates to draw before thinning them out.
ROUTE_REJECTED_DRAW_LIMIT = 150


def _corridor_band(route_xy: np.ndarray, half_width_m: float) -> np.ndarray:
    """Polygon of everything within `half_width_m` of the route, panel-oriented.

    Offsetting along the segment normals rather than buffering keeps this cheap
    enough to redo every frame. The band self-intersects on a curve tighter than
    its own width; that reads fine at 13% opacity and is not worth a real
    polygon offset to avoid.
    """
    d = np.gradient(route_xy, axis=0)
    normal = np.stack([-d[:, 1], d[:, 0]], axis=1)
    length = np.linalg.norm(normal, axis=1, keepdims=True)
    normal = normal / np.where(length > 1e-9, length, 1.0)
    left = route_xy + half_width_m * normal
    right = route_xy - half_width_m * normal
    ring = np.concatenate([left, right[::-1]], axis=0)
    return ring[:, [1, 0]]          # panel plots lateral on x, forward on y


def _set_gt_line(artists: dict, gt_future_xy) -> None:
    """Draw the recorded drive in panel coordinates (lateral on x, forward on y)."""

    line = artists.get("gt")
    if line is None:
        return
    if gt_future_xy is None or len(gt_future_xy) < 2:
        line.set_data([], [])
        return
    line.set_data(gt_future_xy[:, 1], gt_future_xy[:, 0])


def _gt_future_in_ego_frame(
    gt_trajectory,
    ego_trajectory,
    now_us: int,
    horizon_s: float = 4.0,
    step_s: float = 0.1,
):
    """The recorded drive over the next `horizon_s`, expressed in the ego frame.

    The BEV panel is drawn in the model's own coordinates -- x forward, y left,
    origin at the current rear axle -- so the recording has to be pulled into
    that frame before it can be compared against the candidates. The origin is
    the pose the ROLLOUT reached, not the recorded one, so once the model has
    drifted the line shows where the person was relative to where the car
    actually is, which is the comparison that matters.

    Returns None when the rollout has run past the end of the recording, which
    is ordinary near the end of a clip rather than an error.
    """

    if gt_trajectory is None or ego_trajectory is None:
        return None

    try:
        stamps = (
            int(now_us) + (np.arange(0.0, horizon_s + 1e-9, step_s) * 1e6)
        ).astype(np.uint64)
        last_us = np.uint64(int(np.asarray(gt_trajectory.timestamps_us)[-1]))
        stamps = stamps[stamps <= last_us]
        if stamps.size < 2:
            return None

        gt = gt_trajectory.interpolate(stamps)
        ego = ego_trajectory.interpolate(np.array([int(now_us)], dtype=np.uint64))
        origin = np.asarray(ego.positions, dtype=np.float64)[0, :2]
        yaw = float(np.asarray(ego.yaws, dtype=np.float64)[0])
        points = np.asarray(gt.positions, dtype=np.float64)[:, :2]
    except Exception:
        return None

    cos_y, sin_y = np.cos(-yaw), np.sin(-yaw)
    rotation = np.array([[cos_y, -sin_y], [sin_y, cos_y]])
    return (points - origin) @ rotation.T


def _update_drivesuprim_bev(
    ax: plt.Axes,
    artists: dict[str, object],
    driver_response,
    gt_future_xy=None,
) -> list[plt.Artist]:
    """Update perception and candidate overlays in model ego coordinates."""
    visible_artists = [
        artists["gt"],
        artists["drivable"],
        artists["rejected_drivable"],
        artists["rejected_collision"],
        artists["rejected_route"],
        artists["rejected_route_checked"],
        artists["route_corridor"],
        artists["cached_route"],
        artists["route_message"],
        artists["rerank_original"],
        artists["survived"],
        artists["obstacles"],
        artists["coarse"],
        artists["final"],
        artists["ego"],
        artists["status"],
    ]
    if artists["last_response"] is driver_response:
        return visible_artists
    artists["last_response"] = driver_response

    if driver_response is None:
        artists["drivable"].set_data(np.zeros((2, 2), dtype=np.float32))
        artists["rejected_drivable"].set_segments([])
        artists["rejected_collision"].set_segments([])
        artists["rejected_route"].set_segments([])
        artists["rejected_route_checked"].set_segments([])
        artists["cached_route"].set_data([], [])
        artists["route_message"].set_data([], [])
        artists["rerank_original"].set_data([], [])
        artists["route_corridor"].set_visible(False)
        artists["survived"].set_segments([])
        artists["obstacles"].set_verts([])
        artists["coarse"].set_segments([])
        artists["final"].set_data([], [])
        artists["status"].set_text("No DriveSuprim debug data")
        _set_gt_line(artists, gt_future_xy)
        return visible_artists + [artists["gt"]]

    pc_range = driver_response.drivesuprim_point_cloud_range
    if pc_range is not None:
        x_min, y_min, _, x_max, y_max, _ = pc_range
        drivable = driver_response.drivesuprim_drivable_probability
        if drivable is not None:
            # The driver transports probabilities as uint8 [0, 255]. Render a
            # crisp binary drivable-area mask at P >= 0.5; this changes only
            # ClipGT visualization, not the model's feasibility computation.
            drivable_mask = (
                np.asarray(drivable, dtype=np.uint8) >= 128
            ).astype(np.float32)
            artists["drivable"].set_data(
                drivable_mask.T
            )
            artists["drivable"].set_extent([y_min, y_max, x_min, x_max])
        # Keep the same metric viewport as AlpaSim's BEV panel. The full
        # DriveSuprim raster remains attached to its native metric extent and
        # is clipped by these limits without altering model resolution.
        ax.set_xlim(*artists["lateral_limits"])
        ax.set_ylim(*artists["forward_limits"])

    vocab = driver_response.drivesuprim_candidate_vocab
    mask = driver_response.drivesuprim_feasibility_mask
    selected_index = driver_response.drivesuprim_selected_index
    rejected_count = 0
    drivable_reject_count = 0
    collision_reject_count = 0
    feasible_count = 0
    candidate_count = 0
    if vocab is not None:
        paths = np.asarray(vocab, dtype=np.float32)[..., :2]
        segments = paths[..., [1, 0]]
        candidate_count = len(paths)
        if mask is None or len(mask) != candidate_count:
            raw_mask = np.ones(candidate_count, dtype=np.bool_)
        else:
            raw_mask = np.asarray(mask, dtype=np.bool_)
        feasible = raw_mask.copy()
        if selected_index is not None and 0 <= selected_index < len(feasible):
            feasible[selected_index] = False

        # Split the rejections by rule. A candidate can fail both; attribute it
        # to the drivable-area rule so the two layers stay disjoint and the
        # counts in the status line add up to the total rejected.
        drv = driver_response.drivesuprim_drivable_mask
        col = driver_response.drivesuprim_collision_mask
        off_drivable = (
            ~np.asarray(drv, dtype=np.bool_)
            if drv is not None and len(drv) == candidate_count
            else np.zeros(candidate_count, dtype=np.bool_)
        )
        hits_agent = (
            ~np.asarray(col, dtype=np.bool_)
            if col is not None and len(col) == candidate_count
            else np.zeros(candidate_count, dtype=np.bool_)
        )
        hits_agent &= ~off_drivable
        # Anything rejected that neither rule claims (e.g. the model exported the
        # combined mask only) is drawn with the drivable-area colour rather than
        # being dropped from the picture.
        unattributed = (~raw_mask) & ~off_drivable & ~hits_agent
        off_drivable = off_drivable | unattributed

        # Rejected candidates are counted but not drawn. Two thousand extra
        # polylines bury the layers that carry the decision -- the coarse top-k
        # and the final path -- and the status line already reports how many the
        # gate dropped and under which rule.
        artists["rejected_drivable"].set_segments([])
        artists["rejected_collision"].set_segments([])
        artists["survived"].set_segments(segments[raw_mask])
        rejected_count = int((~raw_mask).sum())
        drivable_reject_count = int(off_drivable.sum())
        collision_reject_count = int(hits_agent.sum())
        feasible_count = int(feasible.sum())
    else:
        artists["rejected_drivable"].set_segments([])
        artists["rejected_collision"].set_segments([])
        artists["survived"].set_segments([])

    coarse_candidates = driver_response.drivesuprim_coarse_candidates
    coarse_count = 0
    if coarse_candidates is not None and len(coarse_candidates):
        coarse_candidates = np.asarray(coarse_candidates, dtype=np.float32)
        coarse_segments = coarse_candidates[..., :2][..., [1, 0]]
        artists["coarse"].set_segments(coarse_segments)
        coarse_count = len(coarse_segments)
    else:
        artists["coarse"].set_segments([])
    final = driver_response.drivesuprim_final_path
    if final is not None and len(final):
        final = np.asarray(final)
        artists["final"].set_data(final[:, 1], final[:, 0])
    else:
        artists["final"].set_data([], [])

    _set_gt_line(artists, gt_future_xy)

    agents = driver_response.drivesuprim_detected_agents
    obstacle_segments = []
    if agents is not None:
        for cx, cy, yaw, length, width, class_index, _confidence in np.asarray(agents):
            half_l, half_w = 0.5 * float(length), 0.5 * float(width)
            local_x = np.array([half_l, half_l, -half_l, -half_l, half_l])
            local_y = np.array([half_w, -half_w, -half_w, half_w, half_w])
            cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
            box_x = cx + cos_yaw * local_x - sin_yaw * local_y
            box_y = cy + sin_yaw * local_x + cos_yaw * local_y
            obstacle_segments.append(np.column_stack((box_y, box_x)))
    artists["obstacles"].set_verts(obstacle_segments)

    # Route-corridor gate. The panel plots lateral on x and forward on y, so
    # both layers are swapped into that order like every other overlay here.
    cached_route = driver_response.drivesuprim_cached_route
    if cached_route is not None and len(cached_route) >= 2:
        cached_route = np.asarray(cached_route, dtype=np.float32)
        artists["cached_route"].set_data(cached_route[:, 1], cached_route[:, 0])
        artists["route_corridor"].set_xy(
            _corridor_band(cached_route, ROUTE_CORRIDOR_HALF_WIDTH_M)
        )
        artists["route_corridor"].set_visible(True)
    else:
        artists["cached_route"].set_data([], [])
        artists["route_corridor"].set_visible(False)

    route_rejected = driver_response.drivesuprim_route_rejected_paths
    if route_rejected is not None and len(route_rejected):
        route_rejected = np.asarray(route_rejected, dtype=np.float32)
        # Thinned for legibility, not for size: drawn at this weight the whole
        # set becomes a solid magenta wedge and stops showing which way the
        # discarded trajectories went, which is the only thing it is here for.
        if len(route_rejected) > ROUTE_REJECTED_DRAW_LIMIT:
            route_rejected = route_rejected[
                np.linspace(0, len(route_rejected) - 1,
                            ROUTE_REJECTED_DRAW_LIMIT).astype(int)
            ]
        gate_poses = driver_response.drivesuprim_route_gate_poses
        if gate_poses is not None:
            gate_poses = max(1, min(int(gate_poses), route_rejected.shape[1]))
            checked = route_rejected[:, :gate_poses]
            # Repeat the boundary point in the continuation so the two colours
            # meet without a one-sample visual gap.
            continuation = route_rejected[:, max(gate_poses - 1, 0):]
            artists["rejected_route_checked"].set_segments(
                checked[..., [1, 0]]
            )
            artists["rejected_route"].set_segments(
                continuation[..., [1, 0]]
            )
        else:
            artists["rejected_route_checked"].set_segments([])
            artists["rejected_route"].set_segments(route_rejected[..., [1, 0]])
    else:
        artists["rejected_route"].set_segments([])
        artists["rejected_route_checked"].set_segments([])

    route_mask = driver_response.drivesuprim_route_mask
    if route_mask is None:
        route_line = "route gate: no cached route yet"
    else:
        route_mask = np.asarray(route_mask, dtype=bool)
        route_line = (
            f"route gate: kept {int(route_mask.sum()):,}/{route_mask.size:,}"
            f" | rejected {int((~route_mask).sum()):,}"
            f" | checked poses "
            f"{driver_response.drivesuprim_route_gate_poses or 'N/A'}"
            f"{' | RESELECTED' if driver_response.drivesuprim_route_reselected else ''}"
        )

    # Route reranker: the far route it measured against, and the model's own
    # winner on frames where the reranker overrode it.
    route_message = driver_response.drivesuprim_route_message
    if route_message is not None and len(route_message) >= 2:
        route_message = np.asarray(route_message, dtype=np.float32)
        artists["route_message"].set_data(route_message[:, 1], route_message[:, 0])
    else:
        artists["route_message"].set_data([], [])
    rerank = driver_response.drivesuprim_route_rerank
    if rerank is None:
        rerank_line = "rerank: off"
        artists["rerank_original"].set_data([], [])
    else:
        changed = bool(rerank.get("changed", False))
        original = rerank.get("original_traj")
        if changed and original is not None and len(original) >= 2:
            original = np.asarray(original, dtype=np.float32)
            artists["rerank_original"].set_data(original[:, 1], original[:, 0])
        else:
            artists["rerank_original"].set_data([], [])
        rerank_line = (
            f"rerank: {'CHANGED' if changed else 'kept'}"
            f" | eligible {int(rerank.get('comparable_count', 0))}"
            f"/{int(rerank.get('candidate_count', 0))}"
            f" | winner eligible {'yes' if rerank.get('comparable_original') else 'no'}"
            f" | orig #{int(rerank.get('original_index', -1))}"
            f" meanL2 {float(rerank.get('cost_original', float('nan'))):.1f} m"
            f" | sel #{int(rerank.get('selected_index', -1))}"
            f" meanL2 {float(rerank.get('cost_selected', float('nan'))):.1f} m"
        )

    # `survived` is the AND of every gate, so the per-rule counts below it are
    # each that rule ALONE -- they overlap and do not sum to the total dropped.
    # Spelling that out beats three numbers that look like they should add up.
    artists["status"].set_text(
        f"candidates {candidate_count:,} | "
        f"survived all gates {int(raw_mask.sum()) if candidate_count else 0:,}\n"
        f"dropped by rule (overlapping): drivable {drivable_reject_count:,}"
        f" / collision {collision_reject_count:,}\n"
        f"{route_line}\n"
        f"{rerank_line}\n"
        f"coarse {coarse_count:,} | objects {len(obstacle_segments)} | "
        f"selected #{selected_index if selected_index is not None else 'N/A'}"
    )
    return visible_artists


def _list_in_dict_in_dict_to_list(
    artist_map: dict[str, dict[str, list[plt.Artist]]],
) -> list[plt.Artist]:
    all_artists = []
    for sub_dict in artist_map.values():
        for list_of_artists in sub_dict.values():
            all_artists.extend(list_of_artists)
    return all_artists


def _compute_frame_timing(
    timestamps_us: np.ndarray,
    render_every_nth_frame: int,
) -> tuple[float, float]:
    """Derive animation interval (ms) and FPS from simulation timestamps."""
    if render_every_nth_frame < 1:
        raise ValueError("render_every_nth_frame must be at least 1")
    if len(timestamps_us) <= 1:
        raise ValueError("At least 2 timestamps are required")

    deltas_us = np.diff(timestamps_us.astype(np.int64))
    if not np.all(deltas_us == deltas_us[0]):
        logger.warning(
            "Timestamp deltas are not uniform: %s. Using median delta for frame timing.",
            deltas_us,
        )
    base_delta_us = float(np.median(deltas_us))
    frame_delta_us = base_delta_us * render_every_nth_frame

    fps = max(1e-6, 1_000_000.0 / frame_delta_us)
    interval_ms = frame_delta_us / 1_000.0
    return interval_ms, fps


def get_ego_transform(
    sim_result: SimulationResult,
    cfg: EvalConfig,
    time: int,
) -> transforms.Affine2D:
    ego_transform = transforms.Affine2D()
    if cfg.video.map_video.rotate_map_to_ego:
        ego_yaw = float(
            np.asarray(
                sim_result.actor_trajectories["EGO"]
                .interpolate_to_timestamps(np.array([time]))
                .yaws
            )[0]
        )
        ego_transform = ego_transform.rotate(np.pi / 2 - ego_yaw)

    return ego_transform


def render_table(
    ax: plt.Axes,
    processed_metric_dfs: ProcessedMetricDFs,
    clipgt_id: str,
    rollout_id: str,
    time: int,
    metrics_table_entries: list[str] | None = None,
) -> mpl.table.Table:

    run_name = processed_metric_dfs.trajectory_uid_df["run_name"][0]
    # Prepare aggregated data
    df_long_avg_t = (
        processed_metric_dfs.df_wide_avg_t.drop(
            "rollout_id",
            "clipgt_id",
            "run_name",
            "run_uuid",
            "trajectory_uid",
            "rollout_uid",
        )
        .unpivot()
        .sort("variable")
    )

    available_metric_names = df_long_avg_t["variable"].to_list()
    metric_names = (
        available_metric_names
        if metrics_table_entries is None
        else [
            metric_name
            for metric_name in metrics_table_entries
            if metric_name in available_metric_names
        ]
    )
    avg_value_by_metric = {
        row["variable"]: row["value"] for row in df_long_avg_t.iter_rows(named=True)
    }
    agg_function_by_metric = {
        row["name"]: row["time_aggregation"]
        for row in processed_metric_dfs.agg_function_df.iter_rows(named=True)
    }

    filtered_df_long = processed_metric_dfs.unprocessed_df.filter(
        pl.col("timestamps_us") == time,
    )

    assert (
        len(processed_metric_dfs.df_wide_avg_t) == 1
    ), f"Expected 1 row in df_wide_avg_t, got {len(processed_metric_dfs.df_wide_avg_t)}"
    assert (
        len(processed_metric_dfs.trajectory_uid_df) == 1
    ), f"Expected 1 row in trajectory_uid_df, got {len(processed_metric_dfs.trajectory_uid_df)}"
    # One row per metric
    assert len(filtered_df_long) == len(
        filtered_df_long["name"].unique()
    ), "Expected all metrics to be present in filtered_df_long"

    ax.axis("off")

    # Extract data from polars dataframe
    table_data = []
    # headers = ['Metric Name', 'Metric Value', 'Time Aggregation']
    col_names = ["Agg", "Per-Ts"]
    row_name = []

    for metric_name in metric_names:
        row_name.append(metric_name)
        curr_df = filtered_df_long.filter(
            pl.col("name") == metric_name,
        )
        assert len(curr_df) <= 1
        # We might not have per-ts values for all ts.
        value_str = "N/A" if len(curr_df) == 0 else f"{curr_df['values'][0]:.2f}"
        # We also might not have a value for any ts.
        agg_value = avg_value_by_metric.get(metric_name)
        agg_value_str = "N/A" if agg_value is None else f"{agg_value:.2f}"
        agg_str = agg_function_by_metric.get(metric_name)
        agg_str_display = agg_str if agg_str is not None else "N/A"
        table_data.append([f"{agg_value_str} ({agg_str_display})", value_str])

    table = ax.table(
        cellText=table_data,
        colLabels=col_names,
        rowLabels=row_name,
        loc="center right",
        cellLoc="center",
        rowLoc="left",
        edges="horizontal",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.2, 1.5)

    # Style header
    for i in range(len(col_names)):
        table[(0, i)].set_text_props(weight="bold")
        table[(0, i)].set_facecolor("#E6E6E6")
    # Style row labels
    for i in range(len(row_name)):
        table[(i + 1, -1)].set_text_props(weight="bold")
        table[(i + 1, -1)].set_facecolor("#E6E6E6")
    # Make table more narrow by adjusting column widths
    table.auto_set_column_width([0, 1])  # Reduce width multiplier for both columns

    # Add title with run name and clipgt id
    ax.text(
        0.0,
        1.0,
        f"Run: {run_name}\nClip: {clipgt_id}\nRollout: {rollout_id}",
        ha="left",
        va="top",
        fontsize=6,
        transform=ax.transAxes,
    )

    # Remove top and bottom edges
    for col in range(len(col_names)):
        # Top row cells - remove top edge
        top_cell = table.get_celld()[(0, col)]
        top_cell.visible_edges = top_cell.visible_edges.replace("T", "")

    if row_name:
        for col in range(len(col_names) + 1):
            # Bottom row cells - remove bottom edge
            bottom_cell = table.get_celld().get((len(row_name), col - 1))
            if bottom_cell is not None:
                bottom_cell.visible_edges = bottom_cell.visible_edges.replace("B", "")

    return table


def update_table(
    table: mpl.table.Table,
    processed_dfs: ProcessedMetricDFs,
    time: int,
) -> mpl.table.Table:
    celld = table.get_celld()
    # First row is header, column names are column -1!
    n_rows = max(map(lambda coords: coords[0], celld.keys())) + 1

    metric_names = [celld[(row, -1)].get_text().get_text() for row in range(1, n_rows)]

    # Want to update only the Value(ts) cells, which are column 1

    for row, metric_name in enumerate(metric_names, start=1):
        curr_df_long = processed_dfs.unprocessed_df.filter(
            pl.col("timestamps_us") == time,
            pl.col("name") == metric_name,
        )

        assert len(curr_df_long) <= 1
        value_str = (
            "N/A" if len(curr_df_long) == 0 else f"{curr_df_long['values'][0]:.2f}"
        )

        celld[(row, 1)].get_text().set_text(value_str)

    return table


def _driver_state_overlay_text(driver_response) -> str:
    """Format optional ego speed/acceleration fields from driver debug info."""
    if driver_response is None or driver_response.speed_mps is None:
        return ""
    text = f"Speed: {driver_response.speed_mps:.2f} m/s"
    if (
        driver_response.acceleration_longitudinal_mps2 is not None
        and driver_response.acceleration_lateral_mps2 is not None
    ):
        text += (
            "\nAccel: "
            f"long {driver_response.acceleration_longitudinal_mps2:+.2f}, "
            f"lat {driver_response.acceleration_lateral_mps2:+.2f} m/s²"
        )
    return text


def create_video_animation(
    processed_metrics_dfs: ProcessedMetricDFs,
    sim_result: SimulationResult,
    cfg: EvalConfig,
    clipgt_id: str = "unknown",
    rollout_id: str = "unknown",
) -> tuple[animation.FuncAnimation, float]:
    """
    Create a video animation for a simulation result.

    Args:
        processed_metrics_dfs: Processed metrics for display in the video.
        sim_result: The simulation result to visualize.
        cfg: Evaluation configuration.
        clipgt_id: Clip/ground truth identifier (for table display).
        rollout_id: Rollout identifier (for table display).

    Returns:
        Tuple of (animation, fps).
    """
    timestamps_us = sim_result.timestamps_us
    camera = sim_result.cameras.camera_by_logical_id[cfg.video.camera_id_to_render]
    shapely_map = ShapelyMap.from_vec_map(sim_result.vec_map)
    should_render_table = processed_metrics_dfs.df_wide_avg_t.shape[0] > 0

    fig, axs = _setup_fig()
    drivesuprim_bev_artists = _setup_drivesuprim_bev(
        axs["drivesuprim_bev"], cfg
    )
    rank_artists = _setup_rank_panel(axs["rank"])

    first_image = camera.image_at_time(timestamps_us[0])
    img_w = first_image.size[0] if first_image else None
    img_h = first_image.size[1] if first_image else None
    camera.render_image_at_time(timestamps_us[0], axs["image"])
    if img_w is not None and img_h is not None:
        axs["image"].set_xlim(0, img_w)
        axs["image"].set_ylim(img_h, 0)
        axs["image"].set_autoscale_on(False)

    overlay_enabled = cfg.video.overlay_plans_on_camera
    camera_projector: CameraProjector | None = None
    if overlay_enabled:
        if not sim_result.driver_responses.per_timestep_driver_responses:
            logger.info("No driver responses found; disabling camera overlay.")
            overlay_enabled = False
        else:
            calibration = sim_result.cameras.calibrations_by_logical_id.get(
                cfg.video.camera_id_to_render
            )
            if calibration is None:
                logger.warning(
                    "No calibration for camera %s; disabling camera overlay.",
                    cfg.video.camera_id_to_render,
                )
                overlay_enabled = False
            else:
                try:
                    camera_projector = CameraProjector(
                        calibration=calibration,
                        actual_resolution=first_image.size if first_image else None,
                    )
                except ValueError as exc:
                    logger.warning(
                        "Unsupported calibration for camera %s (%s); "
                        "disabling camera overlay.",
                        cfg.video.camera_id_to_render,
                        exc,
                    )
                    overlay_enabled = False

    if overlay_enabled:
        overlay_frame_matches = np.intersect1d(
            timestamps_us, sim_result.driver_responses.timestamps_us
        )
        if len(overlay_frame_matches) == 0:
            logger.info(
                "Driver response timestamps do not align with rendered frames; "
                "camera overlay will be empty."
            )

    if should_render_table:
        table = render_table(
            axs["table"],
            processed_metrics_dfs,
            clipgt_id,
            rollout_id,
            timestamps_us[0],
            cfg.video.metrics_table_entries,
        )

    text_artist = axs["table"].text(
        0.00,
        0.00,
        f"Timestamp_us: {timestamps_us[0]}",
        ha="left",
        va="bottom",
        transform=axs["table"].transAxes,
        fontsize=6,
    )

    # Get initial command name from driver response
    initial_driver_response = sim_result.driver_responses.get_driver_response_for_time(
        timestamps_us[0], which_time="now"
    )
    _update_drivesuprim_bev(
        axs["drivesuprim_bev"],
        drivesuprim_bev_artists,
        initial_driver_response,
        gt_future_xy=_gt_future_in_ego_frame(
            sim_result.ego_recorded_ground_truth_trajectory,
            sim_result.actor_trajectories["EGO"],
            timestamps_us[0],
        ),
    )
    _update_rank_panel(rank_artists, initial_driver_response)
    initial_command = (
        initial_driver_response.command_name
        if initial_driver_response and initial_driver_response.command_name
        else None
    )
    initial_state_text = _driver_state_overlay_text(initial_driver_response)
    command_text_artist = axs["image"].text(
        0.02,
        0.98,
        f"Command: {initial_command}" if initial_command else "",
        ha="left",
        va="top",
        transform=axs["image"].transAxes,
        fontsize=10,
        color="white",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
        visible=initial_command is not None,
    )
    state_text_artist = axs["image"].text(
        0.02,
        0.91,
        initial_state_text,
        ha="left",
        va="top",
        transform=axs["image"].transAxes,
        fontsize=9,
        color="white",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7),
        visible=bool(initial_state_text),
    )

    ego_transform = get_ego_transform(
        sim_result=sim_result,
        cfg=cfg,
        time=timestamps_us[0],
    )

    image_center_xy = sim_result.actor_polygons.set_axis_limits_around_agent(
        axs["map"],
        "EGO",
        timestamps_us[0],
        cfg,
        axis_transform=ego_transform,
    )

    # Outer key: name of the element to plot
    # Inner key: name of the element artists to plot (e.g. border and fill)
    artists_on_map: dict[str, dict[str, list[plt.Artist]]] = {}

    artists_on_map["map"] = shapely_map.render(
        axs["map"],
        cfg,
        center=image_center_xy,
        max_dist=cfg.video.map_video.map_radius_m + 10,
    )

    if (
        cfg.video.map_video.map_elements_to_plot is None
        or MapElements.GT_LINESTRING in cfg.video.map_video.map_elements_to_plot
    ):
        artists_on_map["gt_linestring"] = (
            sim_result.ego_recorded_ground_truth_trajectory.set_linestring_plot_style(
                "gt_linestring",
                linewidth=1,
                style="g-",
                alpha=0.7,
            ).render_linestring(axs["map"])
        )

    if (
        cfg.video.map_video.map_elements_to_plot is None
        or MapElements.AGENTS in cfg.video.map_video.map_elements_to_plot
    ):
        artists_on_map["agent_artists"] = sim_result.actor_polygons.render_at_time(
            axs["map"],
            timestamps_us[0],
            center=image_center_xy,
            max_dist=cfg.video.map_video.map_radius_m + 10,
        )
    else:
        artists_on_map["agent_artists"] = sim_result.actor_polygons.render_at_time(
            axs["map"],
            timestamps_us[0],
            only_agents=["EGO"],
        )

    if (
        cfg.video.map_video.map_elements_to_plot is None
        or MapElements.DRIVER_RESPONSES in cfg.video.map_video.map_elements_to_plot
    ):
        artists_on_map["driver_responses"] = sim_result.driver_responses.render_at_time(
            axs["map"], timestamps_us[0], "now"
        )

    if (
        cfg.video.map_video.map_elements_to_plot is None
        or MapElements.ROUTE in cfg.video.map_video.map_elements_to_plot
    ):
        artists_on_map["route"] = sim_result.routes.render_at_time(
            axs["map"],
            timestamps_us[0],
        )

    if (
        cfg.video.map_video.map_elements_to_plot is None
        or MapElements.EGO_GT_GHOST_POLYGON in cfg.video.map_video.map_elements_to_plot
    ):
        artists_on_map["ego_gt_ghost_polygon"] = (
            sim_result.ego_recorded_ground_truth_trajectory.set_polygon_plot_style(
                fill_color="limegreen",
            ).render_polygon_at_time(axs["map"], timestamps_us[0])
        )

    for artist in _list_in_dict_in_dict_to_list(artists_on_map):
        artist.set_transform(ego_transform + axs["map"].transData)

    def update(time: int) -> list[plt.Artist]:
        if should_render_table:
            update_table(table, processed_metrics_dfs, time)
        camera_artist = camera.render_image_at_time(time, axs["image"])

        ego_transform = get_ego_transform(
            sim_result=sim_result,
            cfg=cfg,
            time=time,
        )
        image_center_xy = sim_result.actor_polygons.set_axis_limits_around_agent(
            axs["map"],
            "EGO",
            time,
            cfg,
            axis_transform=ego_transform,
        )

        artists_on_map["map"] = shapely_map.render(
            axs["map"],
            cfg,
            center=image_center_xy,
            max_dist=cfg.video.map_video.map_radius_m + 10,
        )

        if (
            cfg.video.map_video.map_elements_to_plot is None
            or MapElements.DRIVER_RESPONSES in cfg.video.map_video.map_elements_to_plot
        ):
            artists_on_map["driver_responses"] = (
                sim_result.driver_responses.render_at_time(axs["map"], time, "now")
            )

        if (
            cfg.video.map_video.map_elements_to_plot is None
            or MapElements.ROUTE in cfg.video.map_video.map_elements_to_plot
        ):
            artists_on_map["route"] = sim_result.routes.render_at_time(
                axs["map"],
                time,
            )

        if (
            cfg.video.map_video.map_elements_to_plot is None
            or MapElements.EGO_GT_GHOST_POLYGON
            in cfg.video.map_video.map_elements_to_plot
        ):
            artists_on_map["ego_gt_ghost_polygon"] = (
                sim_result.ego_recorded_ground_truth_trajectory.render_polygon_at_time(
                    axs["map"], time
                )
            )

        if (
            cfg.video.map_video.map_elements_to_plot is None
            or MapElements.AGENTS in cfg.video.map_video.map_elements_to_plot
        ):
            artists_on_map["agent_artists"] = sim_result.actor_polygons.render_at_time(
                axs["map"],
                time,
                center=image_center_xy,
                max_dist=cfg.video.map_video.map_radius_m + 10,
            )
        else:
            artists_on_map["agent_artists"] = sim_result.actor_polygons.render_at_time(
                axs["map"],
                time,
                only_agents=["EGO"],
            )

        for artist in _list_in_dict_in_dict_to_list(artists_on_map):
            artist.set_transform(ego_transform + axs["map"].transData)

        text_artist.set_text(f"Time: {time}")

        # Update command text from driver response
        driver_response = sim_result.driver_responses.get_driver_response_for_time(
            time, which_time="now"
        )
        # The recording, pulled into the ego frame the BEV panel is drawn in.
        # Uses the ego pose the rollout actually reached, so once the model has
        # drifted the white line shows where the person was relative to where
        # the car now is -- which is the comparison that matters.
        gt_future_xy = _gt_future_in_ego_frame(
            sim_result.ego_recorded_ground_truth_trajectory,
            sim_result.actor_trajectories["EGO"],
            time,
        )
        drivesuprim_artists = _update_drivesuprim_bev(
            axs["drivesuprim_bev"],
            drivesuprim_bev_artists,
            driver_response,
            gt_future_xy=gt_future_xy,
        )
        drivesuprim_artists = drivesuprim_artists + _update_rank_panel(
            rank_artists, driver_response
        )
        command_name = (
            driver_response.command_name
            if driver_response and driver_response.command_name
            else None
        )
        if command_name:
            command_text_artist.set_text(f"Command: {command_name}")
            command_text_artist.set_visible(True)
        else:
            command_text_artist.set_visible(False)
        state_text = _driver_state_overlay_text(driver_response)
        state_text_artist.set_text(state_text)
        state_text_artist.set_visible(bool(state_text))

        overlay_artists: list[plt.Artist] = []
        if overlay_enabled and camera_projector is not None:
            overlay_artists = sim_result.driver_responses.render_on_camera(
                axs["image"],
                camera_projector,
                time,
                which_time="now",
            )
            overlay_artists.extend(
                sim_result.routes.render_on_camera(
                    axs["image"],
                    camera_projector,
                    time,
                )
            )

        all_artists = _list_in_dict_in_dict_to_list(artists_on_map)
        all_artists.append(camera_artist)
        all_artists.extend(overlay_artists)
        all_artists.extend(drivesuprim_artists)
        if should_render_table:
            all_artists.append(table)
        all_artists.append(text_artist)
        all_artists.append(command_text_artist)
        all_artists.append(state_text_artist)
        # Keep camera axis locked to image extent
        if camera_artist is not None:
            array = camera_artist.get_array()
            if array is not None:
                h, w = array.shape[:2]
                axs["image"].set_xlim(0, w)
                axs["image"].set_ylim(h, 0)
                axs["image"].set_autoscale_on(False)
        return all_artists

    timestamps_to_render_us = timestamps_us[:: cfg.video.render_every_nth_frame]
    interval_ms, fps = _compute_frame_timing(
        timestamps_us, cfg.video.render_every_nth_frame
    )

    frames_iterator = (
        tqdm(timestamps_to_render_us, desc="Rendering animation frames")
        if cfg.num_processes == 1
        else timestamps_to_render_us
    )

    # Create animation with progress bar
    anim_1 = animation.FuncAnimation(
        fig,
        update,
        frames=frames_iterator,
        interval=interval_ms,
        blit=True,
    )

    return anim_1, fps
