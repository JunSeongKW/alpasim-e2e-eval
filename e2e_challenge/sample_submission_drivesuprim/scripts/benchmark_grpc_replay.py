#!/usr/bin/env python3
"""Replay real driver inputs from an ASL and measure gRPC inference latency."""

from __future__ import annotations

import argparse
import statistics
import struct
import time
from pathlib import Path

import grpc
import numpy as np
from alpasim_grpc.v0 import egodriver_pb2_grpc
from alpasim_grpc.v0.logging_pb2 import LogEntry


def read_entries(path: Path):
    with path.open("rb") as stream:
        while prefix := stream.read(4):
            if len(prefix) != 4:
                raise IOError("truncated ASL size prefix")
            (size,) = struct.unpack(">L", prefix)
            payload = stream.read(size)
            if len(payload) != size:
                raise IOError(f"truncated ASL entry: expected {size}, got {len(payload)}")
            yield LogEntry.FromString(payload)


def summary(label: str, values_s: list[float]) -> None:
    values_ms = np.asarray(values_s, dtype=np.float64) * 1000.0
    print(
        f"{label}: n={len(values_ms)} mean={values_ms.mean():.3f}ms "
        f"p50={np.percentile(values_ms, 50):.3f}ms "
        f"p95={np.percentile(values_ms, 95):.3f}ms "
        f"p99={np.percentile(values_ms, 99):.3f}ms "
        f"min={values_ms.min():.3f}ms max={values_ms.max():.3f}ms "
        f"stdev={statistics.pstdev(values_ms):.3f}ms"
    )


def save_trajectory_plot(
    output: Path,
    ego_xy: list[tuple[float, float]],
    predictions_xy: list[np.ndarray],
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.95, len(predictions_xy)))
    for index, (trajectory, color) in enumerate(zip(predictions_xy, colors)):
        if trajectory.size:
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                color=color,
                linewidth=0.9,
                alpha=0.32,
                label="DriveSuprim predictions" if index == 0 else None,
            )
    if ego_xy:
        ego = np.asarray(ego_xy)
        ax.plot(
            ego[:, 0],
            ego[:, 1],
            color="black",
            linewidth=2.4,
            label="Recorded ego path",
            zorder=5,
        )
        ax.scatter(ego[0, 0], ego[0, 1], s=55, color="limegreen", label="Start", zorder=6)
        ax.scatter(ego[-1, 0], ego[-1, 1], s=55, color="red", label="End", zorder=6)
    ax.set_title("DriveSuprim PAI ASL Replay: Predicted vs Recorded Path")
    ax.set_xlabel("Local X (m)")
    ax.set_ylabel("Local Y (m)")
    ax.grid(True, alpha=0.25)
    ax.axis("equal")
    ax.legend(loc="best")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("asl", type=Path)
    parser.add_argument("--target", default="127.0.0.1:6793")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()

    rpc_latencies: list[float] = []
    cycle_latencies: list[float] = []
    drive_count = 0
    cycle_started_at: float | None = None
    ego_xy: list[tuple[float, float]] = []
    predictions_xy: list[np.ndarray] = []
    channel = grpc.insecure_channel(
        args.target,
        options=(
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
        ),
    )
    grpc.channel_ready_future(channel).result(timeout=args.timeout)
    stub = egodriver_pb2_grpc.EgodriverServiceStub(channel)

    for entry in read_entries(args.asl):
        kind = entry.WhichOneof("log_entry")
        if kind == "actor_poses":
            for actor in entry.actor_poses.actor_poses:
                if actor.actor_id == "EGO":
                    ego_xy.append((actor.actor_pose.vec.x, actor.actor_pose.vec.y))
                    break
        elif kind == "driver_session_request":
            stub.start_session(entry.driver_session_request, timeout=args.timeout)
        elif kind == "driver_camera_image":
            if cycle_started_at is None:
                cycle_started_at = time.perf_counter()
            stub.submit_image_observation(entry.driver_camera_image, timeout=args.timeout)
        elif kind == "driver_ego_trajectory":
            stub.submit_egomotion_observation(
                entry.driver_ego_trajectory, timeout=args.timeout
            )
        elif kind == "route_request":
            stub.submit_route(entry.route_request, timeout=args.timeout)
        elif kind == "ground_truth_request":
            stub.submit_recording_ground_truth(
                entry.ground_truth_request, timeout=args.timeout
            )
        elif kind == "driver_request":
            started_at = time.perf_counter()
            response = stub.drive(entry.driver_request, timeout=args.timeout)
            finished_at = time.perf_counter()
            if drive_count >= args.warmup:
                rpc_latencies.append(finished_at - started_at)
                predictions_xy.append(
                    np.asarray(
                        [
                            (pose.pose.vec.x, pose.pose.vec.y)
                            for pose in response.trajectory.poses
                        ],
                        dtype=np.float64,
                    ).reshape(-1, 2)
                )
                if cycle_started_at is not None:
                    cycle_latencies.append(finished_at - cycle_started_at)
            drive_count += 1
            cycle_started_at = None
            if len(rpc_latencies) >= args.iterations:
                break

    channel.close()
    if not rpc_latencies:
        raise RuntimeError("ASL ended before any measured Drive requests")
    print(
        f"target={args.target} warmup={args.warmup} measured={len(rpc_latencies)} "
        f"source={args.asl}"
    )
    summary("drive_rpc", rpc_latencies)
    if cycle_latencies:
        summary("camera_to_drive_response", cycle_latencies)
    if args.plot is not None:
        save_trajectory_plot(args.plot, ego_xy, predictions_xy)
        print(f"trajectory_plot={args.plot}")


if __name__ == "__main__":
    main()
