"""Batched LinearMPC, matching the NuRec reference controller.

Same cost, same constraints, same condensed QP as ``reference/linear_mpc.py`` --
expressed over a batch of proposals so a whole vocabulary is tracked in one pass
instead of one Python call per candidate.
"""

from __future__ import annotations

from typing import Optional

import torch

from navsim.planning.simulation.planner.nurec_controller import batch_qp
from navsim.planning.simulation.planner.nurec_controller.batch_linearize import linearize
from navsim.planning.simulation.planner.nurec_controller.device import DTYPE, resolve_device
from navsim.planning.simulation.planner.nurec_controller.reference.mpc_controller import (
    DEFAULT_DT_MPC,
    DEFAULT_N_HORIZON,
    MPCGains,
)
from navsim.planning.simulation.planner.nurec_controller.reference.vehicle_model import VehicleModel

IX, IY, IYAW, IVX, IVY, IYAW_RATE, ISTEERING, IACCEL = range(8)
NX, NU = 8, 2


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def interpolate_reference(
    trajectory: torch.Tensor,
    trajectory_times_us: torch.Tensor,
    target_times_us: torch.Tensor,
) -> torch.Tensor:
    """Sample ``[B, T, 3]`` (x, y, yaw) trajectories at ``[H]`` timestamps.

    Reproduces ``utils_rs``: the time range is half-open ``[first, last + 1)``,
    targets are clamped into it rather than extrapolated, position interpolates
    linearly and orientation by shortest arc -- which is what quaternion slerp
    reduces to for planar poses.
    """
    times = trajectory_times_us
    targets = torch.clamp(target_times_us, times[0], times[-1])

    idx = torch.clamp(torch.searchsorted(times, targets, right=True) - 1, 0, times.numel() - 2)
    t0, t1 = times[idx], times[idx + 1]
    span = (t1 - t0).clamp(min=1).to(trajectory.dtype)
    alpha = torch.where(t1 > t0, (targets - t0).to(trajectory.dtype) / span,
                        torch.zeros_like(span))

    p0, p1 = trajectory[:, idx, :], trajectory[:, idx + 1, :]
    out = torch.empty_like(p0)
    out[..., :2] = p0[..., :2] + alpha[None, :, None] * (p1[..., :2] - p0[..., :2])
    out[..., 2] = p0[..., 2] + alpha[None, :] * wrap_to_pi(p1[..., 2] - p0[..., 2])
    return out


class BatchLinearMPC:
    """``LinearMPC`` over a batch of proposals."""

    OSQP_EPS_ABS = 1e-4
    OSQP_EPS_REL = 1e-4
    OSQP_MAX_ITER = 500

    def __init__(
        self,
        vehicle_params: Optional[VehicleModel.Parameters] = None,
        gains: Optional[MPCGains] = None,
        n_horizon: int = DEFAULT_N_HORIZON,
        dt_mpc: float = DEFAULT_DT_MPC,
        device=None,
    ):
        self._vehicle_params = vehicle_params or VehicleModel.Parameters()
        self._gains = gains or MPCGains()
        self._n_horizon = n_horizon
        self._dt_mpc = dt_mpc
        self._device = resolve_device(device)
        self._iterations = 0

        def diag(values):
            return torch.diag(torch.tensor(values, dtype=DTYPE, device=self._device))

        self._q = diag([
            self._gains.long_position_weight,
            self._gains.lat_position_weight,
            self._gains.heading_weight,
            0.0, 0.0, 0.0, 0.0,
            self._gains.acceleration_weight,
        ])
        self._r_blk = torch.block_diag(*[
            diag([self._gains.rel_front_steering_angle_weight,
                  self._gains.rel_acceleration_weight])
            for _ in range(n_horizon)
        ])

        self._constrained_state_indices = [IYAW, IVX, ISTEERING, IACCEL]
        self._x_min_constrained = torch.tensor(
            [-torch.pi / 2, 0.0, -torch.pi / 4, -8.0], dtype=DTYPE, device=self._device)
        self._x_max_constrained = torch.tensor(
            [torch.pi / 2, 35.0, torch.pi / 4, 6.0], dtype=DTYPE, device=self._device)
        self._u_min = torch.tensor([-2.0, -9.0], dtype=DTYPE, device=self._device)
        self._u_max = torch.tensor([2.0, 6.0], dtype=DTYPE, device=self._device)

        # Which rows of the stacked prediction carry a constrained state.
        self._constrained_rows = torch.tensor(
            [k * NX + idx for k in range(1, n_horizon + 1)
             for idx in self._constrained_state_indices],
            dtype=torch.long, device=self._device)

        # Tracking cost only from idx_start_penalty on; the reference's terminal
        # cost is a copy of the stage cost.
        blocks = [
            self._q if k >= self._gains.idx_start_penalty
            else torch.zeros((NX, NX), dtype=DTYPE, device=self._device)
            for k in range(n_horizon + 1)
        ]
        self._q_blk = torch.block_diag(*blocks)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dt_mpc(self) -> float:
        return self._dt_mpc

    def compute_control(self, states: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
        """
        :param states: ``[B, 8]`` current vehicle states
        :param references: ``[B, n_horizon + 1, 3]`` reference (x, y, yaw)
        :return: ``[B, 2]`` commands ``[steering_cmd, accel_cmd]``
        """
        states = torch.as_tensor(states, dtype=DTYPE, device=self._device)
        references = torch.as_tensor(references, dtype=DTYPE, device=self._device)
        batch, n = states.shape[0], self._n_horizon

        x_ref = torch.zeros((batch, n + 1, NX), dtype=DTYPE, device=self._device)
        x_ref[:, :, IX] = references[:, :, 0]
        x_ref[:, :, IY] = references[:, :, 1]
        x_ref[:, :, IYAW] = references[:, :, 2]

        a_d, b_d = linearize(states, self._vehicle_params, self._dt_mpc)

        # Condensed prediction: x_k = S_x[k] x0 + S_u[k] U.
        s_x = torch.zeros((batch, (n + 1) * NX, NX), dtype=DTYPE, device=self._device)
        s_u = torch.zeros((batch, (n + 1) * NX, n * NU), dtype=DTYPE, device=self._device)
        a_pow = torch.eye(NX, dtype=DTYPE, device=self._device).expand(batch, NX, NX).clone()
        for k in range(n + 1):
            s_x[:, k * NX:(k + 1) * NX, :] = a_pow
            if k < n:
                a_pow = a_d @ a_pow
        for j in range(n):
            psi = b_d
            for k in range(j, n):
                s_u[:, (k + 1) * NX:(k + 2) * NX, j * NU:(j + 1) * NU] = psi
                psi = a_d @ psi

        x_pred_free = (s_x @ states.unsqueeze(-1)).squeeze(-1)
        error = x_pred_free - x_ref.reshape(batch, -1)

        s_u_t = s_u.transpose(1, 2)
        h = s_u_t @ (self._q_blk @ s_u) + self._r_blk
        h = 0.5 * (h + h.transpose(1, 2))
        g = (s_u_t @ (self._q_blk @ error.unsqueeze(-1))).squeeze(-1)

        # Input bounds are plain variable bounds; only the predicted-state rows
        # need a matrix, so the two are handed to the solver separately.
        free_states = x_pred_free[:, self._constrained_rows]
        solution, self._iterations = batch_qp.solve(
            h, g, s_u[:, self._constrained_rows, :],
            self._u_min.repeat(n).expand(batch, n * NU),
            self._u_max.repeat(n).expand(batch, n * NU),
            self._x_min_constrained.repeat(n) - free_states,
            self._x_max_constrained.repeat(n) - free_states,
            eps_abs=self.OSQP_EPS_ABS, eps_rel=self.OSQP_EPS_REL, max_iter=self.OSQP_MAX_ITER,
        )
        return solution[:, :NU]

    def reset_warm_start(self) -> None:
        """Kept for interface parity; the compacted solver carries no iterates."""
