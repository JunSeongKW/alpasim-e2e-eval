"""Batched planar dynamic bicycle model, matching the NuRec ``VehicleModel``.

Scoring a vocabulary means propagating thousands of proposals through the same
dynamics, so the upstream per-instance model is re-expressed over a leading
batch axis. The maths, the parameters, the RK2 sub-stepping and the low-speed
switch are the reference's; only the shape and the array library change.

State is ``[B, 8]``: ``x, y, yaw, vx_cg, vy_cg, yaw_rate, steering, accel``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from navsim.planning.simulation.planner.nurec_controller.device import DTYPE, resolve_device
from navsim.planning.simulation.planner.nurec_controller.reference.vehicle_model import VehicleModel

# The reference integrates in sub-steps of at most this length.
DT_STEP_MAX = 0.01

IX, IY, IYAW, IVX, IVY, IYAW_RATE, ISTEERING, IACCEL = range(8)
NX = 8


class BatchVehicleModel:
    """``VehicleModel`` over a batch of proposals."""

    def __init__(self, batch_size: int, initial_velocity, initial_yaw_rate,
                 parameters: Optional[VehicleModel.Parameters] = None, device=None):
        """
        :param initial_velocity: ``[B, 2]`` rig-frame ``[vx, vy]``
        :param initial_yaw_rate: ``[B]`` rig-frame yaw rate
        """
        self._parameters = parameters or VehicleModel.Parameters()
        self._device = resolve_device(device)
        velocity = torch.as_tensor(initial_velocity, dtype=DTYPE, device=self._device).reshape(batch_size, 2)
        yaw_rate = torch.as_tensor(initial_yaw_rate, dtype=DTYPE, device=self._device).reshape(batch_size)

        # Same kinematic seeding as the reference, including its 0.25 m/s gate.
        moving = velocity[:, 0] > 0.25
        safe_vx = torch.where(moving, velocity[:, 0], torch.ones_like(velocity[:, 0]))
        steering = torch.where(
            moving,
            torch.atan(yaw_rate / safe_vx * self._parameters.wheelbase),
            torch.zeros_like(yaw_rate),
        )

        self._state = torch.zeros((batch_size, NX), dtype=DTYPE, device=self._device)
        self._state[:, IVX] = velocity[:, 0]
        self._state[:, IVY] = velocity[:, 1]
        self._state[:, IYAW_RATE] = yaw_rate
        self._state[:, ISTEERING] = steering
        self._accelerations = torch.zeros((batch_size, 3), dtype=DTYPE, device=self._device)

    @property
    def parameters(self) -> VehicleModel.Parameters:
        return self._parameters

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def state(self) -> torch.Tensor:
        return self._state

    @property
    def accelerations(self) -> torch.Tensor:
        return self._accelerations

    def reset_origin(self) -> None:
        self._state[:, :3] = 0.0

    def set_velocity(self, v_cg_x, v_cg_y) -> None:
        self._state[:, IVX] = torch.as_tensor(v_cg_x, dtype=DTYPE, device=self._device)
        self._state[:, IVY] = torch.as_tensor(v_cg_y, dtype=DTYPE, device=self._device)

    def advance(self, u: torch.Tensor, dt: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Advance every proposal by ``dt``.

        :param u: ``[B, 2]`` commands ``[steering_cmd, accel_cmd]``
        :return: ``([B, 2]`` planar position, ``[B]`` yaw) after the step
        """
        u = torch.as_tensor(u, dtype=DTYPE, device=self._device).reshape(self._state.shape[0], 2)

        # Reproduce the reference's float accumulation, not a computed step
        # count: the two differ on the final partial sub-step.
        total_time = 0.0
        while total_time < dt:
            step_dt = min(DT_STEP_MAX, dt - total_time)
            total_time += step_dt

            k1 = step_dt * self._derivs(self._state, u)
            k2 = step_dt * self._derivs(self._state + k1 / 2.0, u)
            self._state = self._state + k2
            self._state[:, IVX] = torch.clamp(self._state[:, IVX], min=0.0)

        final_derivs = self._derivs(self._state, u)
        self._accelerations = final_derivs[:, [IVX, IVY, IYAW_RATE]].clone()
        return self._state[:, :2].clone(), self._state[:, IYAW].clone()

    def _derivs(self, state: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        params = self._parameters
        yaw = state[:, IYAW]
        v_x = state[:, IVX]
        v_y = state[:, IVY]
        yaw_rate = state[:, IYAW_RATE]
        steering = state[:, ISTEERING]
        longitudinal_acceleration = state[:, IACCEL]

        use_kinematic = v_x < params.kinematic_threshold_speed

        # --- kinematic branch: pull toward no-slip ---
        GAIN = 10.0
        steady_state_v_y = v_x * steering * params.l_rig_to_cg / params.wheelbase
        steady_state_yaw_rate = v_x * steering / params.wheelbase
        kin_d_v_y = GAIN * (steady_state_v_y - v_y)
        kin_d_yaw_rate = GAIN * (steady_state_yaw_rate - yaw_rate)

        # --- dynamic branch ---
        # Both branches are evaluated for the whole batch, so the dynamic one
        # must stay finite even for the elements the mask will throw away: its
        # coefficients divide by v_x, which is exactly what the mask excludes.
        safe_v_x = torch.clamp(v_x, min=params.kinematic_threshold_speed)
        kinetic_mass = params.mass * safe_v_x
        kinetic_inertia = params.inertia * safe_v_x

        lf = params.wheelbase - params.l_rig_to_cg
        lr = params.l_rig_to_cg
        lf_caf = lf * params.front_cornering_stiffness
        lr_car = lr * params.rear_cornering_stiffness

        a_00 = -2 * (params.front_cornering_stiffness + params.rear_cornering_stiffness) / kinetic_mass
        a_01 = -v_x - 2 * (lf_caf - lr_car) / kinetic_mass
        a_10 = -2 * (lf_caf - lr_car) / kinetic_inertia
        a_11 = -2 * (lf * lf_caf + lr * lr_car) / kinetic_inertia
        b_00 = 2 * params.front_cornering_stiffness / params.mass
        b_10 = 2 * lf_caf / params.inertia

        dyn_d_v_y = a_00 * v_y + a_01 * yaw_rate + b_00 * steering
        dyn_d_yaw_rate = a_10 * v_y + a_11 * yaw_rate + b_10 * steering

        d_v_y = torch.where(use_kinematic, kin_d_v_y, dyn_d_v_y)
        d_yaw_rate = torch.where(use_kinematic, kin_d_yaw_rate, dyn_d_yaw_rate)
        # The rig-frame lateral velocity that drives x/y differs per branch:
        # the kinematic branch asserts no slip at the rear.
        v_y_rig = torch.where(use_kinematic, torch.zeros_like(v_y), v_y - yaw_rate * lr)

        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        return torch.stack([
            v_x * cos_yaw - v_y_rig * sin_yaw,
            v_x * sin_yaw + v_y_rig * cos_yaw,
            yaw_rate,
            longitudinal_acceleration,
            d_v_y,
            d_yaw_rate,
            (u[:, 0] - steering) / params.steering_time_constant,
            (u[:, 1] - longitudinal_acceleration) / params.acceleration_time_constant,
        ], dim=1)
