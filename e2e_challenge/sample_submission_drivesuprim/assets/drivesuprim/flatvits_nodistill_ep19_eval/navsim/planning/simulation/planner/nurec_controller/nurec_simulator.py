"""Proposal simulation with the NuRec controller and vehicle model.

Drop-in for ``PDMSimulator``. The difference is what closes the loop: navsim
tracks proposals with an LQR over a kinematic bicycle, while the NuRec Runtime
uses a linear MPC over a dynamic bicycle with tyre stiffness and actuator lag.
Scoring a vocabulary against the wrong controller answers "what would happen if
some other car tried this", so the labels are produced with the one that will
actually drive.

Integration follows the upstream pattern: the vehicle model's pose is reset each
step and ``advance`` returns the step's relative pose, so the controller always
sees its reference in the current rig frame and linearises about zero yaw. The
relative poses are composed back into the global track this class returns.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import numpy.typing as npt
import torch
from nuplan.common.actor_state.ego_state import EgoState
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.planning.simulation.planner.nurec_controller.batch_linear_mpc import (
    BatchLinearMPC,
    interpolate_reference,
    wrap_to_pi,
)
from navsim.planning.simulation.planner.nurec_controller.batch_vehicle_model import BatchVehicleModel
from navsim.planning.simulation.planner.nurec_controller.device import DTYPE, resolve_device
from navsim.planning.simulation.planner.nurec_controller.reference.mpc_controller import (
    DEFAULT_N_HORIZON,
    MPCGains,
)
from navsim.planning.simulation.planner.nurec_controller.reference.vehicle_model import VehicleModel
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_array_representation import (
    ego_state_to_state_array,
)
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_enums import StateIndex

_US = 1_000_000


class NuRecSimulator:
    """Simulate proposals through the NuRec MPC and dynamic bicycle model."""

    def __init__(
        self,
        proposal_sampling: TrajectorySampling,
        vehicle_params: Optional[VehicleModel.Parameters] = None,
        gains: Optional[MPCGains] = None,
        n_horizon: int = DEFAULT_N_HORIZON,
        device=None,
    ):
        self.proposal_sampling = proposal_sampling
        self._vehicle_params = vehicle_params or VehicleModel.Parameters()
        self._device = resolve_device(device)
        self._tracker = BatchLinearMPC(
            vehicle_params=self._vehicle_params,
            gains=gains,
            n_horizon=n_horizon,
            dt_mpc=proposal_sampling.interval_length,
            device=self._device,
        )

    def simulate_proposals(
        self, states: npt.NDArray[np.float64], initial_ego_state: EgoState
    ) -> npt.NDArray[np.float64]:
        """
        :param states: ``[B, num_poses + 1, StateIndex.size()]`` desired trajectories
        :param initial_ego_state: shared starting state
        :return: simulated states, same shape
        """
        initial = ego_state_to_state_array(initial_ego_state)
        rows = np.broadcast_to(initial, (states.shape[0], initial.shape[0]))
        return self.simulate_proposals_from_states(states, rows)

    def simulate_proposals_from_states(
        self, states: npt.NDArray[np.float64], initial_rows: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Same rollout, but every proposal may start from its own state.

        One token's 4096 proposals leave the A6000 less than half busy, and
        stacking several tokens into one call is worth ~1.4x. The rollout is
        already per-row everywhere except the start, which was pinned to a
        single ``EgoState``; taking a row per proposal is what lets a caller
        batch tokens together.

        :param states: ``[B, num_poses + 1, StateIndex.size()]`` desired trajectories
        :param initial_rows: ``[B, StateIndex.size()]`` starting state per proposal
        :return: simulated states, same shape as ``states[:, : num_poses + 1]``
        """
        num_poses = self.proposal_sampling.num_poses
        dt = self.proposal_sampling.interval_length
        proposals = states[:, : num_poses + 1]
        batch = proposals.shape[0]
        if initial_rows.shape[0] != batch:
            raise ValueError(
                f"initial_rows has {initial_rows.shape[0]} rows for {batch} proposals")

        # ``broadcast_to`` hands back a read-only, zero-strided view and torch
        # warns on every one of them; a real array costs nothing next to the
        # rollout and keeps 48k tokens' worth of warnings out of the log.
        initial = np.array(initial_rows, dtype=np.float64, copy=True, order="C")
        simulated = np.zeros_like(proposals)
        simulated[:, 0] = initial

        device = self._device
        velocity = torch.as_tensor(
            initial[:, [StateIndex.VELOCITY_X, StateIndex.VELOCITY_Y]],
            dtype=DTYPE, device=device)
        yaw_rate = torch.as_tensor(initial[:, StateIndex.ANGULAR_VELOCITY],
                                   dtype=DTYPE, device=device)
        model = BatchVehicleModel(batch, velocity, yaw_rate, self._vehicle_params, device=device)

        # Desired track in the global frame, and the times it is defined at.
        desired = torch.as_tensor(
            proposals[:, :, [StateIndex.X, StateIndex.Y, StateIndex.HEADING]],
            dtype=DTYPE, device=device)
        desired, desired_times = self._extend_past_the_end(desired, num_poses, dt)
        horizon_offsets = torch.arange(self._tracker._n_horizon + 1, device=device) * int(dt * _US)

        pose = torch.as_tensor(initial[:, [StateIndex.X, StateIndex.Y, StateIndex.HEADING]],
                               dtype=DTYPE, device=device).clone()

        for step in range(1, num_poses + 1):
            reference = interpolate_reference(
                self._to_rig_frame(desired, pose), desired_times,
                (step - 1) * int(dt * _US) + horizon_offsets,
            )

            # The controller sees itself at the origin of its own rig frame; only
            # the velocity and actuator states carry over.
            control_state = model.state.clone()
            control_state[:, :3] = 0.0
            command = self._tracker.compute_control(control_state, reference)

            model.reset_origin()
            relative_xy, relative_yaw = model.advance(command, dt)
            pose = self._compose(pose, relative_xy, relative_yaw)
            simulated[:, step] = self._pack(pose, model, command).cpu().numpy()

        return simulated

    def _extend_past_the_end(self, desired: torch.Tensor, num_poses: int, dt: float):
        """Continue each proposal past its final pose at its terminal velocity.

        The controller looks ``n_horizon`` steps ahead, so simulating a
        proposal for exactly its own length leaves the last second and a half of
        horizon with nothing to track. The reference implementation clamps to the
        final pose there, which is right in the Runtime -- a fresh plan always
        arrives before the horizon runs out -- but wrong here: clamping reads as
        "stop at the end", and every proposal would brake through its own tail,
        depressing ego_progress and comfort for all of them alike.

        Extending at the terminal velocity keeps the horizon fed with the motion
        the proposal was actually asking for.
        """
        horizon = self._tracker._n_horizon
        step = torch.arange(1, horizon + 1, dtype=desired.dtype,
                            device=desired.device)[None, :, None]
        velocity = (desired[:, -1:, :2] - desired[:, -2:-1, :2]) / dt

        tail = torch.empty((desired.shape[0], horizon, 3), dtype=desired.dtype,
                           device=desired.device)
        tail[:, :, :2] = desired[:, -1:, :2] + step * dt * velocity
        tail[:, :, 2] = desired[:, -1:, 2]

        extended = torch.cat([desired, tail], dim=1)
        times = torch.arange(num_poses + 1 + horizon, device=desired.device) * int(dt * _US)
        return extended, times

    @staticmethod
    def _to_rig_frame(track: torch.Tensor, pose: torch.Tensor) -> torch.Tensor:
        """Express a ``[B, T, 3]`` global track in each proposal's current frame."""
        cos_yaw, sin_yaw = torch.cos(pose[:, 2]), torch.sin(pose[:, 2])
        delta = track[:, :, :2] - pose[:, None, :2]
        out = torch.empty_like(track)
        out[:, :, 0] = delta[:, :, 0] * cos_yaw[:, None] + delta[:, :, 1] * sin_yaw[:, None]
        out[:, :, 1] = -delta[:, :, 0] * sin_yaw[:, None] + delta[:, :, 1] * cos_yaw[:, None]
        out[:, :, 2] = wrap_to_pi(track[:, :, 2] - pose[:, None, 2])
        return out

    @staticmethod
    def _compose(pose: torch.Tensor, relative_xy: torch.Tensor, relative_yaw: torch.Tensor) -> torch.Tensor:
        cos_yaw, sin_yaw = torch.cos(pose[:, 2]), torch.sin(pose[:, 2])
        out = torch.empty_like(pose)
        out[:, 0] = pose[:, 0] + relative_xy[:, 0] * cos_yaw - relative_xy[:, 1] * sin_yaw
        out[:, 1] = pose[:, 1] + relative_xy[:, 0] * sin_yaw + relative_xy[:, 1] * cos_yaw
        out[:, 2] = wrap_to_pi(pose[:, 2] + relative_yaw)
        return out

    def _pack(self, pose: torch.Tensor, model: BatchVehicleModel, command: torch.Tensor) -> torch.Tensor:
        state = model.state
        row = torch.zeros((pose.shape[0], StateIndex.size()),
                          dtype=pose.dtype, device=pose.device)
        row[:, StateIndex.X] = pose[:, 0]
        row[:, StateIndex.Y] = pose[:, 1]
        row[:, StateIndex.HEADING] = pose[:, 2]
        # The model carries velocity and acceleration at the centre of gravity;
        # navsim's state array is rear-axle (= rig) by convention, and the rig
        # sits ``l_rig_to_cg`` behind the CG. On a rotating body those are not
        # the same numbers, so convert exactly as the Runtime's
        # ``System._build_dynamic_state_in_rig_frame`` does:
        #   v_rig = v_cg + omega x r_cg_to_rig
        #   a_rig = a_cg + alpha x r_cg_to_rig + omega x (omega x r_cg_to_rig)
        # with r_cg_to_rig = [-l_rig_to_cg, 0, 0].
        lever = self._vehicle_params.l_rig_to_cg
        yaw_rate = state[:, 5]
        yaw_acceleration = model.accelerations[:, 2]
        row[:, StateIndex.VELOCITY_X] = state[:, 3]
        row[:, StateIndex.VELOCITY_Y] = state[:, 4] - yaw_rate * lever
        row[:, StateIndex.ACCELERATION_X] = (
            model.accelerations[:, 0] + yaw_rate * yaw_rate * lever
        )
        row[:, StateIndex.ACCELERATION_Y] = (
            model.accelerations[:, 1] - yaw_acceleration * lever
        )
        row[:, StateIndex.STEERING_ANGLE] = state[:, 6]
        # The model has no steering-rate state; it is the first-order lag that
        # drives the steering angle, which is what the comfort metrics read.
        row[:, StateIndex.STEERING_RATE] = (
            (command[:, 0] - state[:, 6]) / self._vehicle_params.steering_time_constant
        )
        row[:, StateIndex.ANGULAR_VELOCITY] = state[:, 5]
        row[:, StateIndex.ANGULAR_ACCELERATION] = model.accelerations[:, 2]
        return row
