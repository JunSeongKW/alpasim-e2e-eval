"""Batched linearisation of the NuRec bicycle model.

Transcribes ``LinearMPC._linearize_dynamics`` over a leading batch axis. Each
proposal is linearised about its own state, so the low-speed switch is per
element rather than per call.
"""

from __future__ import annotations

from typing import Tuple

import torch

from navsim.planning.simulation.planner.nurec_controller.reference.vehicle_model import VehicleModel

IX, IY, IYAW, IVX, IVY, IYAW_RATE, ISTEERING, IACCEL = range(8)
NX, NU = 8, 2


def linearize(states: torch.Tensor, params: VehicleModel.Parameters, dt: float
              ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Discrete-time ``(A_d, B_d)`` for each state in ``[B, 8]``."""
    batch = states.shape[0]
    device, dtype = states.device, states.dtype

    yaw = states[:, IYAW]
    v_cg_x = states[:, IVX]
    v_cg_y = states[:, IVY]
    yaw_rate = states[:, IYAW_RATE]

    use_kinematic = v_cg_x < params.kinematic_threshold_speed
    cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)

    # The two branches populate overlapping but different entries, so each is
    # built whole and then selected -- filling one array in place would leave
    # the other branch's entries behind wherever the mask flips.
    a_kin = torch.zeros((batch, NX, NX), device=device, dtype=dtype)
    a_dyn = torch.zeros((batch, NX, NX), device=device, dtype=dtype)

    GAIN = 10.0
    l_r = params.l_rig_to_cg
    L = params.wheelbase
    a_kin[:, IX, IYAW] = -v_cg_x * sin_yaw
    a_kin[:, IX, IVX] = cos_yaw
    a_kin[:, IY, IYAW] = v_cg_x * cos_yaw
    a_kin[:, IY, IVX] = sin_yaw
    a_kin[:, IYAW, IYAW_RATE] = 1.0
    a_kin[:, IVX, IACCEL] = 1.0
    a_kin[:, IVY, IVY] = -GAIN
    a_kin[:, IVY, ISTEERING] = GAIN * v_cg_x * l_r / L
    a_kin[:, IYAW_RATE, ISTEERING] = GAIN * v_cg_x / L
    a_kin[:, IYAW_RATE, IYAW_RATE] = -GAIN

    # Coefficients divide by v_cg_x, which is what the kinematic branch exists
    # to avoid; clamp so the discarded elements stay finite.
    safe_v_x = torch.clamp(v_cg_x, min=params.kinematic_threshold_speed)
    kinetic_mass = params.mass * safe_v_x
    kinetic_inertia = params.inertia * safe_v_x
    lf = params.wheelbase - params.l_rig_to_cg
    lr = params.l_rig_to_cg
    caf, car = params.front_cornering_stiffness, params.rear_cornering_stiffness
    lf_caf, lr_car = lf * caf, lr * car

    a_00 = -2 * (caf + car) / kinetic_mass
    a_01 = -v_cg_x - 2 * (lf_caf - lr_car) / kinetic_mass
    a_10 = -2 * (lf_caf - lr_car) / kinetic_inertia
    a_11 = -2 * (lf * lf_caf + lr * lr_car) / kinetic_inertia
    b_00 = 2 * caf / params.mass
    b_10 = 2 * lf_caf / params.inertia
    v_rig_y = v_cg_y - lr * yaw_rate

    a_dyn[:, IX, IYAW] = -v_cg_x * sin_yaw - v_rig_y * cos_yaw
    a_dyn[:, IX, IVX] = cos_yaw
    a_dyn[:, IX, IVY] = -sin_yaw
    a_dyn[:, IX, IYAW_RATE] = lr * sin_yaw
    a_dyn[:, IY, IYAW] = v_cg_x * cos_yaw - v_rig_y * sin_yaw
    a_dyn[:, IY, IVX] = sin_yaw
    a_dyn[:, IY, IVY] = cos_yaw
    a_dyn[:, IY, IYAW_RATE] = -lr * cos_yaw
    a_dyn[:, IYAW, IYAW_RATE] = 1.0
    a_dyn[:, IVX, IACCEL] = 1.0
    a_dyn[:, IVY, IVY] = a_00
    a_dyn[:, IVY, IYAW_RATE] = a_01
    a_dyn[:, IVY, ISTEERING] = b_00
    a_dyn[:, IYAW_RATE, IVY] = a_10
    a_dyn[:, IYAW_RATE, IYAW_RATE] = a_11
    a_dyn[:, IYAW_RATE, ISTEERING] = b_10

    a = torch.where(use_kinematic[:, None, None], a_kin, a_dyn)

    # Actuator lags are shared by both branches.
    tau_s = params.steering_time_constant
    tau_a = params.acceleration_time_constant
    a[:, ISTEERING, ISTEERING] = -1.0 / tau_s
    a[:, IACCEL, IACCEL] = -1.0 / tau_a

    b = torch.zeros((batch, NX, NU), device=device, dtype=dtype)
    b[:, ISTEERING, 0] = 1.0 / tau_s
    b[:, IACCEL, 1] = 1.0 / tau_a

    block = torch.zeros((batch, NX + NU, NX + NU), device=device, dtype=dtype)
    block[:, :NX, :NX] = a * dt
    block[:, :NX, NX:] = b * dt
    # torch's matrix exponential is batched and already uses scaling-and-squaring
    # Pade internally, so there is nothing to hand-roll here.
    exponential = torch.linalg.matrix_exp(block)
    return exponential[:, :NX, :NX], exponential[:, :NX, NX:]
