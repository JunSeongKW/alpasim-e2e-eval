"""Batched QP solver for MPC, ADMM as OSQP formulates it.

Every proposal at every simulation step needs its own small QP. Calling OSQP
once per proposal costs milliseconds of Python and setup per solve, which does
not survive being multiplied by a vocabulary; the same ADMM iteration runs over
a batch axis here, sharing one factorisation across all proposals.

Two structural choices carry most of the speed:

* the constraint matrix an MPC produces is ``[I; S]`` -- plain bounds on the
  inputs stacked on the rows bounding predicted states. Keeping the two apart
  applies the identity block by slicing instead of by a multiply.
* items converge at wildly different rates (OSQP needs a median of ~75
  iterations on these problems and up to its cap on the hardest), so converged
  items leave the working set rather than the whole batch paying the worst case.

Solves, for each item ``i``::

    min 0.5 x' P_i x + q_i' x   s.t.  xl_i <= x <= xu_i,  sl_i <= S_i x <= su_i
"""

from __future__ import annotations

from typing import Tuple

import torch

# OSQP's defaults, so the batched solve converges to the same point.
SIGMA = 1e-6
RHO = 0.1
ALPHA = 1.6
SCALING_ITERATIONS = 10
# OSQP re-tunes rho when the two residuals drift this far apart.
RHO_ADAPT_RATIO = 5.0
# On hitting its iteration cap OSQP re-checks the residuals against a tolerance
# this much looser and, if they pass, returns the iterate as "solved
# inaccurate" -- which the reference controller accepts. Without the same
# second chance a batch here would discard answers the reference would use: on
# this vocabulary that was 500 of 4096 proposals, left coasting on a zero
# command.
#
# The value is OSQP's own, confirmed against it rather than taken on trust:
# running its solver on these QPs and measuring residual/tolerance gives
# 0.98 max for "solved", 7.37 max for "solved inaccurate", and 10.07 min for
# "maximum iterations reached" -- three bands that do not overlap, with the
# cut-off between the last two at ten.
INACCURATE_MULTIPLIER = 10.0


def _equilibrate(p, q, s, iterations=SCALING_ITERATIONS):
    """Ruiz equilibration, as OSQP applies before solving.

    An MPC cost matrix spans orders of magnitude between the position and input
    blocks, and ADMM converges at the rate the worst-scaled direction allows.
    Without this the solver runs to its iteration cap on every step.

    Variables are scaled by ``d`` and the state rows by ``e``; the input rows
    are left as plain bounds so the ``[I; S]`` structure survives.
    """
    batch, n = q.shape
    d = torch.ones_like(q)
    e = torch.ones(s.shape[:2], dtype=q.dtype, device=q.device)
    p_s, q_s, s_s = p.clone(), q.clone(), s.clone()

    for _ in range(iterations):
        # Column norms see the identity block too, hence the floor of one.
        col = torch.maximum(
            torch.maximum(p_s.abs().amax(dim=1), s_s.abs().amax(dim=1)),
            torch.ones_like(q_s),
        )
        delta = col.rsqrt()
        delta_row = s_s.abs().amax(dim=2).clamp(min=1e-12).rsqrt()

        p_s = p_s * delta[:, :, None] * delta[:, None, :]
        s_s = s_s * delta_row[:, :, None] * delta[:, None, :]
        q_s = q_s * delta
        d = d * delta
        e = e * delta_row

    # Cost scaling keeps the objective from dwarfing or vanishing against the
    # constraints.
    mean_col = p_s.abs().amax(dim=1).mean(dim=1)
    cost = 1.0 / torch.maximum(torch.maximum(mean_col, q_s.abs().amax(dim=1)),
                               torch.full_like(mean_col, 1e-12))
    return p_s * cost[:, None, None], q_s * cost[:, None], s_s, d, e, cost


def solve(
    p: torch.Tensor,
    q: torch.Tensor,
    s: torch.Tensor,
    input_lower: torch.Tensor,
    input_upper: torch.Tensor,
    state_lower: torch.Tensor,
    state_upper: torch.Tensor,
    eps_abs: float = 1e-4,
    eps_rel: float = 1e-4,
    max_iter: int = 500,
    rho: float = RHO,
    check_interval: int = 10,
) -> Tuple[torch.Tensor, int]:
    """
    :param p: ``[B, n, n]`` cost, positive semi-definite
    :param q: ``[B, n]`` linear cost
    :param s: ``[B, ms, n]`` state-constraint rows
    :param input_lower, input_upper: ``[B, n]`` bounds on the variables
    :param state_lower, state_upper: ``[B, ms]`` bounds on ``S x``
    :return: ``([B, n]`` solution, iterations run)
    """
    batch, n = q.shape
    device, dtype = q.device, q.dtype
    eye = torch.eye(n, dtype=dtype, device=device)

    p, q, s, d, e, cost = _equilibrate(p, q, s)
    # Bounds follow their variables and rows.
    input_lower, input_upper = input_lower / d, input_upper / d
    state_lower, state_upper = state_lower * e, state_upper * e
    st = s.transpose(1, 2)

    rho_vec = torch.full((batch, 1), rho, dtype=dtype, device=device)
    # A' A = I + S' S for A = [I; S].
    sts = st @ s
    kkt_inv = torch.linalg.inv(p + SIGMA * eye + rho_vec[..., None] * (eye + sts))

    x = torch.zeros((batch, n), dtype=dtype, device=device)
    z_in = torch.zeros_like(x)
    y_in = torch.zeros_like(x)
    z_st = torch.zeros(s.shape[:2], dtype=dtype, device=device)
    y_st = torch.zeros_like(z_st)

    active = torch.arange(batch, device=device)
    solution = torch.zeros((batch, n), dtype=dtype, device=device)
    iterations = 0

    for iteration in range(1, max_iter + 1):
        rhs = SIGMA * x - q + (rho_vec * z_in - y_in) + (st @ (rho_vec * z_st - y_st).unsqueeze(-1)).squeeze(-1)
        x_tilde = (kkt_inv @ rhs.unsqueeze(-1)).squeeze(-1)
        s_tilde = (s @ x_tilde.unsqueeze(-1)).squeeze(-1)

        x = ALPHA * x_tilde + (1.0 - ALPHA) * x
        relaxed_in = ALPHA * x_tilde + (1.0 - ALPHA) * z_in
        relaxed_st = ALPHA * s_tilde + (1.0 - ALPHA) * z_st

        next_in = torch.minimum(torch.maximum(relaxed_in + y_in / rho_vec, input_lower), input_upper)
        next_st = torch.minimum(torch.maximum(relaxed_st + y_st / rho_vec, state_lower), state_upper)
        y_in = y_in + rho_vec * (relaxed_in - next_in)
        y_st = y_st + rho_vec * (relaxed_st - next_st)
        z_in, z_st = next_in, next_st

        iterations = iteration
        # Residuals cost three more batched products, so they are not formed
        # every iteration.
        if iteration % check_interval and iteration != max_iter:
            continue
        sx = (s @ x.unsqueeze(-1)).squeeze(-1)
        px = (p @ x.unsqueeze(-1)).squeeze(-1)
        aty = y_in + (st @ y_st.unsqueeze(-1)).squeeze(-1)
        primal = torch.maximum((x - z_in).abs().amax(dim=1), (sx - z_st).abs().amax(dim=1))
        dual = (px + q + aty).abs().amax(dim=1)
        scale_primal = torch.maximum(
            torch.maximum(x.abs().amax(dim=1), sx.abs().amax(dim=1)),
            torch.maximum(z_in.abs().amax(dim=1), z_st.abs().amax(dim=1)))
        scale_dual = torch.maximum(
            torch.maximum(px.abs().amax(dim=1), q.abs().amax(dim=1)), aty.abs().amax(dim=1))

        done = ((primal <= eps_abs + eps_rel * scale_primal)
                & (dual <= eps_abs + eps_rel * scale_dual))
        if bool(done.any()):
            solution[active[done]] = x[done] * d[done]
            keep = ~done
            # active must shrink before the exit check, or the "everything left
            # failed" zeroing below would wipe the answers just written.
            active = active[keep]
            if active.numel() == 0:
                break
            (x, z_in, z_st, y_in, y_st, q, s, st, p, d, e, cost, kkt_inv, rho_vec,
             input_lower, input_upper, state_lower, state_upper, sts) = (
                x[keep], z_in[keep], z_st[keep], y_in[keep], y_st[keep],
                q[keep], s[keep], st[keep], p[keep], d[keep], e[keep], cost[keep],
                kkt_inv[keep], rho_vec[keep], input_lower[keep], input_upper[keep],
                state_lower[keep], state_upper[keep], sts[keep])
            primal, dual = primal[keep], dual[keep]
            scale_primal, scale_dual = scale_primal[keep], scale_dual[keep]

        # Re-tune rho when the residuals are pulling in different directions,
        # then refactorise. OSQP does the same; without it the iteration count
        # is set by whichever residual happens to lag.
        ratio = torch.sqrt((primal / scale_primal.clamp(min=1e-12))
                           / (dual / scale_dual.clamp(min=1e-12)).clamp(min=1e-12))
        retune = (ratio > RHO_ADAPT_RATIO) | (ratio < 1.0 / RHO_ADAPT_RATIO)
        if bool(retune.any()):
            rho_vec = torch.where(retune[:, None], (rho_vec * ratio[:, None]).clamp(1e-6, 1e6), rho_vec)
            kkt_inv = torch.linalg.inv(p + SIGMA * eye + rho_vec[..., None] * (eye + sts))

    # Whatever is left never met the tight tolerance. Give it OSQP's second
    # chance before discarding it: the reference treats "solved inaccurate" as a
    # usable answer, and on this vocabulary that is most of what reaches the cap.
    # Only what fails even the loose test gets the zero command the reference
    # returns for an unsolved QP.
    if active.numel():
        sx = (s @ x.unsqueeze(-1)).squeeze(-1)
        px = (p @ x.unsqueeze(-1)).squeeze(-1)
        aty = y_in + (st @ y_st.unsqueeze(-1)).squeeze(-1)
        primal = torch.maximum((x - z_in).abs().amax(dim=1), (sx - z_st).abs().amax(dim=1))
        dual = (px + q + aty).abs().amax(dim=1)
        scale_primal = torch.maximum(
            torch.maximum(x.abs().amax(dim=1), sx.abs().amax(dim=1)),
            torch.maximum(z_in.abs().amax(dim=1), z_st.abs().amax(dim=1)))
        scale_dual = torch.maximum(
            torch.maximum(px.abs().amax(dim=1), q.abs().amax(dim=1)), aty.abs().amax(dim=1))
        loose_abs = eps_abs * INACCURATE_MULTIPLIER
        loose_rel = eps_rel * INACCURATE_MULTIPLIER
        usable = ((primal <= loose_abs + loose_rel * scale_primal)
                  & (dual <= loose_abs + loose_rel * scale_dual)
                  & torch.isfinite(x).all(dim=1))
        solution[active[usable]] = x[usable] * d[usable]
        solution[active[~usable]] = 0.0
    return solution, iterations
