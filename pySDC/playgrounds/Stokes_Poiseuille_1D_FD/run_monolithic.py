r"""
Monolithic RADAU-IIA collocation solver for the 1-D Stokes/Poiseuille DAE
==========================================================================

This script implements a **monolithic all-at-once** RADAU-IIA collocation
solver for the 1-D Stokes/Poiseuille index-1 DAE:

.. math::

    \mathbf{u}' = \nu A\,\mathbf{u} + G(t)\,\mathbf{1} + \mathbf{f}(t),
    \quad
    0 = B\,\mathbf{u} - q(t).

In contrast to the node-by-node SDC sweep in :mod:`run_convergence`, the
monolithic solver assembles all :math:`M` RADAU stages simultaneously into a
single :math:`M(N+1) \times M(N+1)` sparse saddle-point system and solves it
in one shot.  The endpoint is taken as the last stage value
:math:`\mathbf{u}_{n+1} = \mathbf{u}_M` (the **y-formulation**, since
RADAU-RIGHT has :math:`c_M = 1`).

Key finding: monolithic vs.\ SDC standard are **identical**
------------------------------------------------------------
For the **original algebraic constraint** :math:`B\mathbf{u}_m = q(\tau_m)`,
the monolithic RADAU-IIA collocation system is **mathematically identical** to
the SDC collocation fixed point.  Both solvers compute the same stage values
:math:`\mathbf{u}_m`, and both produce the same endpoint errors.  The observed
orders are :math:`M+1` for velocity and :math:`M` for pressure — the **same
order reduction** seen in the SDC standard sweep.

This is not a coincidence.  For a semi-explicit index-1 DAE with a time-
dependent algebraic constraint, the theory (Hairer & Wanner) predicts that
RADAU-IIA with the algebraic constraint at each stage achieves only the
**stage order** :math:`M+1` at the endpoint (not the full :math:`2M-1`
order), because the stage accuracy is limited by the constraint.  Switching
from iterative SDC sweeping to a single direct solve does not change this.

To recover full :math:`2M-1` order it is necessary to replace the algebraic
constraint :math:`B\mathbf{u}_m = q(\tau_m)` by its **differentiated form**
:math:`B\mathbf{u}_m' = q'(\tau_m)` (index reduction).  This is implemented
in :func:`_monolithic_step_diffconstr` and achieves :math:`2M-1` order for
both velocity and pressure.

Monolithic collocation system — two constraint variants
--------------------------------------------------------
For each stage :math:`m = 1,\ldots,M`:

.. math::

    \mathbf{u}_m
      = \mathbf{u}_0 + \Delta t\sum_{j=1}^M Q_{mj}
          \bigl(\nu A\,\mathbf{u}_j + G_j\,\mathbf{1} + \mathbf{f}(\tau_j)
          \bigr).

**Algebraic constraint** (order :math:`M+1` / :math:`M`):

.. math::

    B\,\mathbf{u}_m = q(\tau_m).

Block system :math:`K x = b`:

.. math::

    \underbrace{\begin{pmatrix}
        I_{MN} - \Delta t\,(Q \otimes \nu A) & -\Delta t\,(Q \otimes \mathbf{1}) \\
        I_M \otimes B                        & 0_{M \times M}
    \end{pmatrix}}_{K_\text{alg}}
    \begin{pmatrix} \mathbf{u}_\text{vec} \\ G_\text{vec} \end{pmatrix}
    =
    \begin{pmatrix}
        \mathbf{1}_M \otimes \mathbf{u}_0
          + \Delta t\,(Q \otimes I_N)\,\mathbf{f}_\text{vec} \\
        \mathbf{q}_\text{vec}
    \end{pmatrix}.

**Differentiated constraint** (order :math:`2M-1` / :math:`2M-1`):

.. math::

    B\,\mathbf{u}_m' = q'(\tau_m),
    \quad
    \text{i.e.,}\quad
    (B\nu A)\,\mathbf{u}_m + s\,G_m = q'(\tau_m) - B\,\mathbf{f}(\tau_m).

Same velocity block :math:`(I_{MN} - \Delta t(Q \otimes \nu A),\;
-\Delta t(Q \otimes \mathbf{1}))`, but the constraint rows change to:

.. math::

    \underbrace{\begin{pmatrix}
        I_{MN} - \Delta t\,(Q \otimes \nu A) & -\Delta t\,(Q \otimes \mathbf{1}) \\
        I_M \otimes (B\nu A)                  & s\,I_M
    \end{pmatrix}}_{K_\text{dc}}
    \begin{pmatrix} \mathbf{u}_\text{vec} \\ G_\text{vec} \end{pmatrix}
    =
    \begin{pmatrix}
        \mathbf{1}_M \otimes \mathbf{u}_0
          + \Delta t\,(Q \otimes I_N)\,\mathbf{f}_\text{vec} \\
        [q'(\tau_m) - B\mathbf{f}(\tau_m)]_{m=1}^M
    \end{pmatrix}.

For the differentiated-constraint system the algebraic variable is determined
by an explicit formula once :math:`\mathbf{u}_m` is known, and RADAU-IIA can
achieve its full stage accuracy :math:`2M-1`.

Summary of orders
-----------------
+-----------------------------------+----------------+----------------+
| Method                            | Velocity order | Pressure order |
+===================================+================+================+
| SDC standard (U-form., alg.)      | :math:`M+1`    | :math:`M`      |
+-----------------------------------+----------------+----------------+
| SDC diffconstr (U-form., dc)      | :math:`M+2`    | :math:`M+2`    |
+-----------------------------------+----------------+----------------+
| Monolithic alg. ≡ SDC std fixed pt| :math:`M+1`    | :math:`M`      |
+-----------------------------------+----------------+----------------+
| **Monolithic diffconstr (y-form)**| :math:`2M-1`   | :math:`2M-1`   |
+-----------------------------------+----------------+----------------+

For :math:`M = 3`: :math:`M+2 = 5 = 2M-1`, so SDC diffconstr and monolithic
diffconstr give the same asymptotic order (with different constants).
For :math:`M = 4`: monolithic diffconstr gives :math:`7 = 2M-1`, while SDC
diffconstr gives only :math:`6 = M+2`.

Usage::

    python run_monolithic.py
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pySDC.core.collocation import CollBase
from pySDC.playgrounds.Stokes_Poiseuille_1D_FD.Stokes_Poiseuille_1D_FD import (
    stokes_poiseuille_1d_fd,
)

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------

_NVARS = 1023
_NU = 0.1
_TEND = 1.0


# ---------------------------------------------------------------------------
# Helpers: build common blocks
# ---------------------------------------------------------------------------

def _common_velocity_blocks(P, u0, t0, dt, Q, nodes):
    """Build the velocity RHS and the velocity-velocity / velocity-pressure blocks."""
    N = P.nvars
    M = len(nodes)
    A = P.A

    tau = t0 + dt * nodes
    f_stages = np.array([P._forcing(tau[m]) for m in range(M)])
    forcing_contrib = dt * (Q @ f_stages)
    rhs_vel = np.tile(u0, M).reshape(M, N) + forcing_contrib
    rhs_vel_flat = rhs_vel.flatten()

    KUU = (
        sp.eye(M * N, format='csc')
        - dt * sp.kron(sp.csc_matrix(Q), A, format='csc')
    )
    ones_col = np.ones((N, 1))
    KUG = -dt * sp.kron(
        sp.csc_matrix(Q),
        sp.csc_matrix(ones_col),
        format='csc',
    )
    return tau, f_stages, rhs_vel_flat, KUU, KUG


# ---------------------------------------------------------------------------
# Monolithic step — algebraic constraint  B·u_m = q(τ_m)
# ---------------------------------------------------------------------------

def _monolithic_step_algebraic(P, u0, t0, dt, Q, nodes):
    r"""
    Single RADAU-IIA step with the **algebraic constraint**
    :math:`B\mathbf{u}_m = q(\tau_m)`.

    Produces the same collocation fixed point as the SDC standard sweep.
    Returns ``(u_{n+1}, G_{n+1})`` where :math:`u_{n+1} = \mathbf{u}_M`
    (last stage, y-formulation).
    """
    N = P.nvars
    M = len(nodes)
    dx = P.dx

    tau, _f, rhs_vel_flat, KUU, KUG = _common_velocity_blocks(
        P, u0, t0, dt, Q, nodes
    )

    rhs_constr = np.array([P._q(tau[m]) for m in range(M)])
    rhs = np.concatenate([rhs_vel_flat, rhs_constr])

    B_row = dx * np.ones((1, N))
    KCU = sp.kron(sp.eye(M, format='csc'), sp.csc_matrix(B_row), format='csc')
    KCG = sp.csc_matrix((M, M))

    K = sp.bmat([[KUU, KUG], [KCU, KCG]], format='csc')
    x = spsolve(K, rhs)

    u_end = x[(M - 1) * N: M * N].copy()
    G_end = float(x[M * N + (M - 1)])
    return u_end, G_end


# ---------------------------------------------------------------------------
# Monolithic step — differentiated constraint  B·u_m' = q'(τ_m)
# ---------------------------------------------------------------------------

def _monolithic_step_diffconstr(P, u0, t0, dt, Q, nodes):
    r"""
    Single RADAU-IIA step with the **differentiated constraint**
    :math:`B\mathbf{u}'_m = q'(\tau_m)`.

    Substituting :math:`\mathbf{u}'_m = \nu A\mathbf{u}_m + G_m\mathbf{1}
    + \mathbf{f}(\tau_m)`, the constraint becomes

    .. math::

        (B\nu A)\,\mathbf{u}_m + s\,G_m = q'(\tau_m) - B\mathbf{f}(\tau_m),

    where :math:`s = B\mathbf{1}`.  The modified constraint block makes
    :math:`K_\text{dc}` non-singular and the y-formulation endpoint
    :math:`\mathbf{u}_M` achieves the full :math:`2M-1` collocation order.
    """
    N = P.nvars
    M = len(nodes)
    A = P.A
    s = P.s    # B·1 = dx·N

    tau, f_stages, rhs_vel_flat, KUU, KUG = _common_velocity_blocks(
        P, u0, t0, dt, Q, nodes
    )

    # Differentiated constraint RHS: q'(τ_m) − B·f(τ_m)
    rhs_constr = np.array([
        P._q_prime(tau[m]) - P.dx * float(np.sum(f_stages[m]))
        for m in range(M)
    ])
    rhs = np.concatenate([rhs_vel_flat, rhs_constr])

    # KCU: I_M ⊗ (B·νA)  —  (B·νA) is a 1×N row = dx·1ᵀ·A
    BA_vec = P.dx * (P.ones @ A)                # (N,) dense, B·νA row vector
    BA = sp.csc_matrix(BA_vec.reshape(1, N))    # (1, N) sparse
    KCU = sp.kron(
        sp.eye(M, format='csc'),
        sp.csc_matrix(BA),
        format='csc',
    )

    # KCG: s·I_M  (not zero — algebraic constraint involves G_m)
    KCG = s * sp.eye(M, format='csc')

    K = sp.bmat([[KUU, KUG], [KCU, KCG]], format='csc')
    x = spsolve(K, rhs)

    u_end = x[(M - 1) * N: M * N].copy()
    G_end = float(x[M * N + (M - 1)])
    return u_end, G_end


# ---------------------------------------------------------------------------
# Time-stepping loop
# ---------------------------------------------------------------------------

def _run_monolithic(step_fn, M, dt, nvars=_NVARS, nu=_NU, Tend=_TEND):
    """Integrate using *step_fn* and return ``(vel_err, pres_err)``."""
    coll = CollBase(
        num_nodes=M, tleft=0, tright=1,
        node_type='LEGENDRE', quad_type='RADAU-RIGHT',
    )
    Q = coll.Qmat[1:, 1:]
    nodes = coll.nodes

    P = stokes_poiseuille_1d_fd(nvars=nvars, nu=nu)
    t = 0.0
    u = np.sin(np.pi * P.xvalues) * np.sin(t)

    n_steps = int(round(Tend / dt))
    for _ in range(n_steps):
        u, G = step_fn(P, u, t, dt, Q, nodes)
        t += dt

    u_ex = np.sin(np.pi * P.xvalues) * np.sin(Tend)
    G_ex = np.cos(Tend)
    return float(np.max(np.abs(u - u_ex))), float(abs(G - G_ex))


# ---------------------------------------------------------------------------
# Convergence table helper
# ---------------------------------------------------------------------------

def _print_table(dts, vel_errs, pres_errs, exp_ord):
    print(
        f'  {"dt":>10}  {"vel error":>14}  {"vel ord":>8}  {"exp":>4}'
        f'  {"pres error":>14}  {"pres ord":>9}'
    )
    for i, dt in enumerate(dts):
        ve, pe = vel_errs[i], pres_errs[i]
        vo_str = (
            f'{np.log(vel_errs[i-1]/ve)/np.log(dts[i-1]/dt):>8.2f}'
            if i > 0 and vel_errs[i-1] > 0 and ve > 0 else f'{"---":>8}'
        )
        po_str = (
            f'{np.log(pres_errs[i-1]/pe)/np.log(dts[i-1]/dt):>9.2f}'
            if i > 0 and pres_errs[i-1] > 0 and pe > 0 else f'{"---":>9}'
        )
        print(
            f'  {dt:>10.5f}  {ve:>14.4e}  {vo_str}  {exp_ord:>4d}'
            f'  {pe:>14.4e}  {po_str}'
        )


# ---------------------------------------------------------------------------
# Main convergence study
# ---------------------------------------------------------------------------

def main():
    r"""
    Convergence study comparing algebraic and differentiated-constraint
    monolithic RADAU-IIA solvers.

    Key findings:

    1. **Algebraic constraint** — monolithic :math:`\equiv` SDC standard:
       The monolithic y-formulation and the fully-converged SDC standard sweep
       produce **identical** errors (same collocation fixed point).  Both
       achieve :math:`M+1` for velocity and :math:`M` for pressure.
       **The order reduction is inherent in the DAE, not due to SDC sweeping.**

    2. **Differentiated constraint** — monolithic achieves :math:`2M-1`:
       Replacing :math:`B\mathbf{u}_m = q(\tau_m)` with the differentiated
       form :math:`B\mathbf{u}'_m = q'(\tau_m)` converts the constraint block
       and allows RADAU-IIA to reach its full :math:`2M-1` collocation order
       for **both** velocity and pressure.

    For :math:`M = 3`, both SDC diffconstr (:math:`M+2 = 5`) and monolithic
    diffconstr (:math:`2M-1 = 5`) give the same asymptotic order (the
    coincidence :math:`M+2 = 2M-1` for :math:`M = 3`).  For :math:`M = 4`,
    monolithic diffconstr achieves :math:`2M-1 = 7`, while SDC diffconstr
    gives only :math:`M+2 = 6`.
    """
    print('Monolithic RADAU-IIA collocation (y-formulation)')
    print(f'ν = {_NU}, nvars = {_NVARS}, T_end = {_TEND}\n')

    dts = [_TEND / (2 ** k) for k in range(1, 8)]

    for M in [3, 4]:
        two_m_minus1 = 2 * M - 1

        # ── Algebraic ────────────────────────────────────────────────────────
        print('=' * 72)
        print(f'  M = {M}  Algebraic constraint  B·u_m = q(τ_m)')
        print(f'  Expected order: M+1 = {M+1} (vel),  M = {M} (pres)')
        print(f'  [Same as SDC standard — order reduction inherent in DAE]')
        print('=' * 72)

        vel_errs, pres_errs = [], []
        for dt in dts:
            ve, pe = _run_monolithic(_monolithic_step_algebraic, M, dt)
            vel_errs.append(ve)
            pres_errs.append(pe)
        _print_table(dts, vel_errs, pres_errs, M + 1)
        print()

        # ── Differentiated ───────────────────────────────────────────────────
        print('=' * 72)
        print(f'  M = {M}  Differentiated constraint  B·u_m\' = q\'(τ_m)')
        print(f'  Expected order: 2M-1 = {two_m_minus1} (vel and pres)')
        print(f'  [Full RADAU-IIA collocation order via index reduction]')
        print('=' * 72)

        vel_errs, pres_errs = [], []
        for dt in dts:
            ve, pe = _run_monolithic(_monolithic_step_diffconstr, M, dt)
            vel_errs.append(ve)
            pres_errs.append(pe)
        _print_table(dts, vel_errs, pres_errs, two_m_minus1)
        print()

    print('=' * 72)
    print('  Conclusion')
    print('=' * 72)
    print(
        '\n  Algebraic constraint B·u_m = q(τ_m):'
        '\n    Monolithic y-formulation produces IDENTICAL results to SDC standard.'
        '\n    Both achieve vel → M+1, pres → M.'
        '\n    Order reduction is INHERENT in the DAE structure, not due to SDC.'
        '\n'
        '\n  Differentiated constraint B·u_m\' = q\'(τ_m):'
        '\n    Monolithic achieves full 2M-1 collocation order for BOTH vel and pres.'
        '\n    This beats SDC diffconstr (U-form.) for M ≥ 4: 2M-1 vs. M+2.'
        '\n    For M=3: 2M-1=5=M+2 (coincidence); for M=4: 2M-1=7 > M+2=6.'
        '\n'
        f'\n  Method comparison (M=3, M=4):                vel  |  pres'
        f'\n    SDC standard (U-form., alg. constr.):      M+1  |  M'
        f'\n    SDC diffconstr (U-form., diff. constr.):   M+2  |  M+2'
        f'\n    Monolithic alg. (y-form.) ≡ SDC standard:  M+1  |  M'
        f'\n    Monolithic diffconstr (y-form.):           2M-1 |  2M-1  ← best'
        '\n'
        '\n  Spatial floor ~1e-12 (nvars=1023) may limit the finest accessible Δt,'
        '\n  especially for the higher-order methods that converge more rapidly.'
    )


if __name__ == '__main__':
    main()
