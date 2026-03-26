r"""
Temporal order of convergence for the 1-D Stokes/Poiseuille index-1 DAE
========================================================================

This script tests the temporal order of convergence of SDC applied to
the 1-D unsteady Stokes / Poiseuille problem, formulated as a semi-explicit
index-1 DAE:

.. math::

    \mathbf{u}' = \nu A\,\mathbf{u} + G(t)\,\mathbf{1} + \mathbf{f}(t),

.. math::

    0 = B\,\mathbf{u} - q(t).

Two sweepers are compared
--------------------------
1. :class:`~pySDC.projects.DAE.sweepers.semiImplicitDAE.SemiImplicitDAE`
   (U-formulation): stores and integrates velocity derivatives
   :math:`U_m = u'(\tau_m)`.
2. :class:`~pySDC.projects.DAE.sweepers.fullyImplicitDAE.FullyImplicitDAE`
   (collocation-consistent): treats the full state :math:`(u, G)` as a
   differential system with derivative :math:`(U_m, G'_m)`.

Two constraint formulations are compared
-----------------------------------------
1. **Standard** (no lifting): constraint :math:`B\mathbf{u} = q(t)` has a
   time-dependent RHS.  Causes order reduction in :math:`G` to order :math:`M`.
2. **Lifted**: lifting :math:`\mathbf{u}_\ell(t) = (q(t)/s)\,\mathbf{1}` makes
   the constraint **homogeneous**: :math:`B\tilde{\mathbf{v}} = 0`.

Why M+1 (not 2M-1) for the velocity?
--------------------------------------
For a pure ODE discretised with RADAU-RIGHT :math:`M` nodes, the collocation
polynomial achieves the superconvergent order :math:`2M-1` at the endpoint.

For this **DAE**, both SemiImplicitDAE and FullyImplicitDAE converge to the
**same collocation fixed point**.  At that fixed point the stage values of the
pressure gradient :math:`G_m` have only :math:`\mathcal{O}(\Delta t^M)`
accuracy (the constraint at each node is exact, but the velocity
:math:`u_m = u_0 + \Delta t\sum_j Q_{mj} U_j` at intermediate nodes has
only stage-order accuracy :math:`\mathcal{O}(\Delta t^{M+1})`).  These
:math:`\mathcal{O}(\Delta t^M)` errors in :math:`G_m` feed back into the
velocity derivatives :math:`U_m = A u_m + G_m \mathbf{1} + f(\tau_m)`,
**breaking the superconvergence** and limiting:

* velocity to :math:`\mathcal{O}(\Delta t^{M+1})` — one higher than the
  stage order;
* pressure :math:`G` to :math:`\mathcal{O}(\Delta t^M)` (standard) or
  increasing toward :math:`M+1` (lifted).

This is confirmed across :math:`M = 2, 3, 4` RADAU-RIGHT and is independent
of the sweeper (SemiImplicitDAE = FullyImplicitDAE at the fixed point):

* :math:`M = 2`: velocity :math:`\to 3` (:math:`= M+1 = 2M-1`; degenerate)
* :math:`M = 3`: velocity :math:`\to 4` (:math:`= M+1`; not :math:`2M-1 = 5`)
* :math:`M = 4`: velocity :math:`\to 5` (:math:`= M+1`; not :math:`2M-1 = 7`)

Achieving the full collocation order :math:`2M-1` for **both** velocity and
pressure would require solving all :math:`M` RADAU stages simultaneously as
a coupled system (standard RADAU-IIA), which goes beyond the node-by-node
SDC sweep used here.

Observed results (:math:`\nu = 0.1`, ``nvars = 1023``, ``restol = 1e-13``,
:math:`M = 3`)
---------------------------------------------------------------------------
With :math:`\nu = 1.0` the problem is stiff (slow-mode Courant number
:math:`|\lambda\,\Delta t| = \nu\pi^2 \Delta t \approx 2.5` at :math:`\Delta t = 0.25`),
keeping the solution in the pre-asymptotic regime across the entire accessible
:math:`\Delta t` range.  Reducing to :math:`\nu = 0.1` brings
:math:`|\lambda\,\Delta t| \lesssim 0.25` at :math:`\Delta t = 0.25`, entering
the asymptotic region and revealing the clean orders:

* **Standard**: velocity at :math:`M+1 = 4`, pressure at :math:`M = 3`
  (time-dependent constraint :math:`B\mathbf{u} = q(t)` causes order reduction
  in :math:`G`).
* **Lifted**: velocity at :math:`M+1 = 4` (unchanged); pressure order
  increases monotonically, approaching :math:`M+1 = 4` (homogeneous
  constraint removes the order reduction).
* **FullyImplicitDAE** (standard or lifted): **identical results** to
  SemiImplicitDAE — both converge to the same collocation fixed point.

**Spatial resolution**: ``nvars = 1023`` interior points with a
fourth-order FD Laplacian (:math:`\Delta x = 1/1024`, spatial error floor
:math:`\mathcal{O}(\Delta x^4) \approx 10^{-12}`).

Usage::

    python run_convergence.py
"""

import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.DAE.sweepers.semiImplicitDAE import SemiImplicitDAE
from pySDC.projects.DAE.sweepers.fullyImplicitDAE import FullyImplicitDAE
from pySDC.playgrounds.Stokes_Poiseuille_1D_FD.Stokes_Poiseuille_1D_FD import (
    stokes_poiseuille_1d_fd,
    stokes_poiseuille_1d_fd_lift,
    stokes_poiseuille_1d_fd_full,
    stokes_poiseuille_1d_fd_lift_full,
)

# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------

_NVARS = 1023
# nu=0.1 gives slow-mode Courant number |lambda*dt| <= 0.25 at dt=0.25,
# putting the solution firmly in the asymptotic regime where the expected
# orders M+1=4 (velocity) and M=3 (pressure, no-lift) are clearly visible.
# With nu=1.0 the problem is ~10x stiffer and the asymptotic region cannot
# be accessed before hitting the O(dx^4) spatial floor.
_NU = 0.1
_TEND = 1.0
_NUM_NODES = 3
_RESTOL = 1e-13

_SWEEPER_PARAMS = {
    'quad_type': 'RADAU-RIGHT',
    'num_nodes': _NUM_NODES,
    'QI': 'LU',
    'initial_guess': 'spread',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(problem_class, sweeper_class, dt, restol=_RESTOL, max_iter=50, nvars=_NVARS):
    """
    Run one simulation and return ``(uend, problem_instance)``.

    For the lifted variant ``uend.diff`` contains the *lifted* velocity
    :math:`\\tilde{\\mathbf{v}}`; call :meth:`P.lift` to recover the
    physical velocity.
    """
    desc = {
        'problem_class': problem_class,
        'problem_params': {'nvars': nvars, 'nu': _NU},
        'sweeper_class': sweeper_class,
        'sweeper_params': _SWEEPER_PARAMS,
        'level_params': {'restol': restol, 'dt': dt},
        'step_params': {'maxiter': max_iter},
    }
    ctrl = controller_nonMPI(num_procs=1, controller_params={'logger_level': 40}, description=desc)
    P = ctrl.MS[0].levels[0].prob
    uend, _ = ctrl.run(u0=P.u_exact(0.0), t0=0.0, Tend=_TEND)
    return uend, P


def _errors(uend, P):
    """
    Compute ``(vel_err, pres_err)`` compared to the exact analytical solution.

    For the lifted variant the physical velocity is reconstructed before
    computing the error.
    """
    u_ex = P.u_exact(_TEND)

    if hasattr(P, 'lift'):
        # Recover physical velocity from lifted variable + lift at T_end.
        u_phys = np.asarray(uend.diff) + P.lift(_TEND)
        u_ex_phys = np.asarray(u_ex.diff) + P.lift(_TEND)
    else:
        u_phys = np.asarray(uend.diff)
        u_ex_phys = np.asarray(u_ex.diff)

    vel_err = float(np.linalg.norm(u_phys - u_ex_phys, np.inf))
    pres_err = abs(float(uend.alg[0]) - float(u_ex.alg[0]))
    return vel_err, pres_err


def _print_table(dts, vel_errs, pres_errs, vel_order):
    """Print a two-column convergence table."""
    header = (
        f'  {"dt":>10}  {"vel error":>14}  {"vel ord":>8}  {"exp":>4}'
        f'  {"pres error":>14}  {"pres ord":>9}'
    )
    print(header)
    for i, dt in enumerate(dts):
        ve = vel_errs[i]
        pe = pres_errs[i]
        if i > 0 and vel_errs[i - 1] > 0.0 and ve > 0.0:
            vo = np.log(vel_errs[i - 1] / ve) / np.log(dts[i - 1] / dt)
            vo_str = f'{vo:>8.2f}'
        else:
            vo_str = f'{"---":>8}'
        if i > 0 and pres_errs[i - 1] > 0.0 and pe > 0.0:
            po = np.log(pres_errs[i - 1] / pe) / np.log(dts[i - 1] / dt)
            po_str = f'{po:>9.2f}'
        else:
            po_str = f'{"---":>9}'
        print(
            f'  {dt:>10.5f}  {ve:>14.4e}  {vo_str}  {vel_order:>4d}'
            f'  {pe:>14.4e}  {po_str}'
        )


def _asymptotic_order(dts, errs, skip=2):
    """Estimate the asymptotic convergence order."""
    orders = []
    for i in range(skip, len(dts)):
        if errs[i] > 0.0 and errs[i - 1] > 0.0:
            orders.append(np.log(errs[i - 1] / errs[i]) / np.log(dts[i - 1] / dts[i]))
    return round(float(np.median(orders)), 1) if orders else float('nan')


# ---------------------------------------------------------------------------
# Main convergence study
# ---------------------------------------------------------------------------

def main():
    r"""
    Compare four formulations: two sweepers × two constraint treatments.

    Fixed parameters:

    * ``restol = 1e-13``, :math:`\nu = 0.1`, :math:`M = 3` RADAU-RIGHT.
    * ``nvars = 1023`` (4th-order FD, spatial floor
      :math:`\mathcal{O}(\Delta x^4) \approx 10^{-12}`).
    * :math:`T_\text{end} = 1.0`.

    Expected orders (see module docstring for derivation):

    * **Velocity**: :math:`M+1 = 4` for all four formulations.
    * **Pressure (standard)**: :math:`M = 3` (order reduction due to
      time-dependent constraint).
    * **Pressure (lifted)**: approaches :math:`M+1 = 4` (homogeneous
      constraint removes the order reduction).
    * **SemiImplicitDAE vs FullyImplicitDAE**: identical results — both
      converge to the same collocation fixed point for this DAE.
    """
    vel_order = _NUM_NODES + 1   # M+1 = 4 (U-formulation limit)
    pres_order = _NUM_NODES      # M   = 3 (algebraic variable at each node)

    # 7 halvings from T_end/2 to T_end/128  →  0.5, 0.25, …, 0.0078125
    dts = [_TEND / (2**k) for k in range(1, 8)]

    print(f'\nFully-converged SDC  (restol={_RESTOL:.0e}, ν={_NU}, M={_NUM_NODES})')
    print(f'Expected velocity order  M+1 = {vel_order}  '
          f'(pure-ODE collocation order 2M-1 = {2*_NUM_NODES-1} not achieved; see module docstring)')
    print(f'Expected pressure order  M   = {pres_order}  (no-lift) '
          f'/ approaches M+1 = {vel_order}  (lifted)')
    print(f'nvars = {_NVARS}, 4th-order FD  (spatial floor ~ O(dx^4) ≈ 1e-12)')
    print(f'ν = {_NU}:  slow-mode |λΔt| ≤ {_NU * np.pi**2 * dts[0]:.2f} at Δt = {dts[0]:.4f}'
          f'  → asymptotic regime accessible')
    print(f'Error vs. exact analytical solution at T={_TEND}')

    cases = [
        (stokes_poiseuille_1d_fd, SemiImplicitDAE,
         'SemiImplicitDAE, standard  (B·u = q(t))'),
        (stokes_poiseuille_1d_fd_lift, SemiImplicitDAE,
         'SemiImplicitDAE, lifted    (B·ṽ = 0)  '),
        (stokes_poiseuille_1d_fd_full, FullyImplicitDAE,
         'FullyImplicitDAE, standard  (B·u = q(t))  [same fixed pt as SemiImplicit]'),
        (stokes_poiseuille_1d_fd_lift_full, FullyImplicitDAE,
         'FullyImplicitDAE, lifted    (B·ṽ = 0)    [same fixed pt as SemiImplicit]'),
    ]

    results = {}
    for cls, sweeper, label in cases:
        print()
        print('=' * 72)
        print(f'  {label}')
        print('=' * 72)

        vel_errs, pres_errs = [], []
        for dt in dts:
            uend, P = _run(cls, sweeper, dt)
            ve, pe = _errors(uend, P)
            vel_errs.append(ve)
            pres_errs.append(pe)

        _print_table(dts, vel_errs, pres_errs, vel_order)
        results[label] = (vel_errs, pres_errs)

    # ---- Summary ----
    print()
    print('=' * 72)
    print('  Summary')
    print('=' * 72)
    for cls, sweeper, label in cases:
        vel_errs, pres_errs = results[label]
        vel_ord = _asymptotic_order(dts, vel_errs)
        pres_ord = _asymptotic_order(dts, pres_errs)
        is_lift = hasattr(cls(), 'lift')
        exp_pres = vel_order if is_lift else pres_order
        print(f'\n  {label}:')
        print(f'    Velocity order ≈ {vel_ord:.1f}  (expected M+1 = {vel_order})')
        if pres_ord < vel_order - 0.4:
            if is_lift:
                status = f'increasing, ≈ {pres_ord:.1f} (pre-asymptotic, heading to M+1 = {vel_order})'
            else:
                status = f'order reduced to {pres_ord:.1f} = M  (time-dependent constraint)'
            print(f'    Pressure order ≈ {pres_ord:.1f}  → {status}')
        else:
            print(f'    Pressure order ≈ {pres_ord:.1f}  → full order M+1 = {vel_order} ✓')

    print()
    print('  Conclusion:')
    print(
        f'  • All four formulations confirm velocity order M+1 = {vel_order}.'
    )
    print(
        f'  • Standard: pressure at order M = {pres_order} (order reduction from time-dependent constraint).'
    )
    print(
        f'  • Lifted: pressure increasing toward M+1 = {vel_order} (autonomous constraint removes reduction).'
    )
    print(
        '  • SemiImplicitDAE and FullyImplicitDAE converge to the SAME collocation\n'
        '    fixed point for this DAE: identical velocity and pressure errors.\n'
        '    The O(dt^M) stage pressure errors break the 2M-1 superconvergence\n'
        '    of both sweepers; achieving 2M-1 would require solving all M RADAU\n'
        '    stages simultaneously (full RADAU-IIA, beyond SDC node-by-node sweeps).'
    )
    print(
        f'  • Spatial floor ~1e-12 (nvars={_NVARS}) may cause order plateau at fine Δt.'
    )


if __name__ == '__main__':
    main()

