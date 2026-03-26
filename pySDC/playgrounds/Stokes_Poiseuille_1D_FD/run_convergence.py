r"""
Temporal order of convergence for the 1-D Stokes/Poiseuille index-1 DAE
========================================================================

This script tests the temporal order of convergence of IMEX-SDC applied to
the 1-D unsteady Stokes / Poiseuille problem, formulated as a semi-explicit
index-1 DAE:

.. math::

    \mathbf{u}' = \nu A\,\mathbf{u} + G(t)\,\mathbf{1} + \mathbf{f}(t),

.. math::

    0 = B\,\mathbf{u} - q(t),

where :math:`G(t)` is the algebraic variable (pressure gradient) and the
constraint is enforced via a Schur-complement saddle-point solve inside
``solve_system``.

Two quantities are tracked at :math:`T_\text{end}`:

* **Velocity error** :math:`\|\mathbf{u}_h - \mathbf{u}_\text{ex}\|_\infty`
  – the differential variable.
* **Pressure error** :math:`|G_h - G_\text{ex}(T_\text{end})|`
  – the algebraic variable.

With ``restol = 1e-13`` (fully converged SDC) and :math:`M = 3`
RADAU-RIGHT nodes the expected result is:

* **Velocity**: converges at the full collocation order
  :math:`2M - 1 = 5`.
* **Pressure gradient**: may show **order reduction** compared to the
  velocity.  For an index-1 DAE the algebraic variable is determined via a
  Schur-complement involving a :math:`1/\Delta t` factor, which can degrade
  the convergence order relative to the differential variable.

**Spatial resolution**: ``nvars = 1023`` interior points with a
fourth-order FD Laplacian (:math:`\Delta x = 1/1024`, spatial error floor
:math:`\mathcal{O}(\Delta x^4) \approx 10^{-12}`).

Usage::

    python run_convergence.py
"""

import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.playgrounds.Stokes_Poiseuille_1D_FD.Stokes_Poiseuille_1D_FD import stokes_poiseuille_1d_fd

# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------

# 4th-order FD Laplacian: spatial error ~ O(dx^4).
# With nvars=1023 (dx=1/1024), spatial floor ~ 1e-12.
_NVARS = 1023
_NU = 1.0
_TEND = 0.5
_NUM_NODES = 3
_RESTOL = 1e-13

_SWEEPER_PARAMS = {
    'quad_type': 'RADAU-RIGHT',
    'num_nodes': _NUM_NODES,
    'QI': 'LU',
    'QE': 'EE',
    'initial_guess': 'spread',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(dt, restol=_RESTOL, max_iter=50):
    """
    Run one simulation and return ``(uend_array, problem_instance)``.

    Parameters
    ----------
    dt : float
        Time-step size.
    restol : float
        Residual tolerance for SDC convergence.
    max_iter : int
        Maximum number of SDC iterations per step.

    Returns
    -------
    u : numpy.ndarray
        Numerical velocity at :math:`T_\text{end}`.
    P : stokes_poiseuille_1d_fd
        Problem instance (gives access to ``_G_last`` and exact solutions).
    """
    desc = {
        'problem_class': stokes_poiseuille_1d_fd,
        'problem_params': {'nvars': _NVARS, 'nu': _NU},
        'sweeper_class': imex_1st_order,
        'sweeper_params': _SWEEPER_PARAMS,
        'level_params': {'restol': restol, 'dt': dt},
        'step_params': {'maxiter': max_iter},
    }
    ctrl = controller_nonMPI(num_procs=1, controller_params={'logger_level': 40}, description=desc)
    P = ctrl.MS[0].levels[0].prob
    uend, _ = ctrl.run(u0=P.u_exact(0.0), t0=0.0, Tend=_TEND)
    return np.asarray(uend).copy(), P


def _print_table(dts, vel_errs, pres_errs, vel_order, pres_order):
    """Print a two-column convergence table."""
    header = (
        f'  {"dt":>10}  {"vel error":>14}  {"vel ord":>8}  {"exp":>4}'
        f'  {"pres error":>14}  {"pres ord":>9}  {"exp":>4}'
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
            f'  {pe:>14.4e}  {po_str}  {pres_order:>4d}'
        )


# ---------------------------------------------------------------------------
# Main convergence study
# ---------------------------------------------------------------------------

def main():
    r"""
    Temporal convergence study for the Stokes/Poiseuille index-1 DAE.

    Fixed parameters:

    * ``restol = 1e-13``, :math:`\nu = 1.0`, :math:`M = 3` RADAU-RIGHT.
    * ``nvars = 1023`` (4th-order FD, spatial floor :math:`\approx 10^{-12}`).
    * :math:`T_\text{end} = 0.5`.

    Expected collocation order :math:`2M-1 = 5` for the velocity.
    Check whether the pressure gradient :math:`G` shows order reduction.
    """
    max_order = 2 * _NUM_NODES - 1  # = 5

    # dt range: temporal errors dominate for coarser dt; 6 halvings from T/2.
    dts = [_TEND / (2**k) for k in range(1, 7)]  # 0.25 … 0.0078

    print(f'\nFully-converged IMEX-SDC  (restol={_RESTOL:.0e}, ν={_NU}, M={_NUM_NODES})')
    print(f'Expected collocation order for velocity = {max_order}  (= 2M − 1)')
    print(f'nvars = {_NVARS}, 4th-order FD  (spatial floor ~ O(dx⁴) ≈ 1e-12)')
    print(f'Error vs. exact analytical solution at T={_TEND}')
    print(f'Pressure G is extracted from the Schur-complement solve\n')

    vel_errs = []
    pres_errs = []

    for dt in dts:
        u, P = _run(dt)
        u_ex = np.asarray(P.u_exact(_TEND)).copy()
        G_ex = P.G_exact(_TEND)
        vel_errs.append(float(np.linalg.norm(u - u_ex, np.inf)))
        pres_errs.append(abs(P._G_last - G_ex))

    # Estimate asymptotic order for pressure from the middle refinement steps
    # (avoid the regime where spatial floor dominates or pre-asymptotic).
    orders_pres = []
    for i in range(2, len(dts)):
        if pres_errs[i] > 0.0 and pres_errs[i - 1] > 0.0:
            orders_pres.append(
                np.log(pres_errs[i - 1] / pres_errs[i]) / np.log(dts[i - 1] / dts[i])
            )
    pres_est = int(round(np.median(orders_pres))) if orders_pres else 0

    print('=' * 80)
    _print_table(dts, vel_errs, pres_errs, max_order, pres_est)
    print('=' * 80)

    print(f'\nSummary')
    print(f'  Velocity:          converging to collocation order {max_order}')
    print(f'  Pressure gradient: observed order ≈ {pres_est}')
    if pres_est < max_order:
        print(
            f'  → Order reduction in the algebraic variable: {max_order} → {pres_est}.'
        )
        print(
            '    This is expected for index-1 DAEs: the Schur-complement solve'
            ' introduces a 1/Δt factor that degrades the convergence of G by one order.'
        )
    else:
        print(
            f'  → No order reduction detected: both variables converge at order {max_order}.'
        )
    print(
        f'  (Convergence may plateau at the 4th-order spatial floor ~1e-12'
        f'\n   once temporal errors fall below O(dx⁴).)'
    )


if __name__ == '__main__':
    main()
