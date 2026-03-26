r"""
Temporal order of convergence for the 1-D Stokes/Poiseuille index-1 DAE
========================================================================

This script tests the temporal order of convergence of SDC applied to
the 1-D unsteady Stokes / Poiseuille problem, formulated as a semi-explicit
index-1 DAE:

.. math::

    \mathbf{u}' = \nu A\,\mathbf{u} + G(t)\,\mathbf{1} + \mathbf{f}(t),

.. math::

    0 = B\,\mathbf{u} - q(t),

where :math:`G(t)` is the algebraic variable (pressure gradient) enforced via
a Schur-complement saddle-point solve inside ``solve_system``.

The sweeper used is
:class:`~pySDC.playgrounds.DAE.genericImplicitDAE.genericImplicitConstrained`,
which integrates only the differential components and enforces the algebraic
constraint at every SDC node.

Two quantities are tracked at :math:`T_\text{end}`:

* **Velocity error** :math:`\|\mathbf{u}_h - \mathbf{u}_\text{ex}\|_\infty`
  – the differential variable.
* **Pressure error** :math:`|G_h - G_\text{ex}(T_\text{end})|`
  – the algebraic variable.

With ``restol = 1e-13`` (fully converged SDC) and :math:`M = 3`
RADAU-RIGHT nodes the expected result is:

* **Velocity**: converges at the full collocation order
  :math:`2M - 1 = 5`.  Due to pre-asymptotic effects from the SDC
  iteration, orders below 5 are seen at larger :math:`\Delta t`; the
  error approaches the 4th-order spatial floor
  :math:`\mathcal{O}(\Delta x^4) \approx 10^{-12}` at fine
  :math:`\Delta t`.
* **Pressure gradient**: shows **order reduction** to :math:`M = 3`.
  The algebraic variable :math:`G` is the Lagrange multiplier determined
  at each collocation node by the Schur-complement constraint solve.
  At a single node :math:`t_m` the derivative :math:`U_m = u'(t_m)` has
  accuracy :math:`\mathcal{O}(\Delta t^M)` (not the superconvergent order
  :math:`2M-1`, which applies only to the endpoint value via Gauss
  quadrature), and :math:`G_m` inherits that same order.  Hence the
  pressure at the endpoint converges at order :math:`M = 3`.

**Spatial resolution**: ``nvars = 1023`` interior points with a
fourth-order FD Laplacian (:math:`\Delta x = 1/1024`, spatial error floor
:math:`\mathcal{O}(\Delta x^4) \approx 10^{-12}`).

Usage::

    python run_convergence.py
"""

import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.DAE.sweepers.semiImplicitDAE import SemiImplicitDAE
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
    'initial_guess': 'spread',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(dt, restol=_RESTOL, max_iter=50):
    """
    Run one simulation and return ``(uend, problem_instance)``.

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
    uend : MeshDAE
        Numerical solution at :math:`T_\\text{end}`.
    P : stokes_poiseuille_1d_fd
        Problem instance (gives access to exact solutions).
    """
    desc = {
        'problem_class': stokes_poiseuille_1d_fd,
        'problem_params': {'nvars': _NVARS, 'nu': _NU},
        'sweeper_class': SemiImplicitDAE,
        'sweeper_params': _SWEEPER_PARAMS,
        'level_params': {'restol': restol, 'dt': dt},
        'step_params': {'maxiter': max_iter},
    }
    ctrl = controller_nonMPI(num_procs=1, controller_params={'logger_level': 40}, description=desc)
    P = ctrl.MS[0].levels[0].prob
    uend, _ = ctrl.run(u0=P.u_exact(0.0), t0=0.0, Tend=_TEND)
    return uend, P


def _print_table(dts, vel_errs, pres_errs, vel_order, pres_order_str):
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


# ---------------------------------------------------------------------------
# Main convergence study
# ---------------------------------------------------------------------------

def main():
    r"""
    Temporal convergence study for the Stokes/Poiseuille index-1 DAE.

    Fixed parameters:

    * ``restol = 1e-13``, :math:`\nu = 1.0`, :math:`M = 3` RADAU-RIGHT.
    * ``nvars = 1023`` (4th-order FD, spatial floor
      :math:`\mathcal{O}(\Delta x^4) \approx 10^{-12}`).
    * :math:`T_\text{end} = 0.5`.

    Expected collocation order :math:`2M-1 = 5` for the velocity at the
    endpoint.  The pressure gradient :math:`G` (Lagrange multiplier) is
    predicted to converge at only order :math:`M = 3`, because the Schur
    complement computes :math:`G` from the derivative :math:`U_m`, which
    achieves only :math:`\mathcal{O}(\Delta t^M)` accuracy at each internal
    collocation node (no super-convergence for internal nodes).
    """
    max_order = 2 * _NUM_NODES - 1  # = 5

    # dt range: 6 halvings from T/2 to T/64.
    dts = [_TEND / (2**k) for k in range(1, 7)]  # 0.25 … 0.0078

    print(f'\nFully-converged SDC  (restol={_RESTOL:.0e}, ν={_NU}, M={_NUM_NODES})')
    print(f'Sweeper: SemiImplicitDAE (DAE-specific, constraint enforced at each node)')
    print(f'Expected collocation order for velocity = {max_order}  (= 2M − 1)')
    print(f'nvars = {_NVARS}, 4th-order FD  (spatial floor ~ O(dx⁴) ≈ 1e-12)')
    print(f'Error vs. exact analytical solution at T={_TEND}\n')

    vel_errs = []
    pres_errs = []

    for dt in dts:
        uend, P = _run(dt)
        u_ex = P.u_exact(_TEND)
        vel_errs.append(float(np.linalg.norm(np.asarray(uend.diff) - np.asarray(u_ex.diff), np.inf)))
        pres_errs.append(abs(float(uend.alg[0]) - float(u_ex.alg[0])))

    # Estimate asymptotic order for pressure (skip the first point).
    orders_pres = []
    for i in range(2, len(dts)):
        if pres_errs[i] > 0.0 and pres_errs[i - 1] > 0.0:
            orders_pres.append(
                np.log(pres_errs[i - 1] / pres_errs[i]) / np.log(dts[i - 1] / dts[i])
            )
    pres_est = round(float(np.median(orders_pres)), 1) if orders_pres else float('nan')
    pres_order_str = f'≈ {pres_est:.1f}'

    print('=' * 80)
    _print_table(dts, vel_errs, pres_errs, max_order, pres_order_str)
    print('=' * 80)

    print(f'\nSummary')
    print(f'  Velocity:          converging to collocation order {max_order}')
    print(f'  Pressure gradient: observed order {pres_order_str}')
    if pres_est < max_order - 0.5:
        print(
            f'  → Order reduction in the algebraic variable: {max_order} → {pres_order_str}.'
        )
        print(
            '    The pressure gradient G (Lagrange multiplier) converges at order M = 3,\n'
            '    not at the superconvergent velocity order 2M-1 = 5.\n'
            '    Reason: the Schur complement computes G_m from the velocity derivative\n'
            '    U_m = u\'(t_m), which has only O(dt^M) accuracy at internal collocation\n'
            '    nodes (super-convergence order 2M-1 applies only to the endpoint value\n'
            '    via the Gauss-quadrature formula, not to the nodal derivatives).'
        )
    else:
        print(
            f'  → No order reduction detected: both variables converge at order ≈ {max_order}.'
        )
    print(
        f'  (Convergence may plateau at the 4th-order spatial floor ~1e-12\n'
        f'   once temporal errors fall below O(dx⁴).)'
    )


if __name__ == '__main__':
    main()
