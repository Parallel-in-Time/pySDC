r"""
Temporal order of convergence — index-1 semi-explicit DAE
==========================================================

This script tests the temporal order of convergence of semi-implicit SDC
applied to the index-1 semi-explicit DAE

.. math::
    y'(t) = -\lambda y(t) + z(t) + (\lambda - 1)\sin(t),

.. math::
    0 = z(t) - (y(t) + \cos(t)),

whose analytical solution is

.. math::
    y_{\mathrm{ex}}(t) = \sin(t), \quad z_{\mathrm{ex}}(t) = \sin(t) + \cos(t).

**Goal**: with :math:`M = 3` RADAU-RIGHT quadrature nodes and fully-converged
SDC (``restol = 1e-13``, ``maxiter = 50``), confirm that:

* The **differential variable** :math:`y` achieves the full collocation order
  :math:`2M - 1 = 5`.
* Whether the **algebraic variable** :math:`z` shows **order reduction** or
  also achieves the full collocation order.

The SemiImplicitDAE sweeper is used, which treats :math:`y'` and :math:`z`
as unknowns at each collocation node and only integrates the differential
components.

Usage::

    python run_convergence.py
"""

import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.DAE.sweepers.semiImplicitDAE import SemiImplicitDAE
from pySDC.playgrounds.DAE_index1.index1_dae import index1_semiexplicit_dae

# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------

_LAM = 1.0      # stiffness parameter λ
_T0 = 0.0
_TEND = 1.0
_NUM_NODES = 3  # RADAU-RIGHT quadrature nodes
_RESTOL = 1e-13  # tight tolerance → SDC has converged

_SWEEPER_PARAMS = {
    'quad_type': 'RADAU-RIGHT',
    'num_nodes': _NUM_NODES,
    'QI': 'LU',
    'initial_guess': 'spread',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(dt):
    """
    Run one simulation and return ``(uend, problem_instance)``.

    Parameters
    ----------
    dt : float
        Time-step size.

    Returns
    -------
    uend : MeshDAE
        Solution at the final time; ``.diff[0]`` = y, ``.alg[0]`` = z.
    P : index1_semiexplicit_dae
        Problem instance (used to evaluate the exact solution).
    """
    desc = {
        'problem_class': index1_semiexplicit_dae,
        'problem_params': {'lam': _LAM, 'newton_tol': 1e-12},
        'sweeper_class': SemiImplicitDAE,
        'sweeper_params': _SWEEPER_PARAMS,
        'level_params': {'restol': _RESTOL, 'dt': dt},
        'step_params': {'maxiter': 50},
    }
    ctrl = controller_nonMPI(num_procs=1, controller_params={'logger_level': 40}, description=desc)
    P = ctrl.MS[0].levels[0].prob
    uend, _ = ctrl.run(u0=P.u_exact(_T0), t0=_T0, Tend=_TEND)
    return uend, P


def _print_table(dts, errs, expected_order, var_name):
    """Print a convergence table for one variable."""
    print(f'  {var_name}:')
    print(f'  {"dt":>10}  {"error (inf)":>14}  {"order":>8}  {"expected":>10}')
    for i, (dt, err) in enumerate(zip(dts, errs)):
        if i > 0 and errs[i - 1] > 0.0 and err > 0.0:
            order = np.log(errs[i - 1] / err) / np.log(dts[i - 1] / dt)
            print(f'  {dt:>10.5f}  {err:>14.4e}  {order:>8.2f}  {expected_order:>10d}')
        else:
            print(f'  {dt:>10.5f}  {err:>14.4e}  {"---":>8}  {expected_order:>10d}')


# ---------------------------------------------------------------------------
# Main study
# ---------------------------------------------------------------------------

def main():
    r"""
    Compare convergence orders for the differential variable :math:`y` and
    the algebraic variable :math:`z` under fully-converged semi-implicit SDC.

    Parameters (fixed):

    * ``restol = 1e-13``, ``maxiter = 50``, :math:`M = 3` RADAU-RIGHT nodes
    * :math:`\lambda = 1`, :math:`T_{\mathrm{end}} = 1`
    * Error measured vs. analytical solution at :math:`T_{\mathrm{end}}`.

    Full collocation order for RADAU-RIGHT: :math:`2M - 1 = 5`.
    """
    coll_order = 2 * _NUM_NODES - 1  # 5

    # dt range: coarse to fine in factors of 2
    dts = [_TEND / (2**k) for k in range(1, 7)]  # 0.5, 0.25, 0.125, ...

    errs_y = []
    errs_z = []

    for dt in dts:
        uend, P = _run(dt)
        uex = P.u_exact(_TEND)
        errs_y.append(abs(float(uend.diff[0]) - float(uex.diff[0])))
        errs_z.append(abs(float(uend.alg[0]) - float(uex.alg[0])))

    print(f'\nFully-converged Semi-Implicit-SDC  (restol={_RESTOL:.0e}, λ={_LAM}, M={_NUM_NODES})')
    print(f'Expected collocation order = {coll_order}  (= 2M − 1 for RADAU-RIGHT)')
    print(f't ∈ [{_T0}, {_TEND}],  error vs. analytical solution at T_end\n')

    print('=' * 66)
    _print_table(dts, errs_y, coll_order, 'y  (differential variable)')
    print('=' * 66)
    print()
    print('=' * 66)
    _print_table(dts, errs_z, coll_order, 'z  (algebraic variable)  ')
    print('=' * 66)

    # ---- summary ----
    # Compute observed order for last two refinements
    def _obs_order(errs):
        if errs[-1] > 0.0 and errs[-2] > 0.0:
            return np.log(errs[-2] / errs[-1]) / np.log(dts[-2] / dts[-1])
        return float('nan')

    oy = _obs_order(errs_y)
    oz = _obs_order(errs_z)

    print()
    print('=' * 66)
    print('  Summary')
    print('=' * 66)
    print(f'  y (differential):  observed order ≈ {oy:.2f}  (expected {coll_order})')
    print(f'  z (algebraic):     observed order ≈ {oz:.2f}  (expected {coll_order})')
    if abs(oy - coll_order) < 0.5:
        print(f'  → y achieves full collocation order {coll_order}. ✓')
    else:
        print(f'  → y does NOT reach full collocation order {coll_order}.')
    if abs(oz - coll_order) < 0.5:
        print(f'  → z also achieves full collocation order {coll_order}. No order reduction.')
        print(f'     (z is directly recovered from the constraint z = y + cos(t) at each')
        print(f'      collocation node, so it inherits the full order of y.)')
    else:
        print(f'  → z shows order reduction (≈ {oz:.2f} < {coll_order}).')


if __name__ == '__main__':
    main()
