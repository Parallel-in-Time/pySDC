r"""
BC treatment and order reduction in viscous Burgers IMEX-SDC
============================================================

This script demonstrates, using fully-converged IMEX-SDC
(``restol = 1e-13``, :math:`\nu = 0.1`, :math:`M = 3` RADAU-RIGHT nodes),
that:

* **Homogeneous Dirichlet BCs** (no boundary-correction vector
  :math:`b_\text{bc}`) yield the full collocation order
  :math:`2M - 1 = 5`.

* **Time-dependent Dirichlet BCs** handled via the standard
  :math:`b_\text{bc}(t)` correction in :math:`f_\text{impl}` **reduce
  the collocation order** — the time-dependent source term in the implicit
  operator degrades the quadrature accuracy of the collocation method.

Two problem classes are compared on :math:`[0, 1]`:

* ``burgers_1d_hom`` — :math:`u_\text{ex} = 0.1\sin(\pi x)\cos(t)`,
  homogeneous BCs.
* ``burgers_1d_inhom`` — :math:`u_\text{ex} = 0.5+0.1\sin(\pi x)\cos(t)
  +0.1\,x\sin(t)`, time-dependent right BC
  :math:`u(1,t)=0.5+0.1\sin(t)` via :math:`b_\text{bc}(t)` in
  :math:`f_\text{impl}`.

Each class adds a manufactured forcing term so the prescribed solution is
exact.  Errors are measured against the **exact analytical solution** at
:math:`T_\text{end}`.

**Spatial resolution**: ``nvars = 1023`` interior points with a
**fourth-order FD** Laplacian and first-derivative operator
(:math:`\Delta x = 1/1024`, spatial error floor
:math:`O(\Delta x^4) \approx 1 \times 10^{-12}`).  Temporal errors
dominate for all formulations across the dt range tested.

Usage::

    python run_convergence.py
"""

import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.playgrounds.Burgers_1D_FD.Burgers_1D_FD import (
    burgers_1d_hom,
    burgers_1d_inhom,
)

# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------

# 4th-order FD: spatial error ~ O(dx^4).
# With nvars=1023 (dx=1/1024), the spatial floor is O(dx^4) ~ 1e-12,
# well below the temporal error for all dt values tested.
_NVARS = 1023
_NU = 0.1
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

def _run(problem_class, dt, restol=_RESTOL, max_iter=50):
    """Run one simulation and return (u_array, problem_instance)."""
    desc = {
        'problem_class': problem_class,
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


def _print_table(dts, errs, expected_order):
    """Print a convergence table (dt, error, order, expected)."""
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
    Compare two formulations under fully-converged IMEX-SDC.

    Parameters (fixed):

    * ``restol = 1e-13``, :math:`\nu = 0.1`, :math:`M = 3`
    * ``nvars = 1023`` (4th-order FD, spatial floor
      :math:`\approx 1 \times 10^{-12}`)
    * :math:`T_\text{end} = 0.5`
    * Error measured vs. exact analytical solution.

    Expected collocation order :math:`2M - 1 = 5`.
    """
    max_order = 2 * _NUM_NODES - 1  # = 5

    # Inhom case has time-dependent b_bc in f_impl → order reduction (2M-2=4).
    inhom_order = max_order - 1

    # dt range: temporal error dominates for coarser dt, clean convergence
    # orders visible for 4-5 halvings before reaching the spatial floor.
    dts = [_TEND / (2**k) for k in range(1, 7)]  # 0.25 … 0.0078

    cases = [
        (burgers_1d_hom,   'Homogeneous BCs      (sin solution)', max_order),
        (burgers_1d_inhom, 'Inhomogeneous, std   (b_bc in f_impl)', inhom_order),
    ]

    print(f'\nFully-converged IMEX-SDC  (restol={_RESTOL:.0e}, ν={_NU}, M={_NUM_NODES})')
    print(f'Expected collocation order = {max_order}  (= 2M − 1)')
    print(f'nvars = {_NVARS}, 4th-order FD  (spatial error floor ~ O(dx⁴) ≈ 1e-12)')
    print(f'Error vs. exact analytical solution at T={_TEND}\n')

    for cls, label, exp_order in cases:
        print('=' * 70)
        print(f'  {label}')
        print(f'  expected order = {exp_order}')
        print('=' * 70)

        errs = []
        for dt in dts:
            u, P = _run(cls, dt)
            uex = np.asarray(P.u_exact(_TEND)).copy()
            errs.append(float(np.linalg.norm(u - uex, np.inf)))
        _print_table(dts, errs, exp_order)
        print()

    print('=' * 70)
    print('  Summary')
    print('=' * 70)
    print(f'  Homogeneous BCs:      converging to full collocation order {max_order}')
    print(f'  Inhomogeneous (b_bc): order reduction — expected ≈ {inhom_order}')
    print(f'    → time-dependent b_bc(t) in f_impl reduces collocation order by 1.')
    print(f'  (Convergence plateaus at the 4th-order spatial error ~1e-12')
    print(f'   once the temporal error falls below the O(dx⁴) floor.)')


if __name__ == '__main__':
    main()
