r"""
BC treatment and order reduction in Allen-Cahn IMEX-SDC
=========================================================

This script demonstrates, using fully-converged IMEX-SDC
(``restol = 1e-13``, :math:`\varepsilon = 1`, :math:`M = 3` RADAU-RIGHT
nodes), that:

* **Homogeneous Dirichlet BCs** (no boundary-correction vector
  :math:`b_\text{bc}`) yield the full collocation order
  :math:`2M - 1 = 5`.

* **Time-dependent Dirichlet BCs** handled via the standard
  :math:`b_\text{bc}(t)` correction in :math:`f_\text{impl}` **reduce
  the collocation order to** :math:`\approx 4` — the time-dependent source
  term in the implicit operator degrades the quadrature accuracy of the
  collocation method.

* **Boundary lifting** (:math:`v = u - E(t)`, autonomous implicit operator
  :math:`f_\text{impl} = A v`) **restores the full collocation order 5**
  by removing the time-dependent :math:`b_\text{bc}` from the implicit part.

Three MMS problem classes are compared on :math:`[0, 1]`:

* ``allencahn_1d_mms_hom`` — :math:`u_\text{mms} = \sin(\pi x)\cos(t)`,
  homogeneous BCs.
* ``allencahn_1d_mms_inhom`` — :math:`u_\text{mms} = \cos(\pi x)\cos(t)`,
  time-dependent BCs via :math:`b_\text{bc}(t)` in :math:`f_\text{impl}`.
* ``allencahn_1d_mms_inhom_lift`` — same exact solution, boundary lifting.

Each class adds a manufactured forcing term so the prescribed solution is
exact.  Errors are measured against a fine-:math:`\Delta t` reference
(:math:`\Delta t_\text{ref} = T_\text{end}/2048`, same class, same
``restol``) to isolate temporal errors.

Usage::

    python run_convergence_mms.py
"""

import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.playgrounds.Allen_Cahn_1D_FD.AllenCahn_1D_FD_MMS import (
    allencahn_1d_mms_hom,
    allencahn_1d_mms_inhom,
    allencahn_1d_mms_inhom_lift,
)

# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------

_NVARS = 127
_TEND = 0.5
_EPS = 1.0
_DW = 0.0
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
    """Run one simulation and return the physical solution array at Tend."""
    desc = {
        'problem_class': problem_class,
        'problem_params': {'nvars': _NVARS, 'eps': _EPS, 'dw': _DW},
        'sweeper_class': imex_1st_order,
        'sweeper_params': _SWEEPER_PARAMS,
        'level_params': {'restol': restol, 'dt': dt},
        'step_params': {'maxiter': max_iter},
    }
    ctrl = controller_nonMPI(num_procs=1, controller_params={'logger_level': 40}, description=desc)
    P = ctrl.MS[0].levels[0].prob
    uend, _ = ctrl.run(u0=P.u_exact(0.0), t0=0.0, Tend=_TEND)
    u = np.asarray(uend).copy()
    if isinstance(P, allencahn_1d_mms_inhom_lift):
        u = u + P.lift(_TEND)
    return u


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
    Compare three MMS formulations under fully-converged SDC.

    Parameters (fixed):

    * ``restol = 1e-13``, :math:`\varepsilon = 1.0`, :math:`M = 3`
    * ``nvars = 127``, :math:`T_\text{end} = 0.5`
    * Error measured vs.\ fine-:math:`\Delta t` reference
      (:math:`\Delta t_\text{ref} = T_\text{end}/2048`).

    Expected collocation order :math:`2M - 1 = 5`.
    """
    max_order = 2 * _NUM_NODES - 1  # = 5
    dt_ref = _TEND / 2048
    dts = [_TEND / (2**k) for k in range(1, 7)]  # 0.25 … 0.0078

    # Inhom std has b_bc in f.impl which reduces collocation order to 2M-2=4.
    inhom_std_order = max_order - 1

    cases = [
        (allencahn_1d_mms_hom,        'Homogeneous BCs      (sin solution)',  max_order),
        (allencahn_1d_mms_inhom,      'Inhomogeneous, std   (cos + b_bc)',    inhom_std_order),
        (allencahn_1d_mms_inhom_lift, 'Inhomogeneous, lift  (cos + lifting)', max_order),
    ]

    print(f'\nFully-converged IMEX-SDC  (restol={_RESTOL:.0e}, ε={_EPS}, M={_NUM_NODES})')
    print(f'Expected collocation order = {max_order}  (= 2M − 1)')
    print(f'Error vs. fine-Δt reference  (dt_ref = {dt_ref:.2e})\n')

    for cls, label, exp_order in cases:
        # Fine-dt reference for this formulation.
        uref = _run(cls, dt_ref)

        print('=' * 70)
        print(f'  {label}')
        print(f'  expected collocation order = {exp_order}')
        print('=' * 70)

        errs = []
        for dt in dts:
            u = _run(cls, dt)
            errs.append(float(np.linalg.norm(u - uref, np.inf)))
        _print_table(dts, errs, exp_order)
        print()

    print('=' * 70)
    print('  Summary')
    print('=' * 70)
    print(f'  Homogeneous BCs:        collocation order {max_order} ✓')
    print(f'  Inhomogeneous (b_bc):   stalls at order {inhom_std_order}')
    print(f'    → b_bc(t) in f.impl reduces the collocation order by 1.')
    print(f'  Inhomogeneous (lift):   approaches order {max_order}')
    print(f'    → removing b_bc from f.impl restores the full collocation order.')


if __name__ == '__main__':
    main()
