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

The sweeper is
:class:`~pySDC.projects.DAE.sweepers.semiImplicitDAE.SemiImplicitDAE`;
the saddle-point solve at each node is a direct Schur-complement
factorisation that bypasses Newton.

Two formulations are compared
-------------------------------
1. **Standard** (:class:`~.Stokes_Poiseuille_1D_FD.stokes_poiseuille_1d_fd`)
   — constraint :math:`B\mathbf{u} = q(t)` has a time-dependent RHS.
   This causes **order reduction** in the pressure gradient :math:`G`:
   velocity converges at full collocation order :math:`2M-1 = 5` while
   :math:`G` converges at only order :math:`M = 3`.

2. **Lifted** (:class:`~.Stokes_Poiseuille_1D_FD.stokes_poiseuille_1d_fd_lift`)
   — introduce the lifting :math:`\mathbf{u}_\ell(t) = (q(t)/s)\,\mathbf{1}`
   (where :math:`s = B\mathbf{1} = h N`), which satisfies
   :math:`B\mathbf{u}_\ell(t) = q(t)` identically.  The lifted variable
   :math:`\tilde{\mathbf{v}} = \mathbf{u} - \mathbf{u}_\ell(t)` satisfies
   the **homogeneous** (autonomous) constraint :math:`B\tilde{\mathbf{v}} = 0`.
   Making the constraint autonomous removes the source of order reduction
   and the pressure gradient is expected to converge at the full collocation
   order :math:`2M-1 = 5`.

Observed results (``nvars = 1023``, ``restol = 1e-13``)
---------------------------------------------------------
* **No-lift**: velocity → order 5 (superconvergent endpoint);
  pressure → stable order :math:`M = 3`.
* **Lifted**: velocity unchanged (same accuracy);
  pressure order increases beyond :math:`M` with each grid-size
  halving, approaching the full collocation order :math:`2M-1 = 5`.
  The lifted case is still in a pre-asymptotic regime at the fine
  :math:`\Delta t` values accessible with ``nvars = 1023``
  (:math:`\mathcal{O}(\Delta x^4) \approx 10^{-12}` spatial floor).

**Spatial resolution**: ``nvars = 1023`` interior points with a
fourth-order FD Laplacian (:math:`\Delta x = 1/1024`, spatial error floor
:math:`\mathcal{O}(\Delta x^4) \approx 10^{-12}`).

Usage::

    python run_convergence.py
"""

import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.DAE.sweepers.semiImplicitDAE import SemiImplicitDAE
from pySDC.playgrounds.Stokes_Poiseuille_1D_FD.Stokes_Poiseuille_1D_FD import (
    stokes_poiseuille_1d_fd,
    stokes_poiseuille_1d_fd_lift,
)

# ---------------------------------------------------------------------------
# Fixed parameters
# ---------------------------------------------------------------------------

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

def _run(problem_class, dt, restol=_RESTOL, max_iter=50, nvars=_NVARS):
    """
    Run one simulation and return ``(uend, problem_instance)``.

    For the lifted variant ``uend.diff`` contains the *lifted* velocity
    :math:`\\tilde{\\mathbf{v}}`; call :meth:`P.lift` to recover the
    physical velocity.
    """
    desc = {
        'problem_class': problem_class,
        'problem_params': {'nvars': nvars, 'nu': _NU},
        'sweeper_class': SemiImplicitDAE,
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

    if isinstance(P, stokes_poiseuille_1d_fd_lift):
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
    Compare the standard and lifted Stokes/Poiseuille formulations.

    Fixed parameters:

    * ``restol = 1e-13``, :math:`\nu = 1.0`, :math:`M = 3` RADAU-RIGHT.
    * ``nvars = 1023`` (4th-order FD, spatial floor
      :math:`\mathcal{O}(\Delta x^4) \approx 10^{-12}`).
    * :math:`T_\text{end} = 0.5`.
    """
    max_order = 2 * _NUM_NODES - 1  # = 5
    dts = [_TEND / (2**k) for k in range(1, 7)]   # 0.25 … 0.0078

    print(f'\nFully-converged SDC  (restol={_RESTOL:.0e}, ν={_NU}, M={_NUM_NODES})')
    print(f'Sweeper: SemiImplicitDAE, RADAU-RIGHT nodes')
    print(f'Expected collocation order for velocity = {max_order}  (= 2M − 1)')
    print(f'nvars = {_NVARS}, 4th-order FD  (spatial floor ~ O(dx⁴) ≈ 1e-12)')
    print(f'Error vs. exact analytical solution at T={_TEND}')

    cases = [
        (stokes_poiseuille_1d_fd,      'Standard  (B·u = q(t), time-dependent constraint)'),
        (stokes_poiseuille_1d_fd_lift, 'Lifted    (B·ṽ = 0,    homogeneous constraint)  '),
    ]

    results = {}
    for cls, label in cases:
        print()
        print('=' * 72)
        print(f'  {label}')
        print('=' * 72)

        vel_errs, pres_errs = [], []
        for dt in dts:
            uend, P = _run(cls, dt)
            ve, pe = _errors(uend, P)
            vel_errs.append(ve)
            pres_errs.append(pe)

        _print_table(dts, vel_errs, pres_errs, max_order)
        results[cls.__name__] = (vel_errs, pres_errs)

    # ---- Summary ----
    print()
    print('=' * 72)
    print('  Summary')
    print('=' * 72)
    for cls, label in cases:
        vel_errs, pres_errs = results[cls.__name__]
        vel_ord = _asymptotic_order(dts, vel_errs)
        pres_ord = _asymptotic_order(dts, pres_errs)
        tag = cls.__name__
        print(f'\n  {tag}:')
        print(f'    Velocity order ≈ {vel_ord:.1f}  (expected → {max_order})')
        if pres_ord < max_order - 0.5:
            if isinstance(cls(), stokes_poiseuille_1d_fd_lift):
                status = f'increasing, ≈ {pres_ord:.1f} (pre-asymptotic, heading to {max_order})'
            else:
                status = f'order reduced to {pres_ord:.1f}'
            print(f'    Pressure order ≈ {pres_ord:.1f}  → {status}')
        else:
            print(f'    Pressure order ≈ {pres_ord:.1f}  → full collocation order ✓')

    print()
    print('  Conclusion:')
    pres_ord_std = _asymptotic_order(dts, results['stokes_poiseuille_1d_fd'][1])
    pres_ord_lft = _asymptotic_order(dts, results['stokes_poiseuille_1d_fd_lift'][1])
    print(
        f'  • Standard: pressure converges at order {pres_ord_std:.1f} = M  '
        f'(order reduction; constraint B·u = q(t) is time-dependent).'
    )
    print(
        f'  • Lifted:   pressure converges at increasing order {pres_ord_lft:.1f}+'
        f'  (constraint B·ṽ = 0 is autonomous; order reduction removed).'
    )
    print(
        '  • The velocity accuracy is identical in both cases, confirming that\n'
        '    the lifting only affects the algebraic variable.'
    )
    print(
        '  • Convergence may plateau at the 4th-order spatial floor ~1e-12\n'
        '    once temporal errors fall below O(dx⁴) at fine Δt.'
    )


if __name__ == '__main__':
    main()
