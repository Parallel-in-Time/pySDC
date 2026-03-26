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

Three constraint treatments are compared
-----------------------------------------
1. **Standard** (no lifting, algebraic constraint): constraint
   :math:`B\mathbf{u} = q(t)` has a time-dependent RHS.  Causes order
   reduction in :math:`G` to order :math:`M` (= 3 for 3 RADAU nodes).

2. **Lifted** (homogeneous constraint): lifting
   :math:`\mathbf{u}_\ell(t) = (q(t)/s)\,\mathbf{1}` makes the constraint
   **homogeneous**: :math:`B\tilde{\mathbf{v}} = 0`.  Pressure order
   increases toward :math:`M+1`.

3. **Differentiated constraint** :math:`B\mathbf{u}' = q'(t)`: replaces the
   algebraic constraint with its time derivative.  Converts the DAE to an
   equivalent index-0 ODE system, allowing RADAU to achieve the full
   collocation order :math:`2M-1` for **both** velocity and pressure.

Key findings (:math:`\nu = 0.1`, ``nvars = 1023``, ``restol = 1e-13``,
:math:`M = 3`)
------------------------------------------------------------------------

* **Standard (algebraic)**: velocity :math:`\to M+1 = 4`, pressure
  :math:`\to M = 3`.  The :math:`\mathcal{O}(\Delta t^M)` stage pressure
  errors feed back into the velocity derivatives and break the :math:`2M-1`
  superconvergence.

* **Lifted (homogeneous)**: velocity :math:`\to M+1 = 4` (unchanged);
  pressure order increases monotonically toward :math:`M+1 = 4`.

* **Differentiated constraint**: velocity :math:`\to 2M-1 = 5` ✓,
  pressure :math:`\to 2M-1 = 5` ✓.  By enforcing :math:`B\mathbf{U}_m =
  q'(\tau_m)` at each stage instead of :math:`B\mathbf{u}_m = q(\tau_m)`,
  the stage pressure errors are reduced to
  :math:`e_{G_m} = -BAe_{\mathbf{u}_m}/s = \mathcal{O}(\Delta t^{M+1})`,
  eliminating the feedback that breaks superconvergence.  The index-reduced
  system is essentially an ODE and RADAU achieves its full order :math:`2M-1`.

Why the differentiated constraint works
----------------------------------------
For the original algebraic constraint at each node:

.. math::

    B(\mathbf{v}_m + \alpha\,\mathbf{U}_m) = q(\tau_m)
    \quad\Rightarrow\quad
    G_m = \frac{q(\tau_m) - B\mathbf{v}_m - \alpha B\mathbf{w}}
               {\alpha\,B\mathbf{v}_0}
    = \mathcal{O}(\Delta t^M).

For the differentiated constraint:

.. math::

    B\,\mathbf{U}_m = q'(\tau_m)
    \quad\Rightarrow\quad
    G_m = \frac{q'(\tau_m) - B\mathbf{w}}{B\mathbf{v}_0}
        = G(\tau_m) - \frac{B A\,e_{\mathbf{u}_m}}{s}
        = \mathcal{O}(\Delta t^{M+1}),

since the stage velocity error :math:`e_{\mathbf{u}_m} = \mathcal{O}(\Delta
t^{M+1})` at collocation nodes.  With :math:`G_m` at order :math:`M+1`, the
velocity endpoint recovers the full RADAU superconvergence :math:`2M-1`.

Usage::

    python run_convergence.py
"""

import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.DAE.sweepers.semiImplicitDAE import SemiImplicitDAE
from pySDC.playgrounds.Stokes_Poiseuille_1D_FD.Stokes_Poiseuille_1D_FD import (
    stokes_poiseuille_1d_fd,
    stokes_poiseuille_1d_fd_lift,
    stokes_poiseuille_1d_fd_diffconstr,
    stokes_poiseuille_1d_fd_lift_diffconstr,
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
    Compare three constraint treatments using :class:`SemiImplicitDAE`.

    Fixed parameters:

    * ``restol = 1e-13``, :math:`\nu = 0.1`, :math:`M = 3` RADAU-RIGHT.
    * ``nvars = 1023`` (4th-order FD, spatial floor
      :math:`\mathcal{O}(\Delta x^4) \approx 10^{-12}`).
    * :math:`T_\text{end} = 1.0`.

    Expected orders (see module docstring for derivation):

    * **Standard (algebraic)**: velocity :math:`M+1 = 4`, pressure
      :math:`M = 3`.
    * **Lifted (homogeneous)**: velocity :math:`M+1 = 4`, pressure
      approaches :math:`M+1 = 4`.
    * **Differentiated constraint**: both velocity and pressure
      approach full collocation order :math:`2M-1 = 5`.
    * **Lifted + differentiated**: same as lifted (the differentiated
      homogeneous constraint is equivalent to the original).
    """
    colloc_order = 2 * _NUM_NODES - 1  # 2M-1 = 5

    # 7 halvings from T_end/2 to T_end/128  →  0.5, 0.25, …, 0.0078125
    dts = [_TEND / (2**k) for k in range(1, 8)]

    print(f'\nFully-converged SDC  (restol={_RESTOL:.0e}, ν={_NU}, M={_NUM_NODES})')
    print(f'Full RADAU collocation order: 2M-1 = {colloc_order}')
    print(f'nvars = {_NVARS}, 4th-order FD  (spatial floor ~ O(dx^4) ≈ 1e-12)')
    print(f'ν = {_NU}:  slow-mode |λΔt| ≤ {_NU * np.pi**2 * dts[0]:.2f} at Δt = {dts[0]:.4f}'
          f'  → asymptotic regime accessible')
    print(f'Error vs. exact analytical solution at T={_TEND}')

    cases = [
        (stokes_poiseuille_1d_fd,
         'Standard  (B·u = q(t), algebraic constraint)'),
        (stokes_poiseuille_1d_fd_lift,
         'Lifted    (B·ṽ = 0,   homogeneous constraint)'),
        (stokes_poiseuille_1d_fd_diffconstr,
         'Diff.constr. (B·U = q\'(t), differentiated)  ← remedy'),
        (stokes_poiseuille_1d_fd_lift_diffconstr,
         'Lifted+diff. (B·Ũ = 0,  equiv. to lifted)'),
    ]

    results = {}
    for cls, label in cases:
        print()
        print('=' * 72)
        print(f'  {label}')
        print('=' * 72)

        vel_errs, pres_errs = [], []
        for dt in dts:
            uend, P = _run(cls, SemiImplicitDAE, dt)
            ve, pe = _errors(uend, P)
            vel_errs.append(ve)
            pres_errs.append(pe)

        _print_table(dts, vel_errs, pres_errs, colloc_order)
        results[label] = (vel_errs, pres_errs)

    # ---- Summary ----
    print()
    print('=' * 72)
    print('  Summary')
    print('=' * 72)
    for cls, label in cases:
        vel_errs, pres_errs = results[label]
        vel_ord = _asymptotic_order(dts, vel_errs)
        pres_ord = _asymptotic_order(dts, pres_errs)
        print(f'\n  {label}:')
        print(f'    Velocity order ≈ {vel_ord:.1f}')
        print(f'    Pressure order ≈ {pres_ord:.1f}')

    print()
    print('  Conclusion:')
    print(
        f'  • Standard (algebraic B·u=q): vel→M+1={_NUM_NODES+1}, pres→M={_NUM_NODES}.'
    )
    print(
        f'  • Lifted (B·ṽ=0): vel→M+1, pres increasing toward M+1 (pre-asymptotic).'
    )
    print(
        f'  • Differentiated constraint (B·U=q\'): vel→2M-1={colloc_order}, '
        f'pres→2M-1={colloc_order}.'
    )
    print(
        '    Enforcing B·U_m = q\'(τ_m) at each stage reduces the stage pressure\n'
        '    error from O(dt^M) to O(dt^(M+1)), restoring the full RADAU\n'
        '    superconvergence order 2M-1 for both velocity and pressure.\n'
        '    The differentiated constraint converts the semi-explicit index-1\n'
        '    DAE to an equivalent index-0 ODE system at each stage.'
    )
    print(
        f'  • Lifted+differentiated: same as lifted (the differentiated\n'
        f'    homogeneous constraint B·Ũ=0 is equivalent to the original B·ṽ=0\n'
        f'    at the SDC fixed point; no additional improvement).'
    )
    print(
        f'  • Spatial floor ~1e-12 (nvars={_NVARS}) limits the finest accessible Δt.'
    )


if __name__ == '__main__':
    main()

