"""
Convergence study – 1D Allen-Cahn IMEX-SDC
===========================================

Demonstrates the finite-difference / IMEX-SDC implementation of the 1D
Allen-Cahn equation with a travelling-wave solution in three parts:

**Part 1 – Spatial convergence**
    Fix a small time step (SDC converged) and refine the spatial grid.
    The centred-difference Laplacian should give second-order convergence
    in the grid spacing :math:`\\Delta x`.

**Part 2 – Temporal convergence: standard formulation**
    Fix the spatial grid and perform a fixed number of SDC sweeps *K* per
    time step (instead of iterating to tolerance).  Refining :math:`\\Delta t`
    shows that more sweeps yield higher-order accuracy.

    **Expected orders.**  With Gauss-Radau-Right quadrature (:math:`M` nodes)
    and a constant initial guess (``initial_guess = 'spread'``, which spreads
    :math:`u_0` to all collocation nodes and evaluates :math:`f(u_0)` there,
    giving a first-order-accurate predictor), each IMEX-SDC sweep raises the
    order of accuracy by one.  After :math:`K` sweeps the expected order is
    therefore

    .. math::

        p(K) = \\min(K,\\; 2M-1),

    where :math:`2M-1` is the classical collocation order (the fixed-point
    of the SDC iteration).  For :math:`M = 3` (used here) the maximum is
    :math:`2 \\times 3 - 1 = 5`.  Since we test :math:`K \\in \\{1,2,3,4\\}`,
    the expected orders are simply :math:`1, 2, 3, 4`.

    In practice the order stalls below the expected value for :math:`K \\ge 3`:
    order :math:`\\approx 2.6` for :math:`K = 3` and :math:`\\approx 2.8–3.1`
    for :math:`K = 4`.  This is *pre-asymptotic* behaviour caused by the
    nonlinear explicit reaction term; the observed orders do increase with
    :math:`K` but have not yet reached the asymptotic regime in the
    :math:`\\Delta t` range tested.

**Part 3 – Temporal convergence: boundary-lifting formulation**
    The same study repeated with the boundary-lifted formulation
    (:class:`~AllenCahn_1D_FD_IMEX_Lift.allencahn_1d_imex_lift`).

    **Background.**  In the FEniCS-based problem studied in PR #632, applying
    time-dependent Dirichlet BCs directly inside ``solve_system`` (by
    *overwriting* the solution at boundary DOFs) capped the effective order at
    :math:`\\approx 2` regardless of :math:`K`.  Boundary lifting – reformulating
    the equation for :math:`v = u - E(t)` where :math:`E` is a linear lift
    satisfying the BCs – restored the full collocation order there.

    In the *finite-difference* setting used here, the BCs are handled via a
    boundary-correction vector :math:`b_\\text{bc}(t)` added to
    :math:`f_\\text{impl}`, which is a proper correction consistent with the
    SDC collocation framework (no overwriting occurs).  Consequently, the
    orders for the standard FD formulation are already correct in the
    asymptotic sense.  As the comparison shows, boundary lifting does **not**
    meaningfully improve the orders for this problem: both formulations exhibit
    the same pre-asymptotic stalling for :math:`K \\ge 3`, confirming that
    the stalling originates in the nonlinear explicit reaction term rather than
    the BC treatment.

Usage::

    python run_convergence.py

The script prints convergence tables and saves a log-log plot to
``allen_cahn_1d_convergence.png``.

Parameters used
---------------
* :math:`\\varepsilon = 0.3`  (wider interface – well-resolved on a coarse grid)
* :math:`d_w = -0.04`         (driving force)
* domain :math:`[-0.5, 0.5]`
* :math:`T_{\\text{end}} = 0.5`
* RADAU-RIGHT quadrature with :math:`M = 3` collocation nodes
"""

import numpy as np
import matplotlib.pyplot as plt

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.playgrounds.Allen_Cahn_1D_FD.AllenCahn_1D_FD_IMEX import allencahn_1d_imex
from pySDC.playgrounds.Allen_Cahn_1D_FD.AllenCahn_1D_FD_IMEX_Lift import allencahn_1d_imex_lift


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_PROBLEM_PARAMS = {
    'eps': 0.3,
    'dw': -0.04,
    'interval': (-0.5, 0.5),
}


def _build_description(problem_class, nvars, dt, num_nodes, max_iter, restol=1e-12):
    """Return a pySDC description dict for *problem_class*."""
    return {
        'problem_class': problem_class,
        'problem_params': {'nvars': nvars, **_PROBLEM_PARAMS},
        'sweeper_class': imex_1st_order,
        'sweeper_params': {
            'quad_type': 'RADAU-RIGHT',
            'num_nodes': num_nodes,
            'QI': 'LU',
            'QE': 'EE',
            'initial_guess': 'spread',
        },
        'level_params': {'restol': restol, 'dt': dt},
        'step_params': {'maxiter': max_iter},
    }


def _run(problem_class, nvars, dt, num_nodes=3, max_iter=50, restol=1e-12, t0=0.0, Tend=0.5):
    """
    Run a single IMEX-SDC simulation and return (physical_u_array, problem, stats).

    For the lifted variant the returned array is :math:`v + E(T)`, i.e. the
    physical solution *u* (not the lifted variable *v*).
    """
    desc = _build_description(problem_class, nvars, dt, num_nodes, max_iter, restol)
    ctrl = controller_nonMPI(num_procs=1, controller_params={'logger_level': 40}, description=desc)
    P = ctrl.MS[0].levels[0].prob
    state_end, stats = ctrl.run(u0=P.u_exact(t0), t0=t0, Tend=Tend)
    state_arr = np.asarray(state_end).copy()
    # For the lifted formulation, add back the lift to get physical u.
    if isinstance(P, allencahn_1d_imex_lift):
        state_arr = state_arr + P.lift(Tend)
    return state_arr, P, stats


# ---------------------------------------------------------------------------
# Part 1: Spatial convergence
# ---------------------------------------------------------------------------

def spatial_convergence(Tend=0.5, dt=0.005, num_nodes=3):
    r"""
    Vary *nvars* with a fixed, fine time step to measure the spatial error
    against the exact travelling-wave solution.

    With second-order centred differences the error should decrease as
    :math:`O(\Delta x^2)`.

    Parameters
    ----------
    Tend : float
        Final time.
    dt : float
        Fixed (small) time-step size.
    num_nodes : int
        Number of SDC collocation nodes.

    Returns
    -------
    dxs : list of float
    errors : list of float
    """
    nvars_list = [15, 31, 63, 127, 255, 511]

    print('\n' + '=' * 60)
    print('Part 1: Spatial convergence  (fixed dt = {:.4f})'.format(dt))
    print('=' * 60)
    print(f'{"nvars":>8}  {"dx":>10}  {"error (inf)":>14}  {"order":>8}')

    dxs, errors = [], []
    for nvars in nvars_list:
        dx = 1.0 / (nvars + 1)
        uend, P, _ = _run(allencahn_1d_imex, nvars, dt, num_nodes=num_nodes, Tend=Tend)
        err = float(np.linalg.norm(P.u_exact(Tend) - uend, np.inf))
        dxs.append(dx)
        errors.append(err)
        if len(errors) > 1 and errors[-2] > 0.0:
            order = np.log(errors[-2] / err) / np.log(dxs[-2] / dx)
            print(f'{nvars:>8d}  {dx:>10.5f}  {err:>14.4e}  {order:>8.2f}')
        else:
            print(f'{nvars:>8d}  {dx:>10.5f}  {err:>14.4e}  {"---":>8}')

    return dxs, errors


# ---------------------------------------------------------------------------
# Parts 2 & 3: Temporal convergence – standard vs. boundary-lifted
# ---------------------------------------------------------------------------

def temporal_convergence(problem_class, label, nvars=127, Tend=0.5, num_nodes=3):
    r"""
    Fix the spatial grid and run with a range of time-step sizes *and* a
    fixed number of SDC sweeps *K* per step.  The error is measured against
    a fine-:math:`\Delta t` reference (non-lifted, fully converged) on the
    same grid.

    Parameters
    ----------
    problem_class : type
        Either :class:`allencahn_1d_imex` or :class:`allencahn_1d_imex_lift`.
    label : str
        Short label printed in the header (e.g. ``'standard'`` or ``'lifted'``).
    nvars : int
        Number of interior spatial grid points.
    Tend : float
        Final time.
    num_nodes : int
        Number of SDC collocation nodes (:math:`M`).

    Returns
    -------
    results : dict
        ``{K: (dts, errors)}`` for each sweep count *K* tried.
    """
    # Common fine-dt reference: always use the non-lifted formulation so that
    # both comparisons are measured against the same reference u.
    dt_ref = Tend / 1024
    uref, _, _ = _run(allencahn_1d_imex, nvars, dt_ref, num_nodes=num_nodes,
                      max_iter=50, restol=1e-13, Tend=Tend)

    dts = [Tend / (2**k) for k in range(1, 7)]
    sweep_counts = [1, 2, 3, 4]
    max_order = 2 * num_nodes - 1  # collocation order = 2M-1

    print('\n' + '=' * 70)
    print(f'Temporal convergence  [{label}]  (nvars = {nvars}, M = {num_nodes})')
    print(f'  error measured in physical u against non-lifted reference'
          f'  (dt_ref = {dt_ref:.2e})')
    print(f'  collocation order (max achievable) = 2M-1 = {max_order}')
    print('=' * 70)

    results = {}
    for K in sweep_counts:
        expected = min(K, max_order)
        print(f'\n  K = {K} SDC sweep(s) per step  (expected order {expected}):')
        print(f'  {"dt":>10}  {"error (inf)":>14}  {"order":>8}  {"expected":>10}')
        errs = []
        for dt in dts:
            uend, _, _ = _run(problem_class, nvars, dt, num_nodes=num_nodes,
                              max_iter=K, restol=1e-20, Tend=Tend)
            err = float(np.linalg.norm(uend - uref, np.inf))
            errs.append(err)
            if len(errs) > 1 and errs[-2] > 0.0:
                order = np.log(errs[-2] / err) / np.log(dts[len(errs) - 2] / dt)
                print(f'  {dt:>10.5f}  {err:>14.4e}  {order:>8.2f}  {expected:>10d}')
            else:
                print(f'  {dt:>10.5f}  {err:>14.4e}  {"---":>8}  {expected:>10d}')
        results[K] = (dts, errs)

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plot(dxs, sp_errors, temp_std, temp_lift, num_nodes=3):
    """Save a three-panel convergence figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax1, ax2, ax3 = axes

    # --- Spatial convergence ---
    ax1.loglog(dxs, sp_errors, 'o-', color='C0', label='FD error')
    ref = sp_errors[0] * (np.array(dxs) / dxs[0]) ** 2
    ax1.loglog(dxs, ref, 'k--', label=r'$O(\Delta x^2)$')
    ax1.set_xlabel(r'$\Delta x$', fontsize=12)
    ax1.set_ylabel(r'$\|u_h(T) - u_\mathrm{ex}(T)\|_\infty$', fontsize=12)
    ax1.set_title('Spatial convergence\n(fixed small dt)')
    ax1.legend(fontsize=9)
    ax1.grid(True, which='both', linestyle=':')

    # --- Temporal convergence helper ---
    max_order = 2 * num_nodes - 1
    markers = ['o', 's', '^', 'D']
    colors = ['C0', 'C1', 'C2', 'C3']

    def _plot_temporal(ax, results, title):
        for (K, (dts, errs)), marker, color in zip(results.items(), markers, colors):
            expected = min(K, max_order)
            ax.loglog(dts, errs, marker=marker, color=color, label=f'K={K}')
            ref = errs[-1] * (np.array(dts) / dts[-1]) ** expected
            ax.loglog(dts, ref, linestyle='--', color=color, alpha=0.5,
                      label=f'order {expected} (theory)')
        ax.set_xlabel(r'$\Delta t$', fontsize=12)
        ax.set_ylabel(r'error vs. reference', fontsize=12)
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, which='both', linestyle=':')

    _plot_temporal(ax2, temp_std,
                   f'Temporal – standard\n(fixed K, M={num_nodes})')
    _plot_temporal(ax3, temp_lift,
                   f'Temporal – boundary lifting\n(fixed K, M={num_nodes})')

    plt.tight_layout()
    fname = 'allen_cahn_1d_convergence.png'
    plt.savefig(fname, dpi=150)
    print(f'\nPlot saved to {fname}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dxs, sp_errors = spatial_convergence()
    print('\n' + '#' * 70)
    print('# Part 2: Standard formulation (b_bc correction in solve_system) #')
    print('#' * 70)
    temp_std = temporal_convergence(allencahn_1d_imex, 'standard')
    print('\n' + '#' * 70)
    print('# Part 3: Boundary-lifting formulation (homogeneous BCs for v)   #')
    print('#' * 70)
    temp_lift = temporal_convergence(allencahn_1d_imex_lift, 'lifted')
    make_plot(dxs, sp_errors, temp_std, temp_lift)


if __name__ == '__main__':
    main()

