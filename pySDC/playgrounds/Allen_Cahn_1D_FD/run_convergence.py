"""
Convergence study – 1D Allen-Cahn IMEX-SDC
===========================================

Demonstrates the finite-difference / IMEX-SDC implementation of the 1D
Allen-Cahn equation with a travelling-wave solution in two ways:

**Part 1 – Spatial convergence**
    Fix a small time step (SDC converged) and refine the spatial grid.
    The centred-difference Laplacian should give second-order convergence
    in the grid spacing :math:`\\Delta x`.

**Part 2 – Temporal convergence (fixed sweep count)**
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

    The results below confirm this: the computed order approaches the
    expected value :math:`K` as :math:`\\Delta t \\to 0`.  The convergence
    is pre-asymptotic for larger time steps because the nonlinear reaction
    term delays the onset of the asymptotic regime.

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


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _build_description(nvars, dt, num_nodes, max_iter, restol=1e-12):
    """Return a pySDC description dict."""
    return {
        'problem_class': allencahn_1d_imex,
        'problem_params': {
            'nvars': nvars,
            'eps': 0.3,
            'dw': -0.04,
            'interval': (-0.5, 0.5),
        },
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


def _run(nvars, dt, num_nodes=3, max_iter=50, restol=1e-12, t0=0.0, Tend=0.5):
    """Run a single IMEX-SDC simulation and return (solution array, problem, stats)."""
    desc = _build_description(nvars, dt, num_nodes, max_iter, restol)
    ctrl = controller_nonMPI(num_procs=1, controller_params={'logger_level': 40}, description=desc)
    P = ctrl.MS[0].levels[0].prob
    uend, stats = ctrl.run(u0=P.u_exact(t0), t0=t0, Tend=Tend)
    return np.asarray(uend).copy(), P, stats


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
        uend, P, _ = _run(nvars, dt, num_nodes=num_nodes, Tend=Tend)
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
# Part 2: Temporal convergence (fixed sweep count)
# ---------------------------------------------------------------------------

def temporal_convergence(nvars=127, Tend=0.5, num_nodes=3):
    r"""
    Fix the spatial grid and run with a range of time-step sizes *and* a
    fixed number of SDC sweeps *K* per step.  The error is measured against
    a fine-:math:`\Delta t` reference solution on the same grid.

    With a first-order initial guess (``'spread'``), each IMEX-SDC sweep
    increases the order of the time-stepping error by one.  After :math:`K`
    sweeps the expected convergence order is :math:`\min(K,\,2M-1)`, where
    :math:`2M-1` is the maximum collocation order.  For :math:`M = 3` and
    :math:`K \le 4` this is simply :math:`K`.

    Parameters
    ----------
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
    # Reference solution: very fine dt, fully converged SDC
    dt_ref = Tend / 1024
    uref, _, _ = _run(nvars, dt_ref, num_nodes=num_nodes, max_iter=50, restol=1e-13, Tend=Tend)

    dts = [Tend / (2**k) for k in range(1, 7)]
    sweep_counts = [1, 2, 3, 4]
    max_order = 2 * num_nodes - 1  # collocation order = 2M-1

    print('\n' + '=' * 70)
    print('Part 2: Temporal convergence  (nvars = {}, M = {})'.format(nvars, num_nodes))
    print('        error vs. fine-dt reference  (dt_ref = {:.2e})'.format(dt_ref))
    print('        collocation order (max achievable) = 2M-1 = {}'.format(max_order))
    print('=' * 70)

    results = {}
    for K in sweep_counts:
        expected = min(K, max_order)
        print(f'\n  K = {K} SDC sweep(s) per step  (expected order {expected}):')
        print(f'  {"dt":>10}  {"error (inf)":>14}  {"order":>8}  {"expected":>10}')
        errs = []
        for dt in dts:
            uend, _, _ = _run(nvars, dt, num_nodes=num_nodes, max_iter=K, restol=1e-20, Tend=Tend)
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

def make_plot(dxs, sp_errors, temp_results, num_nodes=3):
    """Save a two-panel convergence figure."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # --- Spatial convergence ---
    ax1.loglog(dxs, sp_errors, 'o-', color='C0', label='FD error')
    ref = sp_errors[0] * (np.array(dxs) / dxs[0]) ** 2
    ax1.loglog(dxs, ref, 'k--', label=r'$O(\Delta x^2)$')
    ax1.set_xlabel(r'$\Delta x$', fontsize=12)
    ax1.set_ylabel(r'$\|u_h(T) - u_\mathrm{ex}(T)\|_\infty$', fontsize=12)
    ax1.set_title('Spatial convergence\n(fixed small dt)')
    ax1.legend(fontsize=9)
    ax1.grid(True, which='both', linestyle=':')

    # --- Temporal convergence ---
    max_order = 2 * num_nodes - 1
    markers = ['o', 's', '^', 'D']
    colors = ['C0', 'C1', 'C2', 'C3']
    for (K, (dts, errs)), marker, color in zip(temp_results.items(), markers, colors):
        expected = min(K, max_order)
        ax2.loglog(dts, errs, marker=marker, color=color, label=f'K={K}')
        # Reference slope for expected order K
        ref = errs[-1] * (np.array(dts) / dts[-1]) ** expected
        ax2.loglog(dts, ref, linestyle='--', color=color, alpha=0.5,
                   label=f'order {expected} (theory)')
    ax2.set_xlabel(r'$\Delta t$', fontsize=12)
    ax2.set_ylabel(r'error vs. reference', fontsize=12)
    ax2.set_title(f'Temporal convergence\n(fixed sweep count K, M={num_nodes})')
    ax2.legend(fontsize=8, ncol=2)
    ax2.grid(True, which='both', linestyle=':')

    plt.tight_layout()
    fname = 'allen_cahn_1d_convergence.png'
    plt.savefig(fname, dpi=150)
    print(f'\nPlot saved to {fname}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    dxs, sp_errors = spatial_convergence()
    temp_results = temporal_convergence()
    make_plot(dxs, sp_errors, temp_results)


if __name__ == '__main__':
    main()
