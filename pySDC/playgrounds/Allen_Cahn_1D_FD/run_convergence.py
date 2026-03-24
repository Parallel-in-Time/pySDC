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
    shows that more sweeps yield higher-order accuracy, illustrating the
    key SDC benefit.

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

from pySDC.helpers.stats_helper import get_sorted
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

    SDC theory predicts that *K* sweeps (with an order-1 predictor) yield
    approximately order *K* + 1 in practice.

    Parameters
    ----------
    nvars : int
        Number of interior spatial grid points.
    Tend : float
        Final time.
    num_nodes : int
        Number of SDC collocation nodes.

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

    print('\n' + '=' * 60)
    print('Part 2: Temporal convergence  (nvars = {}, M = {})'.format(nvars, num_nodes))
    print('        error vs. fine-dt reference  (dt_ref = {:.2e})'.format(dt_ref))
    print('=' * 60)

    results = {}
    for K in sweep_counts:
        print(f'\n  K = {K} SDC sweep(s) per step:')
        print(f'  {"dt":>10}  {"error (inf)":>14}  {"order":>8}')
        errs = []
        for dt in dts:
            uend, _, _ = _run(nvars, dt, num_nodes=num_nodes, max_iter=K, restol=1e-20, Tend=Tend)
            err = float(np.linalg.norm(uend - uref, np.inf))
            errs.append(err)
            if len(errs) > 1 and errs[-2] > 0.0:
                order = np.log(errs[-2] / err) / np.log(dts[len(errs) - 2] / dt)
                print(f'  {dt:>10.5f}  {err:>14.4e}  {order:>8.2f}')
            else:
                print(f'  {dt:>10.5f}  {err:>14.4e}  {"---":>8}')
        results[K] = (dts, errs)

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plot(dxs, sp_errors, temp_results):
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
    markers = ['o', 's', '^', 'D']
    colors = ['C0', 'C1', 'C2', 'C3']
    for (K, (dts, errs)), marker, color in zip(temp_results.items(), markers, colors):
        ax2.loglog(dts, errs, marker=marker, color=color, label=f'K={K}')
    ax2.set_xlabel(r'$\Delta t$', fontsize=12)
    ax2.set_ylabel(r'error vs. reference', fontsize=12)
    ax2.set_title('Temporal convergence\n(fixed sweep count K)')
    ax2.legend(fontsize=9)
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
