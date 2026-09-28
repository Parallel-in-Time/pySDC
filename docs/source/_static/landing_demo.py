"""
The demo on the website's landing page, run in the browser by landing-demo.js: the residual over the iterations of
SDC or two-level MLSDC for one time step of the 1D Allen-Cahn equation. Each run adds a curve to the plot.
"""

import matplotlib.pyplot as plt
import numpy as np

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.AllenCahn_1D_FD import allencahn_periodic_fullyimplicit
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh

RESTOL = 1e-10
MAXITER = 30
RUNS = []


def residuals(dt, QI, num_nodes, levels, eps):
    """The residual after each iteration of one step, on the finest level"""
    description = {
        'problem_class': allencahn_periodic_fullyimplicit,
        'problem_params': {'nvars': [128, 64][:levels], 'eps': eps, 'dw': 0.0, 'newton_tol': 1e-12},
        'sweeper_class': generic_implicit,
        'sweeper_params': {'num_nodes': num_nodes, 'quad_type': 'RADAU-RIGHT', 'QI': QI, 'initial_guess': 'spread'},
        'level_params': {'dt': dt, 'restol': RESTOL},
        'step_params': {'maxiter': MAXITER},
    }
    if levels > 1:
        description['space_transfer_class'] = mesh_to_mesh
        description['space_transfer_params'] = {'rorder': 2, 'iorder': 2, 'periodic': True}
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 40}, description=description)
    u0 = controller.MS[0].levels[0].prob.u_exact(0.0)
    _, stats = controller.run(u0=u0, t0=0.0, Tend=dt)
    return np.array([residual for _, residual in get_sorted(stats, type='residual_post_iteration', sortby='iter')])


def show(dt, QI, num_nodes, levels, eps):
    """Run one setup, print how it went, and plot it together with the previous runs"""
    res = residuals(dt, QI, num_nodes, levels, eps)
    label = f'{"MLSDC" if levels > 1 else "SDC"}, {QI}, M={num_nodes}, Δt={dt:g}, ε={eps:g}'
    RUNS.append((label, res))
    if res[-1] <= RESTOL:
        print(f'{label}: converged in {len(res)} iterations')
    elif not np.isfinite(res[-1]) or res[-1] > res[0]:
        print(f'{label}: diverged')
    else:
        print(f'{label}: residual {res[-1]:.1e} after {len(res)} iterations, not converged')

    fig, ax = plt.subplots(figsize=(8.5, 3.8), constrained_layout=True)
    for label, res in RUNS[-8:]:
        ax.semilogy(np.arange(1, len(res) + 1), np.where(np.isfinite(res), res, np.nan), marker='o', ms=3, label=label)
    ax.axhline(RESTOL, color='grey', ls=':', lw=1)
    ax.set_xlabel('iteration')
    ax.set_ylabel('residual')
    ax.set_ylim(1e-13, 1e3)
    ax.grid(True, which='major', alpha=0.3)
    ax.legend(fontsize=7, loc='upper left', bbox_to_anchor=(1.01, 1))


def clear():
    """Forget the previous runs"""
    RUNS.clear()
