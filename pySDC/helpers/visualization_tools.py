import matplotlib.pyplot as plt
import numpy as np

from pySDC.helpers.stats_helper import filter_stats


# noinspection PyShadowingBuiltins
def show_residual_across_simulation(stats, fname='residuals.png'):
    """
    Helper routine to visualize the residuals across the simulation (one block of PFASST)

    Args:
        stats (dict): statistics object from a PFASST run
        fname (str): filename

    Returns:
        matplotlib.figure.Figure: the figure, which is also saved to `fname`
    """

    # get residuals of the run
    extract_stats = filter_stats(stats, type='residual_post_iteration')

    # find boundaries for x-,y- and c-axis as well as arrays
    maxprocs = 0
    maxiter = 0
    minres = 0
    maxres = -99
    for k, v in extract_stats.items():
        maxprocs = max(maxprocs, k.process)
        maxiter = max(maxiter, k.iter)
        minres = min(minres, np.log10(v))
        maxres = max(maxres, np.log10(v))

    # grep residuals and put into array
    residual = np.zeros((maxiter, maxprocs + 1))
    residual[:] = -99
    for k, v in extract_stats.items():
        step = k.process
        iter = k.iter
        if iter != -1:
            residual[iter - 1, step] = np.log10(v)

    # large fonts for this figure only, not for every figure made after it
    with plt.rc_context(
        {'font.size': 30, 'legend.fontsize': 'small', 'xtick.labelsize': 'small', 'ytick.labelsize': 'small'}
    ):
        fig, ax = plt.subplots(figsize=(15, 10))

        mesh = ax.pcolor(residual.T, cmap=plt.get_cmap('Reds'), vmin=minres, vmax=maxres)
        fig.colorbar(mesh, ax=ax).set_label('log10(residual)')

        ax.set_xlabel('iteration')
        ax.set_ylabel('process')

        ax.set_xticks(np.arange(maxiter) + 0.5, minor=False)
        ax.set_yticks(np.arange(maxprocs + 1) + 0.5, minor=False)
        ax.set_xticklabels(np.arange(maxiter) + 1, minor=False)
        ax.set_yticklabels(np.arange(maxprocs + 1), minor=False)

        fig.savefig(fname, transparent=True, bbox_inches='tight')
    return fig
