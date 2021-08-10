import matplotlib
matplotlib.use('Agg')

from pySDC.helpers.stats_helper import filter_stats

import numpy as np

from matplotlib import rc
import matplotlib.pyplot as plt


# noinspection PyShadowingBuiltins
def show_residual_across_simulation(stats, fname='residuals.png'):
    """
    Helper routine to visualize the residuals across the simulation (one block of PFASST)

    Args:
        stats (dict): statistics object from a PFASST run
        fname (str): filename
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
        if iter is not -1:
            residual[iter - 1, step] = np.log10(v)

    # Set up latex stuff and fonts
    rc('font', **{"sans-serif": ["Arial"], "size": 30})
    rc('legend', fontsize='small')
    rc('xtick', labelsize='small')
    rc('ytick', labelsize='small')

    # create plot and save
    fig, ax = plt.subplots(figsize=(15, 10))

    cmap = plt.get_cmap('Reds')
    plt.pcolor(residual.T, cmap=cmap, vmin=minres, vmax=maxres)

    cax = plt.colorbar()
    cax.set_label('log10(residual)')

    ax.set_xlabel('iteration')
    ax.set_ylabel('process')

    ax.set_xticks(np.arange(maxiter) + 0.5, minor=False)
    ax.set_yticks(np.arange(maxprocs + 1) + 0.5, minor=False)
    ax.set_xticklabels(np.arange(maxiter) + 1, minor=False)
    ax.set_yticklabels(np.arange(maxprocs + 1), minor=False)

    plt.savefig(fname, transparent=True, bbox_inches='tight')
