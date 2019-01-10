import matplotlib
from pySDC.helpers.stats_helper import filter_stats
import numpy as np
from matplotlib import rc
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def show_residual_across_simulation(stats, fname):
    """
    Helper routine to visualize the residuals dependent on the number of iterations across the simulation

    Args:
        stats (dict): statistics object
        fname (str): filename
    """

    # get residuals of the run
    extract_stats = filter_stats(stats, type='residual_post_iteration')

    # find boundaries for x-,y-axis as well as arrays
    maxiter = 0
    for k, v in extract_stats.items():
        iter = getattr(k, 'iter')
        maxiter = max(maxiter, iter)

    # grep residuals and put into array
    residual = np.zeros(maxiter)
    residual[:] = -99
    for k, v in extract_stats.items():
        iter = getattr(k, 'iter')
        if iter is not -1:
            residual[iter - 1] = np.log10(v)

    # Set up latex stuff and fonts
    rc('font', **{"sans-serif": ["Arial"], "size": 30})
    rc('legend', fontsize='small')
    rc('xtick', labelsize='small')
    rc('ytick', labelsize='small')

    # create plot and save
    fig, ax = plt.subplots(figsize=(15, 10))

    ax.set_xlabel('iteration')
    ax.set_ylabel('log10(residual)')

    plt.axis([0, 14, -12, 3])
    plt.plot(np.linspace(1, maxiter, num=maxiter), residual)

    plt.savefig(fname)
