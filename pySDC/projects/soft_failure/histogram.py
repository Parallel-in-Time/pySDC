import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def show_iter_hist(stats, fname, maxiter, nruns):
    """
    Helper routine to visualize the maximal iteration number across the simulation in a histogram

    Args:
        stats (dict): statistics object
        fname (str): filename
        maxiter: maximal iterations per run
        nruns: number of runs
    """

    # create plot and save
    fig, ax = plt.subplots(figsize=(15, 10))

    plt.hist(maxiter, bins=np.arange(min(maxiter), max(maxiter) + 2, 1), align='left', rwidth=0.9)

    # with correction allowed: axis instead of xticks
    # plt.axis([12, 51, 0, nruns+1])
    plt.xticks([13, 15, 20, 25, 30, 35, 40, 45, 50])

    ax.set_xlabel('iterations until convergence')

    plt.hlines(nruns, min(maxiter), max(maxiter), colors='red', linestyle='dashed')

    # with correction allowed: no logscale
    plt.yscale('log')

    plt.savefig(fname)
