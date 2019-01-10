import matplotlib
import os
import numpy as np
from matplotlib import rc
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def show_min_max_residual_across_simulation(stats, fname, minres, maxres, meanres, medianres, maxiter):
    """
    Helper routine to visualize the minimal, maximal, mean, median residual vectors dependent on the
    number of iterations across the simulation

    Args:
        stats (dict): statistics object
        fname (str): filename
        minres: minimal residual vector
        maxres: maximal residual vector
        meanres: mean residual vector
        medianres: median residual vector
        maxiter (int): length of residual vectors, maximal iteration index
    """

    # Set up latex stuff and fonts
    rc('font', **{"sans-serif": ["Arial"], "size": 30})
    rc('legend', fontsize='small')
    rc('xtick', labelsize='small')
    rc('ytick', labelsize='small')

    # create plot and save
    fig, ax = plt.subplots(figsize=(15, 10))

    ax.set_xlabel('iteration')
    ax.set_ylabel('log10(residual)')

    plt.plot(np.linspace(1, maxiter, num=maxiter), np.log10(minres), 'ob--', label='min')
    plt.plot(np.linspace(1, maxiter, num=maxiter), np.log10(maxres), 'og--', label='max')
    plt.plot(np.linspace(1, maxiter, num=maxiter), np.log10(meanres), 'or--', label='mean')
    plt.plot(np.linspace(1, maxiter, num=maxiter), np.log10(medianres), 'oy--', label='median')
    plt.fill_between(np.linspace(1, maxiter, num=maxiter), np.log10(minres), np.log10(maxres), color='grey',
                     alpha=0.3, label='range')
    plt.axis([0, 14, -12, 3])
    plt.legend()

    plt.savefig(fname)

    assert os.path.isfile(fname), 'ERROR: plotting did not create PNG file'
