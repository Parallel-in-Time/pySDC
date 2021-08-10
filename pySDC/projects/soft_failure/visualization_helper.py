import os

import matplotlib
import numpy as np
from matplotlib import rc

from pySDC.helpers.stats_helper import filter_stats

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
    for k, _ in extract_stats.items():
        maxiter = max(maxiter, k.iter)

    # grep residuals and put into array
    residual = np.zeros(maxiter)
    residual[:] = -99
    for k, v in extract_stats.items():
        if k.iter is not -1:
            residual[k.iter - 1] = np.log10(v)

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

    assert os.path.isfile(fname), 'ERROR: plotting did not create PNG file'


def show_min_max_residual_across_simulation(fname, minres, maxres, meanres, medianres, maxiter):
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


def show_iter_hist(fname, maxiter, nruns):
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

    assert os.path.isfile(fname), 'ERROR: plotting did not create PNG file'
