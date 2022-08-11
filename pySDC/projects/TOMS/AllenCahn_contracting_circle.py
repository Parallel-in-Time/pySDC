import os

import dill
import matplotlib.ticker as ticker
import numpy as np

import pySDC.helpers.plot_helper as plt_helper
from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.AllenCahn_2D_FD import (
    allencahn_fullyimplicit,
    allencahn_semiimplicit,
    allencahn_semiimplicit_v2,
    allencahn_multiimplicit,
    allencahn_multiimplicit_v2,
)
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.sweeper_classes.multi_implicit import multi_implicit
from pySDC.projects.TOMS.AllenCahn_monitor import monitor


# http://www.personal.psu.edu/qud2/Res/Pre/dz09sisc.pdf


def setup_parameters():
    """
    Helper routine to fill in all relevant parameters

    Note that this file will be used for all versions of SDC, containing more than necessary for each individual run

    Returns:
        description (dict)
        controller_params (dict)
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-08
    level_params['dt'] = 1e-03
    level_params['nsweeps'] = [1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]
    sweeper_params['Q1'] = ['LU']
    sweeper_params['Q2'] = ['LU']
    sweeper_params['QI'] = ['LU']
    sweeper_params['QE'] = ['EE']
    sweeper_params['initial_guess'] = 'zero'

    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['nu'] = 2
    problem_params['nvars'] = [(128, 128)]
    problem_params['eps'] = [0.04]
    problem_params['newton_maxiter'] = 100
    problem_params['newton_tol'] = 1e-09
    problem_params['lin_tol'] = 1e-10
    problem_params['lin_maxiter'] = 100
    problem_params['radius'] = 0.25

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = monitor

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = None  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = None  # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters

    return description, controller_params


def run_SDC_variant(variant=None, inexact=False):
    """
    Routine to run particular SDC variant

    Args:
        variant (str): string describing the variant
        inexact (bool): flag to use inexact nonlinear solve (or nor)

    Returns:
        timing (float)
        niter (float)
    """

    # load (incomplete) default parameters
    description, controller_params = setup_parameters()

    # add stuff based on variant
    if variant == 'fully-implicit':
        description['problem_class'] = allencahn_fullyimplicit
        description['sweeper_class'] = generic_implicit
        if inexact:
            description['problem_params']['newton_maxiter'] = 1
    elif variant == 'semi-implicit':
        description['problem_class'] = allencahn_semiimplicit
        description['sweeper_class'] = imex_1st_order
        if inexact:
            description['problem_params']['lin_maxiter'] = 10
    elif variant == 'semi-implicit_v2':
        description['problem_class'] = allencahn_semiimplicit_v2
        description['sweeper_class'] = imex_1st_order
        if inexact:
            description['problem_params']['newton_maxiter'] = 1
    elif variant == 'multi-implicit':
        description['problem_class'] = allencahn_multiimplicit
        description['sweeper_class'] = multi_implicit
        if inexact:
            description['problem_params']['newton_maxiter'] = 1
            description['problem_params']['lin_maxiter'] = 10
    elif variant == 'multi-implicit_v2':
        description['problem_class'] = allencahn_multiimplicit_v2
        description['sweeper_class'] = multi_implicit
        if inexact:
            description['problem_params']['newton_maxiter'] = 1
    else:
        raise NotImplementedError('Wrong variant specified, got %s' % variant)

    if inexact:
        out = 'Working on inexact %s variant...' % variant
    else:
        out = 'Working on exact %s variant...' % variant
    print(out)

    # setup parameters "in time"
    t0 = 0
    Tend = 0.032

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # filter statistics by variant (number of iterations)
    iter_counts = get_sorted(stats, type='niter', sortby='time')

    # compute and print statistics
    niters = np.array([item[1] for item in iter_counts])
    out = '   Mean number of iterations: %4.2f' % np.mean(niters)
    print(out)
    out = '   Range of values for number of iterations: %2i ' % np.ptp(niters)
    print(out)
    out = '   Position of max/min number of iterations: %2i -- %2i' % (int(np.argmax(niters)), int(np.argmin(niters)))
    print(out)
    out = '   Std and var for number of iterations: %4.2f -- %4.2f' % (float(np.std(niters)), float(np.var(niters)))
    print(out)

    print('   Iteration count (nonlinear/linear): %i / %i' % (P.newton_itercount, P.lin_itercount))
    print(
        '   Mean Iteration count per call: %4.2f / %4.2f'
        % (P.newton_itercount / max(P.newton_ncalls, 1), P.lin_itercount / max(P.lin_ncalls, 1))
    )

    timing = get_sorted(stats, type='timing_run', sortby='time')

    print('Time to solution: %6.4f sec.' % timing[0][1])
    print()

    return stats


def show_results(fname, cwd=''):
    """
    Plotting routine

    Args:
        fname (str): file name to read in and name plots
        cwd (str): current working directory
    """

    file = open(cwd + fname + '.pkl', 'rb')
    results = dill.load(file)
    file.close()

    # plt_helper.mpl.style.use('classic')
    plt_helper.setup_mpl()

    # set up plot for timings
    fig, ax1 = plt_helper.newfig(textwidth=238.96, scale=1.5, ratio=0.4)

    timings = {}
    niters = {}
    for key, item in results.items():
        timings[key] = get_sorted(item, type='timing_run', sortby='time')[0][1]
        iter_counts = get_sorted(item, type='niter', sortby='time')
        niters[key] = np.mean(np.array([item[1] for item in iter_counts]))

    xcoords = list(range(len(timings)))
    sorted_timings = sorted([(key, timings[key]) for key in timings], reverse=True, key=lambda tup: tup[1])
    sorted_niters = [(k, niters[k]) for k in [key[0] for key in sorted_timings]]
    heights_timings = [item[1] for item in sorted_timings]
    heights_niters = [item[1] for item in sorted_niters]
    keys = [(item[0][1] + ' ' + item[0][0]).replace('-', '\n').replace('_v2', ' mod.') for item in sorted_timings]

    ax1.bar(xcoords, heights_timings, align='edge', width=-0.3, label='timings (left axis)')
    ax1.set_ylabel('time (sec)')

    ax2 = ax1.twinx()
    ax2.bar(xcoords, heights_niters, color='lightcoral', align='edge', width=0.3, label='iterations (right axis)')
    ax2.set_ylabel('mean number of iterations')

    ax1.set_xticks(xcoords)
    ax1.set_xticklabels(keys, rotation=90, ha='center')

    # ask matplotlib for the plotted objects and their labels
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)

    # save plot, beautify
    f = fname + '_timings'
    plt_helper.savefig(f)

    assert os.path.isfile(f + '.pdf'), 'ERROR: plotting did not create PDF file'
    # assert os.path.isfile(f + '.pgf'), 'ERROR: plotting did not create PGF file'
    assert os.path.isfile(f + '.png'), 'ERROR: plotting did not create PNG file'

    # set up plot for radii
    fig, ax = plt_helper.newfig(textwidth=238.96, scale=1.0)

    exact_radii = []
    for key, item in results.items():
        computed_radii = get_sorted(item, type='computed_radius', sortby='time')

        xcoords = [item0[0] for item0 in computed_radii]
        radii = [item0[1] for item0 in computed_radii]
        if key[0] + ' ' + key[1] == 'fully-implicit exact':
            ax.plot(xcoords, radii, label=(key[0] + ' ' + key[1]).replace('_v2', ' mod.'))

        exact_radii = get_sorted(item, type='exact_radius', sortby='time')

        diff = np.array([abs(item0[1] - item1[1]) for item0, item1 in zip(exact_radii, computed_radii)])
        max_pos = int(np.argmax(diff))
        assert max(diff) < 0.07, 'ERROR: computed radius is too far away from exact radius, got %s' % max(diff)
        assert 0.028 < computed_radii[max_pos][0] < 0.03, (
            'ERROR: largest difference is at wrong time, got %s' % computed_radii[max_pos][0]
        )

    xcoords = [item[0] for item in exact_radii]
    radii = [item[1] for item in exact_radii]
    ax.plot(xcoords, radii, color='k', linestyle='--', linewidth=1, label='exact')

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.2f'))
    ax.set_ylabel('radius')
    ax.set_xlabel('time')
    ax.grid()
    ax.legend(loc=3)

    # save plot, beautify
    f = fname + '_radii'
    plt_helper.savefig(f)

    assert os.path.isfile(f + '.pdf'), 'ERROR: plotting did not create PDF file'
    # assert os.path.isfile(f + '.pgf'), 'ERROR: plotting did not create PGF file'
    assert os.path.isfile(f + '.png'), 'ERROR: plotting did not create PNG file'

    # set up plot for interface width
    fig, ax = plt_helper.newfig(textwidth=238.96, scale=1.0)

    interface_width = []
    for key, item in results.items():
        interface_width = get_sorted(item, type='interface_width', sortby='time')
        xcoords = [item[0] for item in interface_width]
        width = [item[1] for item in interface_width]
        if key[0] + ' ' + key[1] == 'fully-implicit exact':
            ax.plot(xcoords, width, label=key[0] + ' ' + key[1])

    xcoords = [item[0] for item in interface_width]
    init_width = [interface_width[0][1]] * len(xcoords)
    ax.plot(xcoords, init_width, color='k', linestyle='--', linewidth=1, label='exact')

    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%1.2f'))
    ax.set_ylabel(r'interface width ($\epsilon$)')
    ax.set_xlabel('time')
    ax.grid()
    ax.legend(loc=3)

    # save plot, beautify
    f = fname + '_interface'
    plt_helper.savefig(f)

    assert os.path.isfile(f + '.pdf'), 'ERROR: plotting did not create PDF file'
    # assert os.path.isfile(f + '.pgf'), 'ERROR: plotting did not create PGF file'
    assert os.path.isfile(f + '.png'), 'ERROR: plotting did not create PNG file'

    return None


def main(cwd=''):
    """
    Main driver

    Args:
        cwd (str): current working directory (need this for testing)
    """

    # Loop over variants, exact and inexact solves
    results = {}
    for variant in ['multi-implicit', 'semi-implicit', 'fully-implicit', 'semi-implicit_v2', 'multi-implicit_v2']:

        results[(variant, 'exact')] = run_SDC_variant(variant=variant, inexact=False)
        results[(variant, 'inexact')] = run_SDC_variant(variant=variant, inexact=True)

    # dump result
    fname = 'data/results_SDC_variants_AllenCahn_1E-03'
    file = open(cwd + fname + '.pkl', 'wb')
    dill.dump(results, file)
    file.close()
    assert os.path.isfile(cwd + fname + '.pkl'), 'ERROR: dill did not create file'

    # visualize
    show_results(fname, cwd=cwd)


if __name__ == "__main__":
    main()
