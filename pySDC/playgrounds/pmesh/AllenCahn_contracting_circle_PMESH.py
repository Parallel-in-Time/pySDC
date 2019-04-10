import os
import matplotlib
matplotlib.use('TkAgg')
import dill
import matplotlib.ticker as ticker
import numpy as np

# import pySDC.helpers.plot_helper as plt_helper
from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.projects.TOMS.AllenCahn_monitor import monitor

from pySDC.playgrounds.pmesh.AllenCahn_PMESH import allencahn_imex
from pySDC.playgrounds.pmesh.TransferMesh_PMESH_new import pmesh_to_pmesh

import matplotlib.pyplot as plt


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
    level_params['restol'] = 1E-08
    level_params['dt'] = 1E-03
    level_params['nsweeps'] = [1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]
    sweeper_params['QI'] = ['LU']
    sweeper_params['QE'] = ['EE']
    sweeper_params['spread'] = False

    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['nu'] = 2
    problem_params['L'] = 1.0
    problem_params['nvars'] = [(256, 256), (64, 64)]
    problem_params['eps'] = [0.04]
    problem_params['radius'] = 0.25

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    # controller_params['hook_class'] = monitor

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = None  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = None  # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = pmesh_to_pmesh

    return description, controller_params


def run_SDC_variant(variant=None):
    """
    Routine to run particular SDC variant

    Args:
        variant (str): string describing the variant

    Returns:
        timing (float)
        niter (float)
    """

    # load (incomplete) default parameters
    description, controller_params = setup_parameters()

    # add stuff based on variant
    if variant == 'semi-implicit':
        description['problem_class'] = allencahn_imex
        description['sweeper_class'] = imex_1st_order
    # elif variant == 'semi-implicit-stab':
    #     description['problem_class'] = allencahn2d_imex_stab
    #     description['sweeper_class'] = imex_1st_order
    else:
        raise NotImplemented('Wrong variant specified, got %s' % variant)

    # setup parameters "in time"
    t0 = 0
    Tend = 0.032
    # Tend = 0.01

    # instantiate controller
    controller = controller_nonMPI(num_procs=8, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # plt.figure(1)
    # plt.imshow(uinit.values)
    # plt.colorbar()
    # plt.show()
    # exit()

    # plt_helper.plt.imshow(uinit.values)
    # plt_helper.plt.show()

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # plt_helper.plt.imshow(uend.values)
    # plt_helper.plt.show()

    # filter statistics by variant (number of iterations)
    filtered_stats = filter_stats(stats, type='niter')

    # convert filtered statistics to list of iterations count, sorted by process
    iter_counts = sort_stats(filtered_stats, sortby='time')

    # compute and print statistics
    niters = np.array([item[1] for item in iter_counts])
    out = '   Mean number of iterations: %4.2f' % np.mean(niters)
    print(out)
    out = '   Range of values for number of iterations: %2i ' % np.ptp(niters)
    print(out)
    out = '   Position of max/min number of iterations: %2i -- %2i' % \
          (int(np.argmax(niters)), int(np.argmin(niters)))
    print(out)
    out = '   Std and var for number of iterations: %4.2f -- %4.2f' % (float(np.std(niters)), float(np.var(niters)))
    print(out)

    timing = sort_stats(filter_stats(stats, type='timing_run'), sortby='time')

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
        timings[key] = sort_stats(filter_stats(item, type='timing_run'), sortby='time')[0][1]
        iter_counts = sort_stats(filter_stats(item, type='niter'), sortby='time')
        niters[key] = np.mean(np.array([item[1] for item in iter_counts]))

    xcoords = [i for i in range(len(timings))]
    sorted_timings = sorted([(key, timings[key]) for key in timings], reverse=True, key=lambda tup: tup[1])
    sorted_niters = [(k, niters[k]) for k in [key[0] for key in sorted_timings]]
    heights_timings = [item[1] for item in sorted_timings]
    heights_niters = [item[1] for item in sorted_niters]
    keys = [(item[0][1] + ' ' + item[0][0]).replace('-', '\n').replace('_v2', ' mod.') for item in sorted_timings]

    ax1.bar(xcoords, heights_timings, align='edge', width=-0.3, label='timings (left axis)')
    ax1.set_ylabel('time (sec)')

    ax2 = ax1.twinx()
    ax2.bar(xcoords, heights_niters, color='r', align='edge', width=0.3, label='iterations (right axis)')
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
    assert os.path.isfile(f + '.pgf'), 'ERROR: plotting did not create PGF file'
    assert os.path.isfile(f + '.png'), 'ERROR: plotting did not create PNG file'

    # set up plot for radii
    fig, ax = plt_helper.newfig(textwidth=238.96, scale=1.0)

    exact_radii = []
    for key, item in results.items():
        computed_radii = sort_stats(filter_stats(item, type='computed_radius'), sortby='time')

        xcoords = [item0[0] for item0 in computed_radii]
        radii = [item0[1] for item0 in computed_radii]
        if key[0] + ' ' + key[1] == 'semi-implicit-stab exact':
            ax.plot(xcoords, radii, label=(key[0] + ' ' + key[1]).replace('_v2', ' mod.'))

        exact_radii = sort_stats(filter_stats(item, type='exact_radius'), sortby='time')

        # diff = np.array([abs(item0[1] - item1[1]) for item0, item1 in zip(exact_radii, computed_radii)])
        # max_pos = int(np.argmax(diff))
        # assert max(diff) < 0.07, 'ERROR: computed radius is too far away from exact radius, got %s' % max(diff)
        # assert 0.028 < computed_radii[max_pos][0] < 0.03, \
        #     'ERROR: largest difference is at wrong time, got %s' % computed_radii[max_pos][0]

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
    assert os.path.isfile(f + '.pgf'), 'ERROR: plotting did not create PGF file'
    assert os.path.isfile(f + '.png'), 'ERROR: plotting did not create PNG file'

    # set up plot for interface width
    fig, ax = plt_helper.newfig(textwidth=238.96, scale=1.0)

    interface_width = []
    for key, item in results.items():
        interface_width = sort_stats(filter_stats(item, type='interface_width'), sortby='time')
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
    assert os.path.isfile(f + '.pgf'), 'ERROR: plotting did not create PGF file'
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
    for variant in ['semi-implicit']:

        results[(variant, 'exact')] = run_SDC_variant(variant=variant)

    # dump result
    fname = 'data/results_SDC_variants_AllenCahn_1E-03'
    file = open(cwd + fname + '.pkl', 'wb')
    dill.dump(results, file)
    file.close()
    assert os.path.isfile(cwd + fname + '.pkl'), 'ERROR: dill did not create file'

    # visualize
    # show_results(fname, cwd=cwd)


if __name__ == "__main__":
    main()
