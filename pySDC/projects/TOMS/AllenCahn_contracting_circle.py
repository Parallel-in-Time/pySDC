import pySDC.helpers.plot_helper as plt_helper

import dill
import os
import numpy as np

from pySDC.implementations.datatype_classes.mesh import mesh, rhs_imex_mesh, rhs_comp2_mesh
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.sweeper_classes.multi_implicit import multi_implicit
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.allinclusive_multigrid_nonMPI import allinclusive_multigrid_nonMPI
from pySDC.implementations.problem_classes.AllenCahn_2D_FD import allencahn_fullyimplicit, allencahn_semiimplicit, \
    allencahn_semiimplicit_v2, allencahn_multiimplicit, allencahn_multiimplicit_v2

from pySDC.helpers.stats_helper import filter_stats, sort_stats
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
    level_params['restol'] = 1E-08
    level_params['dt'] = 1E-03
    level_params['nsweeps'] = [1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]
    sweeper_params['Q1'] = ['LU']
    sweeper_params['Q2'] = ['LU']
    sweeper_params['QI'] = ['LU']
    sweeper_params['spread'] = False

    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['nu'] = 2
    problem_params['nvars'] = [(128, 128)]
    problem_params['eps'] = [0.0390625]
    problem_params['newton_maxiter'] = 100
    problem_params['newton_tol'] = 1E-09
    problem_params['lin_tol'] = 1E-10
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
    description['dtype_u'] = mesh  # pass data type for u
    description['dtype_f'] = None  # pass data type for f
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
        description['dtype_f'] = mesh
        description['sweeper_class'] = generic_implicit
    elif variant == 'semi-implicit':
        description['problem_class'] = allencahn_semiimplicit
        description['dtype_f'] = rhs_imex_mesh
        description['sweeper_class'] = imex_1st_order
    elif variant == 'multi-implicit':
        description['problem_class'] = allencahn_multiimplicit
        description['dtype_f'] = rhs_comp2_mesh
        description['sweeper_class'] = multi_implicit
    else:
        raise NotImplemented('Wrong variant specified, got %s' % variant)

    if inexact:
        description['problem_params']['newton_maxiter'] = 1
        out = 'Working on inexact %s variant...' % variant
    else:
        out = 'Working on exact %s variant...' % variant
    print(out)

    # setup parameters "in time"
    t0 = 0
    Tend = 0.001

    # instantiate controller
    controller = allinclusive_multigrid_nonMPI(num_procs=1, controller_params=controller_params,
                                               description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

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

    print('Iteration count (nonlinear/linear): %i / %i' % (P.newton_itercount, P.lin_itercount))
    print('Mean Iteration count per call: %4.2f / %4.2f' % (P.newton_itercount / max(P.newton_ncalls, 1),
                                                            P.lin_itercount / max(P.lin_ncalls, 1)))

    timing = sort_stats(filter_stats(stats, type='timing_run'), sortby='time')

    print('Time to solution: %6.4f sec.' % timing[0][1])
    print()

    # assert np.mean(niters) <= 23, 'ERROR: number of iterations is too high, got %s' % np.mean(niters)

    return timing[0][1], np.mean(niters)


def show_results(fname):
    """
    Plotting routine

    Args:
        fname: file name to read in and name plots
    """

    file = open(fname + '.pkl', 'rb')
    results = dill.load(file)
    file.close()

    # plt_helper.mpl.style.use('classic')
    plt_helper.setup_mpl()

    plt_helper.newfig(textwidth=238.96, scale=1.0)

    xcoords = [i for i in range(len(results))]
    sorted_data = sorted([(key, results[key][0]) for key in results], reverse=True, key=lambda tup: tup[1])
    heights = [item[1] for item in sorted_data]
    keys = [(item[0][1] + ' ' + item[0][0]).replace('-', '\n') for item in sorted_data]

    plt_helper.plt.bar(xcoords, heights, align='center')

    plt_helper.plt.xticks(xcoords, keys, rotation=90)
    plt_helper.plt.ylabel('time (sec)')

    # save plot, beautify
    plt_helper.savefig(fname)

    assert os.path.isfile(fname + '.pdf'), 'ERROR: plotting did not create PDF file'
    assert os.path.isfile(fname + '.pgf'), 'ERROR: plotting did not create PGF file'
    assert os.path.isfile(fname + '.png'), 'ERROR: plotting did not create PNG file'

    return None


def main():
    """
    Main driver

    Args:
        cwd (str): current working directory (need this for testing)
    """

    # Loop over variants, exact and inexact solves
    results = {}
    for variant in ['multi-implicit', 'semi-implicit', 'fully-implicit']:

        results[(variant, 'exact')] = run_SDC_variant(variant=variant, inexact=False)
        results[(variant, 'inexact')] = run_SDC_variant(variant=variant, inexact=True)

    # dump result
    fname = 'data/timings_SDC_variants_AllenCahn'
    file = open(fname + '.pkl', 'wb')
    dill.dump(results, file)
    file.close()
    assert os.path.isfile(fname + '.pkl'), 'ERROR: dill did not create file'

    # visualize
    show_results(fname)


if __name__ == "__main__":
    main()
