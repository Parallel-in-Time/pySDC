import pickle
import os
import numpy as np

import cProfile, io, pstats

from pySDC.implementations.problem_classes.GeneralizedFisher_1D_PETSc import petsc_fisher_multiimplicit, \
    petsc_fisher_fullyimplicit, petsc_fisher_semiimplicit
from pySDC.implementations.datatype_classes.petsc_dmda_grid import petsc_data, rhs_2comp_petsc_data, rhs_imex_petsc_data
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.sweeper_classes.multi_implicit import multi_implicit
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.transfer_classes.TransferPETScDMDA import mesh_to_mesh_petsc_dmda
from pySDC.implementations.controller_classes.allinclusive_multigrid_nonMPI import allinclusive_multigrid_nonMPI

from pySDC.helpers.stats_helper import filter_stats, sort_stats
import pySDC.helpers.plot_helper as plt_helper


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
    level_params['restol'] = 1E-06
    level_params['dt'] = 0.25
    level_params['nsweeps'] = [1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]
    sweeper_params['Q1'] = ['LU']
    sweeper_params['Q2'] = ['LU']
    sweeper_params['QI'] = ['LU']
    sweeper_params['spread'] = False

    # initialize problem parameters
    problem_params = dict()
    problem_params['nu'] = 1
    problem_params['nvars'] = 2049
    problem_params['lambda0'] = 2.0
    problem_params['interval'] = (-50, 50)
    problem_params['nlsol_tol'] = 1E-10
    problem_params['nlsol_maxiter'] = 100
    problem_params['lsol_tol'] = 1E-10
    problem_params['lsol_maxiter'] = 100

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize space transfer parameters
    # space_transfer_params = dict()
    # space_transfer_params['finter'] = True

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = None  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['dtype_u'] = petsc_data  # pass data type for u
    description['dtype_f'] = None  # pass data type for f
    description['sweeper_class'] = None  # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    # description['space_transfer_class'] = mesh_to_mesh_petsc_dmda  # pass spatial transfer class
    # description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

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

    description, controller_params = setup_parameters()

    if variant == 'fully-implicit':
        description['problem_class'] = petsc_fisher_fullyimplicit
        description['dtype_f'] = petsc_data
        description['sweeper_class'] = generic_implicit
    elif variant == 'semi-implicit':
        description['problem_class'] = petsc_fisher_semiimplicit
        description['dtype_f'] = rhs_imex_petsc_data
        description['sweeper_class'] = imex_1st_order
    elif variant == 'multi-implicit':
        description['problem_class'] = petsc_fisher_multiimplicit
        description['dtype_f'] = rhs_2comp_petsc_data
        description['sweeper_class'] = multi_implicit
    else:
        raise NotImplemented('Wrong variant specified, got %s' % variant)

    if inexact:
        description['problem_params']['nlsol_maxiter'] = 1
        out = 'Working on inexact %s variant...' % variant
    else:
        out = 'Working on exact %s variant...' % variant
    print(out)

    # set time parameters
    t0 = 0.0
    Tend = 1.0

    # instantiate controller
    controller = allinclusive_multigrid_nonMPI(num_procs=1, controller_params=controller_params,
                                               description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # pr = cProfile.Profile()
    # pr.enable()
    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    # pr.disable()
    # s = io.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    # ps.print_stats()
    # print(s.getvalue())


    # compute exact solution and compare
    uex = P.u_exact(Tend)
    err = abs(uex - uend)

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

    print('Iteration count (nonlinear/linear): %i / %i' % (P.snes_itercount, P.ksp_itercount))
    print('Mean Iteration count per call: %4.2f / %4.2f' % (P.snes_itercount / max(P.snes_ncalls, 1),
                                                            P.ksp_itercount / max(P.ksp_ncalls, 1)))

    timing = sort_stats(filter_stats(stats, type='timing_run'), sortby='time')

    print('Time to solution: %6.4f sec.' % timing[0][1])
    print('Error vs. PDE solution: %6.4e' % err)
    print()

    assert err < 7E-05, 'ERROR: variant %s did not match error tolerance, got %s' % (variant, err)
    # exit()
    return timing[0][1], np.mean(niters)



def show_results(fname):
    """
    Plotting routine

    Args:
        fname: file name to read in and name plots
    """

    file = open(fname + '.pkl', 'rb')
    results = pickle.load(file)
    file.close()

    plt_helper.mpl.style.use('classic')
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

    results = {}
    for variant in ['fully-implicit', 'multi-implicit', 'semi-implicit']:

        # results[(variant, 'exact')] = run_SDC_variant(variant=variant, inexact=False)
        results[(variant, 'inexact')] = run_SDC_variant(variant=variant, inexact=True)

    fname = 'data/timings_SDC_variants_Fisher'
    file = open(fname + '.pkl', 'wb')
    pickle.dump(results, file)
    file.close()
    assert os.path.isfile(fname + '.pkl'), 'ERROR: pickle did not create file'

    show_results(fname)


if __name__ == "__main__":
    main()
