import os
import numpy as np

from pySDC.implementations.datatype_classes.mesh import mesh, rhs_imex_mesh
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.AllenCahn_2D_FD import allencahn_fullyimplicit
from pySDC.implementations.problem_classes.AllenCahn_2D_FFT import allencahn2d_imex

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.playgrounds.Allen_Cahn.AllenCahn_monitor import monitor


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
    level_params['restol'] = 1E-11
    level_params['dt'] = 1E-05
    level_params['nsweeps'] = [1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [5]
    sweeper_params['QI'] = ['LU']
    sweeper_params['spread'] = False

    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['nu'] = 2
    problem_params['nvars'] = [(128, 128)]
    problem_params['eps'] = [0.04]
    problem_params['newton_maxiter'] = 100
    problem_params['newton_tol'] = 1E-12
    problem_params['lin_tol'] = 1E-12
    problem_params['lin_maxiter'] = 100
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
    description['problem_class'] = allencahn_fullyimplicit  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = generic_implicit  # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters

    return description, controller_params


def setup_parameters_FFT():
    """
    Helper routine to fill in all relevant parameters

    Note that this file will be used for all versions of SDC, containing more than necessary for each individual run

    Returns:
        description (dict)
        controller_params (dict)
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-11
    level_params['dt'] = 1E-04
    level_params['nsweeps'] = [1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [5]
    sweeper_params['QI'] = ['LU']
    sweeper_params['spread'] = False

    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['nu'] = 2
    problem_params['nvars'] = [(128, 128)]
    problem_params['eps'] = [0.04]
    problem_params['newton_maxiter'] = 100
    problem_params['newton_tol'] = 1E-12
    problem_params['lin_tol'] = 1E-12
    problem_params['lin_maxiter'] = 100
    problem_params['radius'] = 0.25

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = monitor

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = allencahn2d_imex  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters

    return description, controller_params


def run_reference(Tend):
    """
    Routine to run particular SDC variant

    Args:
        Tend (float): end time for dumping
    """

    # load (incomplete) default parameters
    description, controller_params = setup_parameters_FFT()

    # setup parameters "in time"
    t0 = 0

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params,
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

    timing = sort_stats(filter_stats(stats, type='timing_run'), sortby='time')

    print('Time to solution: %6.4f sec.' % timing[0][1])
    print()

    computed_radii_tmp = sort_stats(filter_stats(stats, type='computed_radius'), sortby='time')
    computed_radii = np.array([item0[1] for item0 in computed_radii_tmp])
    print(len(computed_radii_tmp), len(computed_radii))

    fname = 'data/AC_reference_FFT_Tend{:.1e}'.format(Tend)
    np.savez_compressed(file=fname, uend=uend.values, radius=computed_radii)


def main(cwd=''):
    """
    Main driver

    Args:
        cwd (str): current working directory (need this for testing)
    """
    Tend = 0.032
    run_reference(Tend=Tend)

    fname = cwd + 'data/AC_reference_FFT_Tend{:.1e}'.format(Tend) + '.npz'
    assert os.path.isfile(fname), 'ERROR: numpy did not create file'

    loaded = np.load(fname)
    uend = loaded['uend']



if __name__ == "__main__":
    main()
