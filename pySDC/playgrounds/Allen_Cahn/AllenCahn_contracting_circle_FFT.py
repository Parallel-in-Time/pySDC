import sys

import numpy as np

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.AllenCahn_2D_FFT import allencahn2d_imex
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.transfer_classes.TransferMesh_FFT2D import mesh_to_mesh_fft2d
from pySDC.playgrounds.Allen_Cahn.AllenCahn_output import output


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
    level_params['restol'] = 1E-07
    level_params['dt'] = 1E-03
    level_params['nsweeps'] = None

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
    problem_params['nvars'] = None
    problem_params['eps'] = 0.04
    problem_params['radius'] = 0.25
    problem_params['init_type'] = 'circle'

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = output

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = allencahn2d_imex  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh_fft2d

    return description, controller_params


def run_variant(nlevels=None):
    """
    Routine to run particular SDC variant

    Args:

    Returns:

    """

    # load (incomplete) default parameters
    description, controller_params = setup_parameters()

    # add stuff based on variant
    if nlevels == 1:
        description['level_params']['nsweeps'] = 1
        description['problem_params']['nvars'] = [(128, 128)]
        # description['problem_params']['nvars'] = [(32, 32)]
    elif nlevels == 2:
        description['level_params']['nsweeps'] = [1, 1]
        description['problem_params']['nvars'] = [(128, 128), (32, 32)]
        # description['problem_params']['nvars'] = [(32, 32), (16, 16)]
    else:
        raise NotImplemented('Wrong variant specified, got %s' % nlevels)

    out = 'Working on %s levels...' % nlevels
    print(out)

    # setup parameters "in time"
    t0 = 0.0
    Tend = 0.032

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

    fname = 'data/AC_reference_FFT_Tend{:.1e}'.format(Tend) + '.npz'
    loaded = np.load(fname)
    uref = loaded['uend']

    err = np.linalg.norm(uref - uend.values, np.inf)
    print('Error vs. reference solution: %6.4e' % err)
    print()

    return stats


def main(cwd=''):
    """
    Main driver

    Args:
        cwd (str): current working directory (need this for testing)
    """

    if len(sys.argv) == 2:
        nlevels = int(sys.argv[1])
    else:
        nlevels = 1
        # raise NotImplementedError('Need input of nsweeps, got % s' % sys.argv)

    _ = run_variant(nlevels=nlevels)


if __name__ == "__main__":
    main()
