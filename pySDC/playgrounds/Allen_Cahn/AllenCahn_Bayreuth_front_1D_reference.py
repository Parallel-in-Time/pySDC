import timeit
import numpy as np

import pySDC.helpers.plot_helper as plt_helper
from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.AllenCahn_1D_FD import allencahn_front_fullyimplicit
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.sweeper_classes.explicit import explicit
from pySDC.playgrounds.Allen_Cahn.AllenCahn_monitor_Bayreuth import monitor


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
    level_params['dt'] = 1.0 / (16.0 * 128.0**2)
    level_params['nsweeps'] = 1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = [1]
    sweeper_params['Q1'] = ['LU']
    sweeper_params['Q2'] = ['LU']
    sweeper_params['QI'] = ['LU']
    sweeper_params['QE'] = ['EE']
    sweeper_params['initial_guess'] = 'zero'

    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['nvars'] = 127
    problem_params['dw'] = -0.04
    problem_params['eps'] = 0.04
    problem_params['newton_maxiter'] = 100
    problem_params['newton_tol'] = 1e-08
    problem_params['lin_tol'] = 1e-08
    problem_params['lin_maxiter'] = 100
    problem_params['radius'] = 0.25
    problem_params['interval'] = (-0.5, 0.5)

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = monitor

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = allencahn_front_fullyimplicit
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = explicit
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
        results and statistics of the run
    """

    # load (incomplete) default parameters
    description, controller_params = setup_parameters()

    # add stuff based on variant
    if variant == 'explicit':
        description['problem_class'] = allencahn_front_fullyimplicit
        description['sweeper_class'] = explicit
    else:
        raise NotImplemented('Wrong variant specified, got %s' % variant)

    # setup parameters "in time"
    t0 = 0
    Tend = 1.0 / (16.0 * 128.0**2)

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    # uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    wrapped = wrapper(controller.run, u0=uinit, t0=t0, Tend=Tend)
    print(timeit.timeit(wrapped, number=10000) / 10000.0)


def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)

    return wrapped


def main(cwd=''):
    """
    Main driver

    Args:
        cwd (str): current working directory (need this for testing)
    """

    # Loop over variants, exact and inexact solves
    run_SDC_variant(variant='explicit')


if __name__ == "__main__":
    main()
