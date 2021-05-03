from mpi4py import MPI

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.problem_classes.HeatEquation_1D_FD import heat1d
from pySDC.implementations.sweeper_classes.generic_LU import generic_implicit
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh


def set_parameters_ml():
    """
    Helper routine to set parameters for the following multi-level runs

    Returns:
        dict: dictionary containing the simulation parameters
        dict: dictionary containing the controller parameters
        float: starting time
        float: end time
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 5E-10
    level_params['dt'] = 0.125

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['QI'] = 'LU'
    sweeper_params['num_nodes'] = [3]

    # initialize problem parameters
    problem_params = dict()
    problem_params['nu'] = 0.1  # diffusion coefficient
    problem_params['freq'] = 2  # frequency for the test value
    problem_params['nvars'] = [63, 31]  # number of degrees of freedom for each level

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50
    step_params['errtol'] = 1E-05

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 6

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['all_to_done'] = True  # can ask the controller to keep iterating all steps until the end
    controller_params['use_iteration_estimator'] = False  # activate iteration estimator

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = heat1d  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = generic_implicit  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

    # set time parameters
    t0 = 0.0
    Tend = 1.0

    return description, controller_params, t0, Tend


if __name__ == "__main__":
    """
    A simple test program to do MPI-parallel PFASST runs
    """

    # set MPI communicator
    comm = MPI.COMM_WORLD

    # get parameters from Part A
    description, controller_params, t0, Tend = set_parameters_ml()

    # instantiate controllers
    controller = controller_MPI(controller_params=controller_params, description=description, comm=comm)
    # get initial values on finest level
    P = controller.S.levels[0].prob
    uinit = P.u_exact(t0)

    # call main functions to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # filter statistics by type (number of iterations)
    filtered_stats = filter_stats(stats, type='niter')

    # convert filtered statistics to list of iterations count, sorted by process
    iter_counts = sort_stats(filtered_stats, sortby='time')

    # combine statistics into list of statistics
    iter_counts_list = comm.gather(iter_counts, root=0)

    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:

        out = 'Working with %2i processes...' % size
        print(out)

        # compute exact solutions and compare with both results
        uex = P.u_exact(Tend)
        err = abs(uex - uend)

        out = 'Error vs. exact solution: %12.8e' % err
        print(out)

        # build one list of statistics instead of list of lists, the sort by time
        iter_counts_gather = [item for sublist in iter_counts_list for item in sublist]
        iter_counts = sorted(iter_counts_gather, key=lambda tup: tup[0])

        # compute and print statistics
        for item in iter_counts:
            out = 'Number of iterations for time %4.2f: %1i ' % (item[0], item[1])
            print(out)
