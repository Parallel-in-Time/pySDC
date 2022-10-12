import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI

from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order

from pySDC.playgrounds.Dedalus.TransferDedalusFields import dedalus_field_transfer
from pySDC.playgrounds.Dedalus.HeatEquation_1D_Dedalus_forced import heat1d_dedalus_forced


def main():
    """
    A simple test program to do PFASST runs for the heat equation
    """

    # set MPI communicator
    comm = MPI.COMM_WORLD

    world_rank = comm.Get_rank()
    # split world communicator to create space-communicators
    color = int(world_rank / 1)
    space_comm = comm.Split(color=color)
    space_size = space_comm.Get_size()
    space_rank = space_comm.Get_rank()

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-08
    level_params['dt'] = 1.0 / 4
    level_params['nsweeps'] = [1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = [3]
    sweeper_params['QI'] = ['LU']  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    # sweeper_params['spread'] = False

    # initialize problem parameters
    problem_params = dict()
    problem_params['nu'] = 0.1  # diffusion coefficient
    problem_params['freq'] = 2  # frequency for the test value
    problem_params['nvars'] = [16, 4]  # number of degrees of freedom for each level
    problem_params['comm'] = space_comm

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 10

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    # controller_params['hook_class'] = error_output

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = heat1d_dedalus_forced
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = dedalus_field_transfer
    # description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

    # set time parameters
    t0 = 0.0
    Tend = 1.0

    # instantiate controller
    controller = controller_MPI(controller_params=controller_params, description=description, comm=comm)

    # get initial values on finest level
    P = controller.S.levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # compute exact solution and compare
    uex = P.u_exact(Tend)
    err = abs(uex - uend)

    # filter statistics by type (number of iterations)
    iter_counts = get_sorted(stats, type='niter', sortby='time')

    # compute and print statistics
    for item in iter_counts:
        out = 'Number of iterations for time %4.2f: %2i' % item
        print(out)

    # niters = np.array([item[1] for item in iter_counts])
    # out = '   Mean number of iterations: %4.2f' % np.mean(niters)
    # print(out)
    # # out = '   Range of values for number of iterations: %2i ' % np.ptp(niters)
    # # print(out)
    # out = '   Position of max/min number of iterations: %2i -- %2i' % \
    #       (int(np.argmax(niters)), int(np.argmin(niters)))
    # print(out)
    # out = '   Std and var for number of iterations: %4.2f -- %4.2f' % (float(np.std(niters)), float(np.var(niters)))
    # print(out)

    print('CFL number: %4.2f' % (level_params['dt'] * problem_params['nu'] / (1.0 / problem_params['nvars'][0]) ** 2))
    print('Error: %8.4e' % err)


if __name__ == "__main__":
    main()
