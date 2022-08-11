import sys

import numpy as np
from mpi4py import MPI

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
from pySDC.implementations.problem_classes.HeatEquation_2D_PETSc_forced import heat2d_petsc_forced
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.transfer_classes.TransferPETScDMDA import mesh_to_mesh_petsc_dmda


def main():
    """
    Program to demonstrate usage of PETSc data structures and spatial parallelization,
    combined with parallelization in time.
    """
    # set MPI communicator
    comm = MPI.COMM_WORLD

    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    # split world communicator to create space-communicators
    if len(sys.argv) >= 2:
        color = int(world_rank / int(sys.argv[1]))
    else:
        color = int(world_rank / 1)
    space_comm = comm.Split(color=color)
    space_size = space_comm.Get_size()
    space_rank = space_comm.Get_rank()

    # split world communicator to create time-communicators
    if len(sys.argv) >= 2:
        color = int(world_rank % int(sys.argv[1]))
    else:
        color = int(world_rank / world_size)
    time_comm = comm.Split(color=color)
    time_size = time_comm.Get_size()
    time_rank = time_comm.Get_rank()

    print(
        "IDs (world, space, time):  %i / %i -- %i / %i -- %i / %i"
        % (world_rank, world_size, space_rank, space_size, time_rank, time_size)
    )

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-08
    level_params['dt'] = 0.125
    level_params['nsweeps'] = [3, 1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [5]
    sweeper_params['QI'] = ['LU']  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['initial_guess'] = 'zero'

    # initialize problem parameters
    problem_params = dict()
    problem_params['nu'] = 1.0  # diffusion coefficient
    problem_params['freq'] = 2  # frequency for the test value
    problem_params['cnvars'] = [(129, 129)]  # number of degrees of freedom on coarse level
    problem_params['refine'] = [1, 0]  # number of refinements
    problem_params['comm'] = space_comm  # pass space-communicator to problem class
    problem_params['sol_tol'] = 1e-10  # set tolerance to PETSc' linear solver

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 2
    space_transfer_params['periodic'] = False

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20 if space_rank == 0 else 99  # set level depending on rank
    controller_params['dump_setup'] = False

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = heat2d_petsc_forced  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh_petsc_dmda  # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

    # set time parameters
    t0 = 0.0
    Tend = 3.0

    # instantiate controller
    controller = controller_MPI(controller_params=controller_params, description=description, comm=time_comm)

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

    niters = np.array([item[1] for item in iter_counts])

    # limit output to space-rank 0 (as before when setting the logger level)
    if space_rank == 0:

        out = 'This is time-rank %i...' % time_rank
        print(out)

        # compute and print statistics
        for item in iter_counts:
            out = 'Number of iterations for time %4.2f: %2i' % item
            print(out)

        out = '   Mean number of iterations: %4.2f' % np.mean(niters)
        print(out)
        out = '   Range of values for number of iterations: %2i ' % np.ptp(niters)
        print(out)
        out = '   Position of max/min number of iterations: %2i -- %2i' % (
            int(np.argmax(niters)),
            int(np.argmin(niters)),
        )
        print(out)
        out = '   Std and var for number of iterations: %4.2f -- %4.2f' % (float(np.std(niters)), float(np.var(niters)))
        print(out)

        print('   Iteration count linear solver: %i' % P.ksp_itercount)
        print('   Mean Iteration count per call: %4.2f' % (P.ksp_itercount / max(P.ksp_ncalls, 1)))

        timing = get_sorted(stats, type='timing_run', sortby='time')

        out = 'Time to solution: %6.4f sec.' % timing[0][1]
        print(out)
        out = 'Error vs. PDE solution: %6.4e' % err
        print(out)


if __name__ == "__main__":
    main()
