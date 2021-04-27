import numpy as np
import sys
import matplotlib.pyplot as plt
from mpi4py import MPI

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order

from pySDC.playgrounds.Dedalus.TransferDedalusFields import dedalus_field_transfer
# from pySDC.playgrounds.Dedalus.Dynamo_2D_Dedalus import dynamo_2d_dedalus
from pySDC.playgrounds.Dedalus.Dynamo_2D_Dedalus_NEW import dynamo_2d_dedalus
from pySDC.playgrounds.Dedalus.Dynamo_monitor import monitor


def main():
    """
    A simple test program to do PFASST runs for the heat equation
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

    print("IDs (world, space, time):  %i / %i -- %i / %i -- %i / %i" % (world_rank, world_size, space_rank, space_size,
                                                                        time_rank, time_size))

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-08
    level_params['dt'] = 0.5
    level_params['nsweeps'] = [1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]
    sweeper_params['QI'] = ['LU']  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['initial_guess'] = 'zero'

    # initialize problem parameters
    problem_params = dict()
    problem_params['Rm'] = 4
    problem_params['kz'] = 0.45
    problem_params['initial'] = 'low-res'
    problem_params['nvars'] = [(64, 64)]  # number of degrees of freedom for each level
    problem_params['comm'] = space_comm

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50
    # step_params['errtol'] = 1E-07

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20 if space_rank == 0 else 99
    controller_params['hook_class'] = monitor
    # controller_params['use_iteration_estimator'] = True

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = dynamo_2d_dedalus
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = dedalus_field_transfer
    # description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

    # set time parameters
    t0 = 0.0
    Tend = 10.0

    # instantiate controller
    controller = controller_MPI(controller_params=controller_params, description=description, comm=time_comm)

    # get initial values on finest level
    P = controller.S.levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    timings = sort_stats(filter_stats(stats, type='timing_run'), sortby='time')[0][1]
    print(f'Time it took to run the simulation: {timings:6.3f} seconds')

    if space_size == 1:
        bx_maxes = sort_stats(filter_stats(stats, type='bx_max'), sortby='time')

        times = [t0 + i * level_params['dt'] for i in range(int((Tend - t0) / level_params['dt']) + 1)]
        half = int(len(times) / 2)
        gr = np.polyfit(times[half::], np.log([item[1] for item in bx_maxes])[half::], 1)[0]
        print("Growth rate: {:.3e}".format(gr))

        plt.figure(3)
        plt.semilogy(times, [item[1] for item in bx_maxes])
        plt.pause(0.1)

if __name__ == "__main__":
    main()
