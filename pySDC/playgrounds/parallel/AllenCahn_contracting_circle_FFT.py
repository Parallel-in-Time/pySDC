import numpy as np
import sys

from mpi4py import MPI

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
from pySDC.implementations.problem_classes.AllenCahn_2D_FFT import allencahn2d_imex, allencahn2d_imex_stab
# from pySDC.implementations.problem_classes.AllenCahn_2D_parFFT import allencahn2d_imex, allencahn2d_imex_stab
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.transfer_classes.TransferMesh_FFT2D import mesh_to_mesh_fft2d
from pySDC.playgrounds.parallel.AllenCahn_parallel_monitor import monitor

import pySDC.helpers.plot_helper as plt_helper


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
    level_params['dt'] = 1E-02
    level_params['nsweeps'] = [3, 1]

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
    problem_params['eps'] = [0.04, 0.16]
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
    description['problem_class'] = None  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = None  # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh_fft2d

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
        description['problem_class'] = allencahn2d_imex
        description['sweeper_class'] = imex_1st_order
    elif variant == 'semi-implicit-stab':
        description['problem_class'] = allencahn2d_imex_stab
        description['sweeper_class'] = imex_1st_order
    else:
        raise NotImplemented('Wrong variant specified, got %s' % variant)

    # setup parameters "in time"
    t0 = 0.0
    Tend = 0.02

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

    description['problem_params']['comm'] = space_comm
    # set level depending on rank
    controller_params['logger_level'] = controller_params['logger_level'] if space_rank == 0 else 99

    # instantiate controller
    controller = controller_MPI(controller_params=controller_params, description=description, comm=time_comm)

    # get initial values on finest level
    P = controller.S.levels[0].prob
    uinit = P.u_exact(t0)

    # if time_rank == 0:
    #     plt_helper.plt.imshow(uinit.values)
    #     plt_helper.savefig(f'uinit_{space_rank}', save_pdf=False, save_pgf=False, save_png=True)
    # exit()

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # if time_rank == 0:
    #     plt_helper.plt.imshow(uend.values)
    #     plt_helper.savefig(f'uend_{space_rank}', save_pdf=False, save_pgf=False, save_png=True)
    # exit()

    rank = comm.Get_rank()

    # filter statistics by variant (number of iterations)
    filtered_stats = filter_stats(stats, type='niter')

    # convert filtered statistics to list of iterations count, sorted by process
    iter_counts = sort_stats(filtered_stats, sortby='time')

    # compute and print statistics
    niters = np.array([item[1] for item in iter_counts])
    print(f'Mean number of iterations on rank {rank}: {np.mean(niters):.4f}')

    if rank == 0:
        timing = sort_stats(filter_stats(stats, type='timing_run'), sortby='time')

        print(f'---> Time to solution: {timing[0][1]:.4f} sec.')
        print()

    return stats


def main(cwd=''):
    """
    Main driver

    Args:
        cwd (str): current working directory (need this for testing)
    """

    # Loop over variants, exact and inexact solves
    results = {}
    for variant in ['semi-implicit-stab']:

        results[(variant, 'exact')] = run_SDC_variant(variant=variant)


if __name__ == "__main__":
    main()
