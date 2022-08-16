from argparse import ArgumentParser
import numpy as np
from mpi4py import MPI

from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.controller_classes.controller_MPI import controller_MPI
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.problem_classes.AllenCahn_MPIFFT import allencahn_imex, allencahn_imex_timeforcing
from pySDC.implementations.transfer_classes.TransferMesh_MPIFFT import fft_to_fft

# from pySDC.projects.AllenCahn_Bayreuth.AllenCahn_dump import dump

# from pySDC.projects.Performance.controller_MPI_scorep import controller_MPI


def run_simulation(name=None, nprocs_space=None):
    """
    A simple test program to do PFASST runs for the AC equation
    """

    # set MPI communicator
    comm = MPI.COMM_WORLD

    world_rank = comm.Get_rank()
    world_size = comm.Get_size()

    # split world communicator to create space-communicators
    if nprocs_space is not None:
        color = int(world_rank / nprocs_space)
    else:
        color = int(world_rank / 1)
    space_comm = comm.Split(color=color)
    space_comm.Set_name('Space-Comm')
    space_size = space_comm.Get_size()
    space_rank = space_comm.Get_rank()

    # split world communicator to create time-communicators
    if nprocs_space is not None:
        color = int(world_rank % nprocs_space)
    else:
        color = int(world_rank / world_size)
    time_comm = comm.Split(color=color)
    time_comm.Set_name('Time-Comm')
    time_size = time_comm.Get_size()
    time_rank = time_comm.Get_rank()

    # print(time_size, space_size, world_size)

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-08
    level_params['dt'] = 1e-03
    level_params['nsweeps'] = [3, 1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = [3]
    sweeper_params['QI'] = ['LU']  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['initial_guess'] = 'zero'

    # initialize problem parameters
    problem_params = dict()
    problem_params['L'] = 4.0
    # problem_params['L'] = 16.0
    problem_params['nvars'] = [(48 * 12, 48 * 12), (8 * 12, 8 * 12)]
    # problem_params['nvars'] = [(48 * 48, 48 * 48), (8 * 48, 8 * 48)]
    problem_params['eps'] = [0.04]
    problem_params['radius'] = 0.25
    problem_params['comm'] = space_comm
    problem_params['name'] = name
    problem_params['init_type'] = 'circle_rand'
    problem_params['spectral'] = False

    if name == 'AC-bench-constforce':
        problem_params['dw'] = [-23.59]

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30 if space_rank == 0 else 99  # set level depending on rank
    controller_params['predict_type'] = 'fine_only'
    # controller_params['hook_class'] = dump  # activate to get data output at each step

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = fft_to_fft

    if name == 'AC-bench-noforce' or name == 'AC-bench-constforce':
        description['problem_class'] = allencahn_imex
    elif name == 'AC-bench-timeforce':
        description['problem_class'] = allencahn_imex_timeforcing
    else:
        raise NotImplementedError(f'{name} is not implemented')

    # set time parameters
    t0 = 0.0
    Tend = 240 * 0.001

    if space_rank == 0 and time_rank == 0:
        out = f'---------> Running {name} with {time_size} process(es) in time and {space_size} process(es) in space...'
        print(out)

    # instantiate controller
    controller = controller_MPI(controller_params=controller_params, description=description, comm=time_comm)

    # get initial values on finest level
    P = controller.S.levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    timing = get_sorted(stats, type='timing_setup', sortby='time')
    max_timing_setup = time_comm.allreduce(timing[0][1], MPI.MAX)
    timing = get_sorted(stats, type='timing_run', sortby='time')
    max_timing = time_comm.allreduce(timing[0][1], MPI.MAX)

    if space_rank == 0 and time_rank == time_size - 1:
        print()

        out = f'Setup time: {max_timing_setup:.4f} sec.'
        print(out)

        out = f'Time to solution: {max_timing:.4f} sec.'
        print(out)

        iter_counts = get_sorted(stats, type='niter', sortby='time')
        niters = np.array([item[1] for item in iter_counts])
        out = f'Mean number of iterations: {np.mean(niters):.4f}'
        print(out)


if __name__ == "__main__":
    # Add parser to get number of processors in space and setup (have to do this here to enable automatic testing)
    parser = ArgumentParser()
    parser.add_argument(
        "-s",
        "--setup",
        help='Specifies the setup',
        type=str,
        default='AC-bench-noforce',
        choices=['AC-bench-noforce', 'AC-bench-constforce', 'AC-bench-timeforce'],
    )
    parser.add_argument("-n", "--nprocs_space", help='Specifies the number of processors in space', type=int)
    args = parser.parse_args()

    run_simulation(name=args.setup, nprocs_space=args.nprocs_space)
