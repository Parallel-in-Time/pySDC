from argparse import ArgumentParser
import numpy as np
from mpi4py import MPI

from pySDC.helpers.stats_helper import get_sorted

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.problem_classes.AllenCahn_Temp_MPIFFT import allencahn_temp_imex

from pySDC.projects.AllenCahn_Bayreuth.AllenCahn_dump import dump


def run_simulation(name='', spectral=None, nprocs_space=None):
    """
    A test program to create reference data for the AC equation with temporal forcing
    Args:
        name (str): name of the run, will be used to distinguish different setups
        spectral (bool): run in real or spectral space
        nprocs_space (int): number of processors in space (None if serial)
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
    space_rank = space_comm.Get_rank()
    space_size = space_comm.Get_size()

    assert world_size == space_size, 'This script cannot run parallel-in-time with MPI, only spatial parallelism'

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-12
    level_params['dt'] = 1e-03
    level_params['nsweeps'] = [1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = [7]
    sweeper_params['QI'] = ['LU']  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['initial_guess'] = 'spread'

    # initialize problem parameters
    problem_params = dict()
    problem_params['L'] = 1.0
    problem_params['nvars'] = [(128, 128)]
    problem_params['eps'] = [0.03]
    problem_params['radius'] = 0.35682
    problem_params['TM'] = 1.0
    problem_params['D'] = 10.0
    problem_params['dw'] = [1.0]
    problem_params['comm'] = space_comm
    problem_params['init_type'] = 'circle'
    problem_params['spectral'] = spectral

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20 if space_rank == 0 else 99  # set level depending on rank
    controller_params['hook_class'] = dump

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['problem_class'] = allencahn_temp_imex

    # set time parameters
    t0 = 0.0
    Tend = 100 * 0.001

    if space_rank == 0:
        out = f'---------> Running {name} with spectral={spectral} and {space_size} process(es) in space...'
        print(out)

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    if space_rank == 0:
        print()

        # convert filtered statistics of iterations count, sorted by time
        iter_counts = get_sorted(stats, type='niter', sortby='time')
        niters = np.mean(np.array([item[1] for item in iter_counts]))
        out = f'Mean number of iterations: {niters:.4f}'
        print(out)

        # get setup time
        timing = get_sorted(stats, type='timing_setup', sortby='time')
        out = f'Setup time: {timing[0][1]:.4f} sec.'
        print(out)

        # get running time
        timing = get_sorted(stats, type='timing_run', sortby='time')
        out = f'Time to solution: {timing[0][1]:.4f} sec.'
        print(out)

        out = '...Done <---------\n'
        print(out)

    space_comm.Free()


def main(nprocs_space=None):
    """
    Little helper routine to run the whole thing
    Args:
        nprocs_space (int): number of processors in space (None if serial)
    """
    name = 'AC-realistic-tempforce'
    run_simulation(name=name, spectral=False, nprocs_space=nprocs_space)
    # run_simulation(name=name, spectral=True, nprocs_space=nprocs_space)


if __name__ == "__main__":
    # Add parser to get number of processors in space (have to do this here to enable automatic testing)
    parser = ArgumentParser()
    parser.add_argument("-n", "--nprocs_space", help='Specifies the number of processors in space', type=int)
    args = parser.parse_args()

    main(nprocs_space=args.nprocs_space)
