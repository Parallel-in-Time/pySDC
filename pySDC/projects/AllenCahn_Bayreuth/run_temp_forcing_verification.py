from argparse import ArgumentParser
import json
import numpy as np
from mpi4py import MPI
from mpi4py_fft import newDistArray

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.problem_classes.AllenCahn_Temp_MPIFFT import allencahn_temp_imex
from pySDC.implementations.transfer_classes.TransferMesh_MPIFFT import fft_to_fft


def run_simulation(name='', spectral=None, nprocs_time=None, nprocs_space=None, dt=None, cwd='.'):
    """
    A test program to do PFASST runs for the AC equation with temperature-based forcing

    (slightly inefficient, but will run for a few seconds only)

    Args:
        name (str): name of the run, will be used to distinguish different setups
        spectral (bool): run in real or spectral space
        nprocs_time (int): number of processors in time
        nprocs_space (int): number of processors in space (None if serial)
        dt (float): time-step size
        cwd (str): current working directory
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
    level_params['restol'] = 1E-12
    level_params['dt'] = dt
    level_params['nsweeps'] = [1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]
    sweeper_params['QI'] = ['LU']  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['initial_guess'] = 'spread'

    # initialize problem parameters
    problem_params = dict()
    problem_params['L'] = 1.0
    problem_params['nvars'] = [(128, 128), (32, 32)]
    problem_params['eps'] = [0.04]
    problem_params['radius'] = 0.25
    problem_params['TM'] = 1.0
    problem_params['D'] = 0.1
    problem_params['dw'] = [21.0]
    problem_params['comm'] = space_comm
    problem_params['name'] = name
    problem_params['init_type'] = 'circle'
    problem_params['spectral'] = spectral

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30 if space_rank == 0 else 99  # set level depending on rank
    controller_params['predict_type'] = 'pfasst_burnin'

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = fft_to_fft
    description['problem_class'] = allencahn_temp_imex

    # set time parameters
    t0 = 0.0
    Tend = 1 * 0.001

    if space_rank == 0:
        out = f'---------> Running {name} with spectral={spectral} and {space_size} process(es) in space...'
        print(out)

    # instantiate controller
    controller = controller_nonMPI(num_procs=nprocs_time, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    if space_rank == 0:

        # convert filtered statistics of iterations count, sorted by time
        iter_counts = sort_stats(filter_stats(stats, type='niter'), sortby='time')
        niters = np.mean(np.array([item[1] for item in iter_counts]))
        out = f'Mean number of iterations: {niters:.4f}'
        print(out)

        # get setup time
        timing = sort_stats(filter_stats(stats, type='timing_setup'), sortby='time')
        out = f'Setup time: {timing[0][1]:.4f} sec.'
        print(out)

        # get running time
        timing = sort_stats(filter_stats(stats, type='timing_run'), sortby='time')
        out = f'Time to solution: {timing[0][1]:.4f} sec.'
        print(out)

        refname = f'{cwd}/data/AC-reference-tempforce_00001000'
        with open(f'{refname}.json', 'r') as fp:
            obj = json.load(fp)

        array = np.fromfile(f'{refname}.dat', dtype=obj['datatype'])
        array = array.reshape(obj['shape'], order='C')

        if spectral:
            ureal = newDistArray(P.fft, False)
            ureal = P.fft.backward(uend[..., 0], ureal)
            Treal = newDistArray(P.fft, False)
            Treal = P.fft.backward(uend[..., 1], Treal)
            err = max(np.amax(abs(ureal - array[..., 0])), np.amax(abs(Treal - array[..., 1])))
        else:
            err = abs(array - uend)

        out = f'...Done <---------\n'
        print(out)

        return err


def main(nprocs_space=None, cwd='.'):
    """
    Little helper routine to run the whole thing

    Args:
        nprocs_space (int): number of processors in space (None if serial)
        cwd (str): current working directory
    """
    name = 'AC-test-tempforce'

    nsteps = [2 ** i for i in range(4)]

    errors = [1]
    orders = []
    for n in nsteps:
        err = run_simulation(name=name, spectral=False, nprocs_time=n, nprocs_space=nprocs_space, dt=1E-03 / n, cwd=cwd)
        errors.append(err)
        orders.append(np.log(errors[-1] / errors[-2]) / np.log(0.5))
        print(f'Error: {errors[-1]:6.4e}')
        print(f'Order of accuracy: {orders[-1]:4.2f}\n')

    assert errors[2 + 1] < 1.4E-09, f'Errors are too high, got {errors[2 + 1]}'
    assert np.isclose(orders[3], 5, rtol=2E-02), f'Order of accuracy is not within tolerance, got {orders[3]}'

    print()

    errors = [1]
    orders = []
    for n in nsteps:
        err = run_simulation(name=name, spectral=True, nprocs_time=n, nprocs_space=nprocs_space, dt=1E-03 / n, cwd=cwd)
        errors.append(err)
        orders.append(np.log(errors[-1] / errors[-2]) / np.log(0.5))
        print(f'Error: {errors[-1]:6.4e}')
        print(f'Order of accuracy: {orders[-1]:4.2f}\n')

    assert errors[2 + 1] < 1.4E-09, f'Errors are too high, got {errors[2 + 1]}'
    assert np.isclose(orders[1], 5, rtol=7E-02), f'Order of accuracy is not within tolerance, got {orders[1]}'


if __name__ == "__main__":

    # Add parser to get number of processors in space (have to do this here to enable automatic testing)
    parser = ArgumentParser()
    parser.add_argument("-n", "--nprocs_space", help='Specifies the number of processors in space', type=int)
    args = parser.parse_args()

    main(nprocs_space=args.nprocs_space)
