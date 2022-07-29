import numpy as np
from mpi4py import MPI

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.problem_classes.NonlinearSchroedinger_MPIFFT import nonlinearschroedinger_imex
from pySDC.implementations.transfer_classes.TransferMesh_MPIFFT import fft_to_fft


def run_simulation(spectral=None, ml=None, num_procs=None):
    """
    A test program to do SDC, MLSDC and PFASST runs for the 2D NLS equation

    Args:
        spectral (bool): run in real or spectral space
        ml (bool): single or multiple levels
        num_procs (int): number of parallel processors
    """

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-08
    level_params['dt'] = 1e-01 / 2
    level_params['nsweeps'] = [1]

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = [3]
    sweeper_params['QI'] = ['LU']  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['initial_guess'] = 'zero'

    # initialize problem parameters
    problem_params = dict()
    if ml:
        problem_params['nvars'] = [(128, 128), (32, 32)]
    else:
        problem_params['nvars'] = [(128, 128)]
    problem_params['spectral'] = spectral
    problem_params['comm'] = comm

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30 if rank == 0 else 99
    # controller_params['predict_type'] = 'fine_only'

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_params'] = problem_params  # pass problem parameters
    description['problem_class'] = nonlinearschroedinger_imex
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = fft_to_fft

    # set time parameters
    t0 = 0.0
    Tend = 1.0

    f = None
    if rank == 0:
        f = open('step_7_B_out.txt', 'a')
        out = f'Running with ml={ml} and num_procs={num_procs}...'
        f.write(out + '\n')
        print(out)

    # instantiate controller
    controller = controller_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    uex = P.u_exact(Tend)
    err = abs(uex - uend)

    if rank == 0:
        # filter statistics by type (number of iterations)
        filtered_stats = filter_stats(stats, type='niter')

        # convert filtered statistics to list of iterations count, sorted by process
        iter_counts = sort_stats(filtered_stats, sortby='time')

        niters = np.array([item[1] for item in iter_counts])
        out = (
            f'   Min/Mean/Max number of iterations: '
            f'{np.min(niters):4.2f} / {np.mean(niters):4.2f} / {np.max(niters):4.2f}'
        )
        f.write(out + '\n')
        print(out)
        out = '   Range of values for number of iterations: %2i ' % np.ptp(niters)
        f.write(out + '\n')
        print(out)
        out = '   Position of max/min number of iterations: %2i -- %2i' % (
            int(np.argmax(niters)),
            int(np.argmin(niters)),
        )
        f.write(out + '\n')
        print(out)
        out = '   Std and var for number of iterations: %4.2f -- %4.2f' % (float(np.std(niters)), float(np.var(niters)))
        f.write(out + '\n')
        print(out)

        out = f'Error: {err:6.4e}'
        f.write(out + '\n')
        print(out)

        timing = sort_stats(filter_stats(stats, type='timing_run'), sortby='time')
        out = f'Time to solution: {timing[0][1]:6.4f} sec.'
        f.write(out + '\n')
        print(out)

        assert err <= 1.133e-05, 'Error is too high, got %s' % err
        if ml:
            if num_procs > 1:
                maxmean = 12.5
            else:
                maxmean = 6.6
        else:
            maxmean = 12.7
        assert np.mean(niters) <= maxmean, 'Mean number of iterations is too high, got %s' % np.mean(niters)

        f.write('\n')
        print()
        f.close()


def main():
    """
    Little helper routine to run the whole thing

    Note: This can also be run with "mpirun -np 2 python B_pySDC_with_mpi4pyfft.py"
    """
    run_simulation(spectral=False, ml=False, num_procs=1)
    run_simulation(spectral=True, ml=False, num_procs=1)
    run_simulation(spectral=False, ml=True, num_procs=1)
    run_simulation(spectral=True, ml=True, num_procs=1)
    run_simulation(spectral=False, ml=True, num_procs=10)
    run_simulation(spectral=True, ml=True, num_procs=10)


if __name__ == "__main__":
    main()
