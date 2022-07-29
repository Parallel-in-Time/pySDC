import subprocess

import numpy as np
from mpi4py import MPI

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.AllenCahn_2D_FD import allencahn_fullyimplicit
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.transfer_classes.TransferMesh_FFT2D import mesh_to_mesh_fft2d
from pySDC.playgrounds.Allen_Cahn.AllenCahn_monitor import monitor
from pySDC.projects.parallelSDC.BaseTransfer_MPI import base_transfer_MPI
from pySDC.projects.parallelSDC.generic_implicit_MPI import generic_implicit_MPI


# http://www.personal.psu.edu/qud2/Res/Pre/dz09sisc.pdf


def run_variant(variant=None):
    """
    Routine to run a particular variant

    Args:
        variant (str): string describing the variant

    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-07
    level_params['dt'] = 1e-03 / 2
    level_params['nsweeps'] = 1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    sweeper_params['initial_guess'] = 'zero'

    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['nu'] = 2

    problem_params['eps'] = 0.04
    problem_params['newton_maxiter'] = 100
    problem_params['newton_tol'] = 1e-08
    problem_params['lin_tol'] = 1e-09
    problem_params['lin_maxiter'] = 100
    problem_params['radius'] = 0.25

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = monitor

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = allencahn_fullyimplicit
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters

    do_print = True

    # add stuff based on variant
    if variant == 'sl_serial':
        maxmeaniters = 5.0
        sweeper_params['QI'] = ['LU']
        problem_params['nvars'] = [(128, 128)]
        description['problem_params'] = problem_params  # pass problem parameters
        description['sweeper_class'] = generic_implicit  # pass sweeper
        description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    elif variant == 'sl_parallel':
        maxmeaniters = 5.12
        assert MPI.COMM_WORLD.Get_size() == sweeper_params['num_nodes']
        sweeper_params['QI'] = ['MIN3']
        sweeper_params['comm'] = MPI.COMM_WORLD
        problem_params['nvars'] = [(128, 128)]
        description['problem_params'] = problem_params  # pass problem parameters
        description['sweeper_class'] = generic_implicit_MPI  # pass sweeper
        description['sweeper_params'] = sweeper_params  # pass sweeper parameters
        do_print = MPI.COMM_WORLD.Get_rank() == 0
    elif variant == 'ml_serial':
        maxmeaniters = 3.125
        sweeper_params['QI'] = ['LU']
        problem_params['nvars'] = [(128, 128), (64, 64)]
        description['space_transfer_class'] = mesh_to_mesh_fft2d
        description['problem_params'] = problem_params  # pass problem parameters
        description['sweeper_class'] = generic_implicit  # pass sweeper
        description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    elif variant == 'ml_parallel':
        assert MPI.COMM_WORLD.Get_size() == sweeper_params['num_nodes']
        maxmeaniters = 4.25
        sweeper_params['QI'] = ['MIN3']
        sweeper_params['comm'] = MPI.COMM_WORLD
        problem_params['nvars'] = [(128, 128), (64, 64)]
        description['problem_params'] = problem_params  # pass problem parameters
        description['sweeper_class'] = generic_implicit_MPI  # pass sweeper
        description['sweeper_params'] = sweeper_params  # pass sweeper parameters
        description['space_transfer_class'] = mesh_to_mesh_fft2d
        description['base_transfer_class'] = base_transfer_MPI
        do_print = MPI.COMM_WORLD.Get_rank() == 0
    else:
        raise NotImplementedError('Wrong variant specified, got %s' % variant)

    if do_print:
        out = 'Working on %s variant...' % variant
        print(out)

    # setup parameters "in time"
    t0 = 0
    Tend = 0.004

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # filter statistics by variant (number of iterations)
    filtered_stats = filter_stats(stats, type='niter')

    # convert filtered statistics to list of iterations count, sorted by process
    iter_counts = sort_stats(filtered_stats, sortby='time')

    # compute and print statistics
    niters = np.array([item[1] for item in iter_counts])

    if do_print:
        out = '   Mean number of iterations: %4.2f' % np.mean(niters)
        assert np.mean(niters) <= maxmeaniters, 'ERROR: number of iterations is too high, got %s instead of %s' % (
            np.mean(niters),
            maxmeaniters,
        )
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

        print('   Iteration count (nonlinear/linear): %i / %i' % (P.newton_itercount, P.lin_itercount))
        print(
            '   Mean Iteration count per call: %4.2f / %4.2f'
            % (P.newton_itercount / max(P.newton_ncalls, 1), P.lin_itercount / max(P.lin_ncalls, 1))
        )

        timing = sort_stats(filter_stats(stats, type='timing_run'), sortby='time')

        print('Time to solution: %6.4f sec.' % timing[0][1])

    return None


def main():
    """
    Main driver

    """

    run_variant(variant='sl_serial')
    print()
    run_variant(variant='ml_serial')
    print()

    cmd = (
        "mpirun -np 3 python -c \"from pySDC.projects.parallelSDC.AllenCahn_parallel import *; "
        "run_variant(\'sl_parallel\');\""
    )
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
    p.wait()
    (output, err) = p.communicate()
    print(output)

    cmd = (
        "mpirun -np 3 python -c \"from pySDC.projects.parallelSDC.AllenCahn_parallel import *; "
        "run_variant(\'ml_parallel\');\""
    )
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
    p.wait()
    (output, err) = p.communicate()
    print(output)


if __name__ == "__main__":
    main()
