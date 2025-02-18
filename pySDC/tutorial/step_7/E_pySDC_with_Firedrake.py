"""
Simple example running a forced heat equation in Firedrake.

The function `setup` generates the description and controller_params dictionaries needed to run SDC with diagonal preconditioner.
This proceeds very similar to earlier tutorials. The interesting part of this tutorial is rather in the problem class.
See `pySDC/implementations/problem_classes/HeatFiredrake` for an easy example of how to use Firedrake within pySDC.

The script allows to run in three different ways. Use
 - `python E_pySDC_with_Firedrake.py` for single-level serial SDC
 - `mpiexec -np 3 E_pySDC_with_Firedrake --useMPIsweeper` for single-level MPI-parallel diagonal SDC
 - `python E_pySDC_with_Firedrake --ML` for three-level serial SDC

You should notice that speedup of MPI parallelisation is quite good and that, while multi-level SDC reduces the number
of SDC iterations quite a bit, it does not reduce time to solution in this case. This is partly due to more solvers being
constructed when using coarse levels. Also, we do not claim to have found the best parameters, though. This is just an
example to demonstrate how to use it.
"""

import numpy as np
from mpi4py import MPI


def setup(useMPIsweeper):
    """
    Helper routine to set up parameters

    Returns:
        description and controller_params parameter dictionaries
    """
    from pySDC.implementations.problem_classes.HeatFiredrake import Heat1DForcedFiredrake
    from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
    from pySDC.implementations.sweeper_classes.imex_1st_order_MPI import imex_1st_order_MPI
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun
    from pySDC.implementations.hooks.log_work import LogWork
    from pySDC.helpers.firedrake_ensemble_communicator import FiredrakeEnsembleCommunicator

    # setup space-time parallelism via ensemble for Firedrake, see https://www.firedrakeproject.org/firedrake/parallelism.html
    num_nodes = 3
    ensemble = FiredrakeEnsembleCommunicator(MPI.COMM_WORLD, max([MPI.COMM_WORLD.size // num_nodes, 1]))

    level_params = dict()
    level_params['restol'] = 5e-10
    level_params['dt'] = 0.2

    step_params = dict()
    step_params['maxiter'] = 20

    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = num_nodes
    sweeper_params['QI'] = 'MIN-SR-S'
    sweeper_params['QE'] = 'PIC'
    sweeper_params['comm'] = ensemble

    problem_params = dict()
    problem_params['nu'] = 0.1
    problem_params['n'] = 128
    problem_params['c'] = 1.0
    problem_params['comm'] = ensemble.space_comm

    controller_params = dict()
    controller_params['logger_level'] = 15 if MPI.COMM_WORLD.rank == 0 else 30
    controller_params['hook_class'] = [LogGlobalErrorPostRun, LogWork]

    description = dict()
    description['problem_class'] = Heat1DForcedFiredrake
    description['problem_params'] = problem_params
    description['sweeper_class'] = imex_1st_order_MPI if useMPIsweeper else imex_1st_order
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    return description, controller_params


def setup_ML():
    """
    Helper routine to set up parameters

    Returns:
        description and controller_params parameter dictionaries
    """
    from pySDC.implementations.problem_classes.HeatFiredrake import Heat1DForcedFiredrake
    from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
    from pySDC.implementations.sweeper_classes.imex_1st_order_MPI import imex_1st_order_MPI
    from pySDC.implementations.transfer_classes.TransferFiredrakeMesh import MeshToMeshFiredrake
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun
    from pySDC.implementations.hooks.log_work import LogWork
    from pySDC.helpers.firedrake_ensemble_communicator import FiredrakeEnsembleCommunicator

    level_params = dict()
    level_params['restol'] = 5e-10
    level_params['dt'] = 0.2

    step_params = dict()
    step_params['maxiter'] = 20

    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'MIN-SR-S'
    sweeper_params['QE'] = 'PIC'

    problem_params = dict()
    problem_params['nu'] = 0.1
    problem_params['n'] = [128, 32, 4]
    problem_params['c'] = 1.0

    base_transfer_params = dict()
    base_transfer_params['finter'] = True

    controller_params = dict()
    controller_params['logger_level'] = 15 if MPI.COMM_WORLD.rank == 0 else 30
    controller_params['hook_class'] = [LogGlobalErrorPostRun, LogWork]

    description = dict()
    description['problem_class'] = Heat1DForcedFiredrake
    description['problem_params'] = problem_params
    description['sweeper_class'] = imex_1st_order
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['space_transfer_class'] = MeshToMeshFiredrake
    description['base_transfer_params'] = base_transfer_params

    return description, controller_params


def runHeatFiredrake(useMPIsweeper=False, ML=False):
    """
    Run the example defined by the above parameters
    """
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.helpers.stats_helper import get_sorted

    Tend = 1.0
    t0 = 0.0

    if ML:
        assert not useMPIsweeper, 'MPI parallel diagonal SDC and ML SDC are not compatible at the moment'
        description, controller_params = setup_ML()
    else:
        description, controller_params = setup(useMPIsweeper)

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(0.0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # see what we get
    error = get_sorted(stats, type='e_global_post_run')
    work_solver_setup = get_sorted(stats, type='work_solver_setup')
    work_solves = get_sorted(stats, type='work_solves')
    work_rhs = get_sorted(stats, type='work_rhs')
    niter = get_sorted(stats, type='niter')

    tot_iter = np.sum([me[1] for me in niter])
    tot_solver_setup = np.sum([me[1] for me in work_solver_setup])
    tot_solves = np.sum([me[1] for me in work_solves])
    tot_rhs = np.sum([me[1] for me in work_rhs])

    time_rank = description["sweeper_params"]["comm"].rank if useMPIsweeper else 0
    print(
        f'Finished with error {error[0][1]:.2e}. Used {tot_iter} SDC iterations, with {tot_solver_setup} solver setups, {tot_solves} solves and {tot_rhs} right hand side evaluations on the finest level of time task {time_rank}.'
    )

    # do tests that we got the same as last time
    n_nodes = 1 if useMPIsweeper else description['sweeper_params']['num_nodes']
    assert error[0][1] < 2e-8
    assert tot_iter == 10 if ML else 29
    assert tot_solver_setup == n_nodes
    assert tot_solves == n_nodes * tot_iter
    assert tot_rhs == n_nodes * tot_iter + (n_nodes + 1) * len(niter)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        '--ML',
        help='Whether you want to run multi-level',
        default=False,
        required=False,
        action='store_const',
        const=True,
    )
    parser.add_argument(
        '--useMPIsweeper',
        help='Whether you want to use MPI parallel diagonal SDC',
        default=False,
        required=False,
        action='store_const',
        const=True,
    )

    args = parser.parse_args()

    runHeatFiredrake(**vars(args))
