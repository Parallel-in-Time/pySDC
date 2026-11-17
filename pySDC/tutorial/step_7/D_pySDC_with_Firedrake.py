# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
#   language_info:
#     name: python
# ---

# %% [markdown]
# # Part D: pySDC and Firedrake
#
# [Firedrake](https://github.com/firedrakeproject/firedrake) is a finite element library. This example runs a forced
# heat equation, $u_t = \nu u_{xx} + f$ on $[0, 1]$ with Dirichlet boundary values, discretized with finite elements in
# Firedrake. Finite elements give a system $M u' = f(u)$ with a mass matrix $M$ and $f$ a load vector, and pySDC solves
# it in this form, without ever inverting $M$: the sweepers are the mass-matrix ones, and the multilevel run uses the
# mass-matrix transfer. The problem class,
# [pySDC/implementations/problem_classes/HeatFiredrake.py](https://github.com/Parallel-in-Time/pySDC/blob/master/pySDC/implementations/problem_classes/HeatFiredrake.py),
# is given by its weak forms; [Part G](G_pySDC_with_Firedrake_forms) shows how to write such a class yourself.
#
# ## Single level, serial or parallel across the nodes
#
# SDC with the diagonal preconditioner `MIN-SR-S`, whose nodes can be solved in parallel. The space-time parallelism
# comes from a Firedrake ensemble: with `--useMPIsweeper` and a multiple of 3 ranks, the ranks are split across the
# 3 collocation nodes, and the ranks of each node share the spatial problem. See the
# [Firedrake documentation on parallelism](https://www.firedrakeproject.org/firedrake/parallelism.html).

# %%
from pathlib import Path

import numpy as np
from mpi4py import MPI


def setup(useMPIsweeper):
    """
    Helper routine to set up parameters

    Returns:
        description and controller_params parameter dictionaries
    """
    from pySDC.implementations.problem_classes.HeatFiredrake import Heat1DForcedFiredrake
    from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
    from pySDC.implementations.sweeper_classes.imex_1st_order_mass_MPI import imex_1st_order_mass_MPI
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
    description['sweeper_class'] = imex_1st_order_mass_MPI if useMPIsweeper else imex_1st_order_mass
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    return description, controller_params


# %% [markdown]
# ## Three levels
#
# Multilevel SDC, serial, coarsened in space from 128 to 32 to 4 elements. The transfer of the mass-matrix
# formulation restricts load vectors with the transpose of the interpolation, and evaluates the right-hand side anew
# after prolongation instead of interpolating it.


# %%
def setup_ML():
    """
    Helper routine to set up parameters

    Returns:
        description and controller_params parameter dictionaries
    """
    from pySDC.implementations.problem_classes.HeatFiredrake import Heat1DForcedFiredrake
    from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
    from pySDC.implementations.transfer_classes.BaseTransfer_mass import base_transfer_mass
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

    controller_params = dict()
    controller_params['logger_level'] = 15 if MPI.COMM_WORLD.rank == 0 else 30
    controller_params['hook_class'] = [LogGlobalErrorPostRun, LogWork]

    description = dict()
    description['problem_class'] = Heat1DForcedFiredrake
    description['problem_params'] = problem_params
    description['sweeper_class'] = imex_1st_order_mass
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['space_transfer_class'] = MeshToMeshFiredrake
    description['base_transfer_class'] = base_transfer_mass

    return description, controller_params


# %% [markdown]
# ## Running it
#
# The run prints the error and the work: SDC iterations, solver setups, solves and right-hand side evaluations on the
# finest level, and checks them against what we got last time.


# %%
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
    work_solves = get_sorted(stats, type='work_solver')
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

    # the results the website shows, from the first rank
    if MPI.COMM_WORLD.rank == 0:
        timing = get_sorted(stats, type='timing_run')[0][1]
        variant = 'three-level SDC' if ML else ('SDC parallel across the nodes' if useMPIsweeper else 'serial SDC')
        Path('data').mkdir(parents=True, exist_ok=True)
        with open('data/step_7_D_out.txt', 'a') as file:
            file.write(
                f'{variant:30s} error {error[0][1]:.2e}, {tot_iter:2d} iterations, {tot_solves:3d} solves on the '
                f'finest level, time to solution {timing:.2f} s\n'
            )

    # do tests that we got the same as last time
    n_nodes = 1 if useMPIsweeper else description['sweeper_params']['num_nodes']
    assert error[0][1] < 2e-7
    assert tot_iter == (13 if ML else 29)
    assert tot_solver_setup == n_nodes
    assert tot_solves == n_nodes * tot_iter
    # with several levels, f is evaluated anew at every node after the coarse-grid correction is prolonged
    assert tot_rhs == n_nodes * tot_iter * (2 if ML else 1) + (n_nodes + 1) * len(niter)


# %% [markdown]
# The script runs in three different ways:
#
# - `python D_pySDC_with_Firedrake.py` for single-level serial SDC,
# - `mpiexec -np 3 python D_pySDC_with_Firedrake.py --useMPIsweeper` for single-level SDC, parallel across the nodes,
# - `python D_pySDC_with_Firedrake.py --ML` for three-level serial SDC.
#
# Multilevel SDC reduces the number of SDC iterations quite a bit, but not necessarily the time to solution, partly
# because more solvers are constructed on the coarse levels, and the parallel variant does the work of one node per
# rank. The results below show both; mind that they come from a CI machine, not from a dedicated one. We do not claim
# to have found the best parameters: this is just an example of how to use it.

# %%
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

# %% [markdown]
# Firedrake does not run in the browser, nor in the environment this website is built in. Our CI runs all three
# variants in a Firedrake container, and checks their errors and work counts. These are its results, in the run that
# built this page:
#
# :::{literalinclude} /../../data_firedrake/step_7_D_out.txt
# :language: text
# :::
