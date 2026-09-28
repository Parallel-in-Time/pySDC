# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Part C: Iteration estimator
#
# When should SDC, MLSDC or PFASST stop iterating? So far, we stopped when the residual was small enough or after a
# fixed number of iterations. Another option is to estimate how many iterations it takes to get the error with
# respect to the exact collocation solution below a tolerance.
#
# Two consecutive iterates give an estimate of the contraction factor of the iteration,
# $\tilde L = \|u^{k} - u^{k-1}\| / \|u^{k-1} - u^{k-2}\|$. If the iteration contracts like this, the error after
# $K$ iterations is at most $\tilde L^K \|u^1 - u^0\| / (1 - \tilde L)$, which is below a tolerance $\varepsilon$
# once
#
# $$
# K \geq \frac{\log\big(\varepsilon\, (1 - \tilde L) / \|u^1 - u^0\|\big)}{\log \tilde L} .
# $$
#
# Add a few magic safety constants ($\tilde L$ is capped at 0.9, and $K$ gets 5% more) and you can guess when to
# stop. In pySDC, this is the convergence controller `CheckIterationEstimatorNonMPI`, with the tolerance as
# `errtol`:
#
# ```python
# description['convergence_controllers'] = {CheckIterationEstimatorNonMPI: {'errtol': 1e-7}}
# ```
#
# ## Checking it
#
# To see whether the estimator does its job, we need the error with respect to the exact collocation solution.
# The hook `error_output`, in `HookClass_error_output.py` next to this tutorial, gets it with some black magic: before
# each step, it runs pySDC once more inside the hook, without the estimator, and solves the collocation problem of
# that step to a residual of $10^{-14}$ or for 50 iterations, whichever comes first. After the step, it records the
# error with respect to that and to the exact solution of the PDE. For this, the description and the controller
# parameters are handed to the hook through the step parameters.
#
# :::{literalinclude} HookClass_error_output.py
# :pyobject: error_output.pre_step
# :::
#
# ## Three test cases
#
# A forced heat equation with CG as the spatial solver, an advection equation with GMRES, and the nonlinear ODE of
# Auzinger, each with SDC and with MLSDC. The setups are long, but the new part is only the entry
# `convergence_controllers` in each:

# %% tags=["hide-input"]
import matplotlib.pyplot as plt
import numpy as np

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.convergence_controller_classes.check_iteration_estimator import CheckIterationEstimatorNonMPI
from pySDC.implementations.problem_classes.AdvectionEquation_ND_FD import advectionNd
from pySDC.implementations.problem_classes.Auzinger_implicit import auzinger
from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_forced
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.implementations.transfer_classes.TransferMesh_NoCoarse import mesh_to_mesh as mesh_to_mesh_nc
from pySDC.tutorial.step_8.HookClass_error_output import error_output


def setup_diffusion(dt=None, ndim=None, ml=False):
    # initialize level parameters
    level_params = {'restol': 1e-10, 'dt': dt, 'nsweeps': 1}

    # initialize sweeper parameters
    sweeper_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': 3,
        'QI': ['LU'],  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    }

    # initialize problem parameters
    problem_params = {
        'order': 8,  # order of accuracy for FD discretization in space
        'nu': 0.1,  # diffusion coefficient
        'bc': 'periodic',  # boundary conditions
        'freq': tuple(2 for _ in range(ndim)),  # frequencies
        'solver_type': 'CG',  # do CG instead of LU
        'liniter': 10,  # number of CG iterations
    }
    if ml:
        problem_params['nvars'] = [tuple(64 for _ in range(ndim)), tuple(32 for _ in range(ndim))]  # number of dofs
    else:
        problem_params['nvars'] = tuple(64 for _ in range(ndim))  # number of dofs

    # initialize step parameters
    step_params = {'maxiter': 50, 'errtol': 1e-07}

    # initialize space transfer parameters
    space_transfer_params = {'rorder': 2, 'iorder': 6, 'periodic': True}

    # setup the iteration estimator
    convergence_controllers = {CheckIterationEstimatorNonMPI: {'errtol': 1e-7}}

    # initialize controller parameters
    controller_params = {'logger_level': 30, 'hook_class': error_output}

    # fill description dictionary for easy step instantiation
    description = {
        'problem_class': heatNd_forced,
        'problem_params': problem_params,
        'sweeper_class': imex_1st_order,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
        'convergence_controllers': convergence_controllers,
    }
    if ml:
        description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class
        description['space_transfer_params'] = space_transfer_params  # pass parameters for spatial transfer

    return description, controller_params


def setup_advection(dt=None, ndim=None, ml=False):
    # initialize level parameters
    level_params = {'restol': 1e-10, 'dt': dt, 'nsweeps': 1}

    # initialize sweeper parameters
    sweeper_params = {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': ['LU']}

    # initialize problem parameters
    problem_params = {
        'order': 6,  # order of accuracy for FD discretization in space
        'stencil_type': 'center',  # centered finite differences
        'bc': 'periodic',  # boundary conditions
        'c': 0.1,  # advection speed
        'freq': tuple(2 for _ in range(ndim)),  # frequencies
        'solver_type': 'GMRES',  # do GMRES instead of LU
        'liniter': 10,  # number of GMRES iterations
    }
    if ml:
        problem_params['nvars'] = [tuple(64 for _ in range(ndim)), tuple(32 for _ in range(ndim))]  # number of dofs
    else:
        problem_params['nvars'] = tuple(64 for _ in range(ndim))  # number of dofs

    # initialize step parameters
    step_params = {'maxiter': 50, 'errtol': 1e-07}

    # initialize space transfer parameters
    space_transfer_params = {'rorder': 2, 'iorder': 6, 'periodic': True}

    # setup the iteration estimator
    convergence_controllers = {CheckIterationEstimatorNonMPI: {'errtol': 1e-7}}

    # initialize controller parameters
    controller_params = {'logger_level': 30, 'hook_class': error_output}

    # fill description dictionary for easy step instantiation
    description = {
        'problem_class': advectionNd,
        'problem_params': problem_params,
        'sweeper_class': generic_implicit,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
        'convergence_controllers': convergence_controllers,
    }
    if ml:
        description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class
        description['space_transfer_params'] = space_transfer_params  # pass parameters for spatial transfer

    return description, controller_params


def setup_auzinger(dt=None, ml=False):
    # initialize level parameters
    level_params = {'restol': 1e-10, 'dt': dt, 'nsweeps': 1}

    # initialize sweeper parameters
    sweeper_params = {'quad_type': 'RADAU-RIGHT', 'num_nodes': [3, 2] if ml else 3, 'QI': ['LU']}

    # initialize problem parameters
    problem_params = {'newton_tol': 1e-12, 'newton_maxiter': 10}

    # initialize step parameters
    step_params = {'maxiter': 50, 'errtol': 1e-07}

    # setup the iteration estimator
    convergence_controllers = {CheckIterationEstimatorNonMPI: {'errtol': 1e-7}}

    # initialize controller parameters
    controller_params = {'logger_level': 30, 'hook_class': error_output}

    # fill description dictionary for easy step instantiation
    description = {
        'problem_class': auzinger,
        'problem_params': problem_params,
        'sweeper_class': generic_implicit,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
        'convergence_controllers': convergence_controllers,
    }
    if ml:
        description['space_transfer_class'] = mesh_to_mesh_nc  # pass spatial transfer class

    return description, controller_params


# %% [markdown]
# ## Running them
#
# For each run, we print the mean number of iterations, then for each step the error with respect to the PDE and
# to the collocation solution, and check that the latter is below the tolerance.


# %%
def run_simulations(type=None, ndim_list=None, Tend=None, nsteps_list=None, ml=False, nprocs=None):
    """
    Run one of the test cases with the iteration estimator, check it and return the collocation errors per step
    """

    t0 = None
    dt = None
    description = None
    controller_params = None

    for ndim in ndim_list:
        for nsteps in nsteps_list:
            if type == 'diffusion':
                # set time parameters
                t0 = 0.0
                dt = (Tend - t0) / nsteps
                description, controller_params = setup_diffusion(dt, ndim, ml)
                mean_number_of_iterations = 3.00 if ml else 5.75
            elif type == 'advection':
                # set time parameters
                t0 = 0.0
                dt = (Tend - t0) / nsteps
                description, controller_params = setup_advection(dt, ndim, ml)
                mean_number_of_iterations = 2.00 if ml else 4.00
            elif type == 'auzinger':
                assert ndim == 1
                # set time parameters
                t0 = 0.0
                dt = (Tend - t0) / nsteps
                description, controller_params = setup_auzinger(dt, ml)
                mean_number_of_iterations = 3.62 if ml else 5.62

            print(f'Running {type} in {ndim} dimensions with time-step size {dt}...\n')

            # Warning: this is black magic used to run an 'exact' collocation solver for each step within the hooks
            description['step_params']['description'] = description
            description['step_params']['controller_params'] = controller_params

            # instantiate controller
            controller = controller_nonMPI(
                num_procs=nprocs, controller_params=controller_params, description=description
            )

            # get initial values on finest level
            P = controller.MS[0].levels[0].prob
            uinit = P.u_exact(t0)

            # call main function to get things done...
            uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

            # filter statistics by type (number of iterations)
            iter_counts = get_sorted(stats, type='niter', sortby='time')

            niters = np.array([item[1] for item in iter_counts])
            print(f'   Mean number of iterations: {np.mean(niters):4.2f}')

            # filter statistics by type (error after time-step)
            PDE_errors = get_sorted(stats, type='PDE_error_after_step', sortby='time')
            coll_errors = get_sorted(stats, type='coll_error_after_step', sortby='time')
            for iters, PDE_err, coll_err in zip(iter_counts, PDE_errors, coll_errors, strict=True):
                assert coll_err[1] < description['step_params']['errtol'], f'Error too high, got {coll_err[1]:8.4e}'
                print(
                    f'   Errors after step {PDE_err[0]:8.4f} with {iters[1]} iterations: '
                    f'{PDE_err[1]:8.4e} / {coll_err[1]:8.4e}'
                )
            print()

            # filter statistics by type (error after time-step)
            timing = get_sorted(stats, type='timing_run', sortby='time')
            print(f'...done, took {timing[0][1]} seconds!')

            print()
        print('-----------------------------------------------------------------------------')

    assert np.isclose(
        mean_number_of_iterations, np.mean(niters), atol=1e-2
    ), f'Expected {mean_number_of_iterations:.2f} mean iterations, but got {np.mean(niters):.2f}'
    return coll_errors


# %% tags=["scroll-output"]
coll_errors = {}
for case in ['diffusion', 'advection', 'auzinger']:
    for ml in [False, True]:
        name = f'{case}, {"MLSDC" if ml else "SDC"}'
        coll_errors[name] = run_simulations(type=case, ndim_list=[1], Tend=1.0, nsteps_list=[8], ml=ml, nprocs=1)

# %% tags=["hide-input"]
fig, ax = plt.subplots(figsize=(7, 3.5))
for name, errors in coll_errors.items():
    ax.semilogy(*zip(*errors, strict=True), 'o-', label=name)
ax.axhline(1e-7, color='k', ls='--', label='errtol')
ax.set_xlabel('time')
ax.set_ylabel('error w.r.t. collocation solution')
ax.legend(frameon=False, fontsize=8, ncol=2)
ax.grid(alpha=0.3)
fig.tight_layout()

# %% [markdown]
# In every run and every step, the estimator stopped the iteration with the error below the tolerance, and without
# ever knowing the collocation solution. It stops earlier than the residual tolerance of $10^{-10}$ alone would:
# without the estimator, the mean numbers of iterations are 7.88, 4.62, 5.00, 3.00, 7.50 and 4.50 instead. It is
# not overly cautious either: the errors end between about $5 \cdot 10^{-11}$ and $3 \cdot 10^{-8}$, below the
# tolerance of $10^{-7}$ in every step.
#
# :::{admonition} Important things to note
# - The estimator can be used with several parallel steps, too. With the controller parameter `all_to_done`, the
#   estimate of the last step then decides when the whole block stops. It has seen little testing there, though.
# - It is not available for the parallel `controller_MPI` yet: as its name says, `CheckIterationEstimatorNonMPI`
#   works with the emulated parallelism of `controller_nonMPI` only.
# :::
#
# The checks the tests run are inside `run_simulations`: the collocation error after every step, and the mean
# number of iterations of each run.
