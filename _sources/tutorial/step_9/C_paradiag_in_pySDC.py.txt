# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # Part C: ParaDiag in pySDC
#
# Here we leave the hand-written linear algebra behind and set ParaDiag up through pySDC's controllers, comparing it
# with single-level PFASST in Jacobi mode and with serial time stepping. In PFASST, we use a diagonal preconditioner,
# which allows for the same amount of parallelism as ParaDiag. We show iteration counts per step, but both schemes
# have further concurrency across the nodes.
#
# Two examples: a linear advection problem, discretized with finite differences, and the nonlinear van der Pol
# oscillator, with `mu` chosen such that the problem is not overly stiff. Neither setup is optimized: with a
# different choice of $\alpha$ in ParaDiag, or with inexactness and coarsening in PFASST, both schemes could be
# improved significantly. This is not meant to show that one parallelization scheme is better than the other. It
# does show that both, without optimization, need fewer iterations per task than serial time stepping. Kindly refrain
# from computing parallel efficiency from these numbers, though. ;)
#
# ## The setups
#
# ParaDiag needs its own sweeper, `QDiagonalization`, and its own controller, `controller_ParaDiag_nonMPI`. Its
# controller parameters set $\alpha$, and whether to average the Jacobian (for the nonlinear problem only, as it
# costs communication).

# %%
import matplotlib.pyplot as plt
import numpy as np

from pySDC.helpers.stats_helper import get_sorted


def get_description(problem='advection', mode='ParaDiag'):
    level_params = {}
    level_params['dt'] = 0.1
    level_params['restol'] = 1e-6

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['initial_guess'] = 'copy'

    if mode == 'ParaDiag':
        from pySDC.implementations.sweeper_classes.ParaDiagSweepers import QDiagonalization as sweeper_class

        # we only want to use the averaged Jacobian and do only one Newton iteration per ParaDiag iteration!
    else:
        from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_class

        # need diagonal preconditioner for same concurrency as ParaDiag
        sweeper_params['QI'] = 'MIN-SR-S'

    if problem == 'advection':
        from pySDC.implementations.problem_classes.AdvectionEquation_ND_FD import advectionNd as problem_class

        problem_params = {'nvars': 64, 'order': 8, 'c': 1, 'solver_type': 'GMRES', 'lintol': 1e-8}
    elif problem == 'vdp':
        from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol as problem_class

        # need to not raise an error when Newton has not converged because we do only one iteration
        problem_params = {'newton_maxiter': 99, 'crash_at_maxiter': False, 'mu': 1, 'newton_tol': 1e-9}

    step_params = {}
    step_params['maxiter'] = 99

    description = {}
    description['problem_class'] = problem_class
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper_class
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    return description


def get_controller_params(problem='advection', mode='ParaDiag'):
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun
    from pySDC.implementations.hooks.log_work import LogWork, LogSDCIterations

    controller_params = {}
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = [LogGlobalErrorPostRun, LogWork, LogSDCIterations]

    if mode == 'ParaDiag':
        controller_params['alpha'] = 1e-4

        # For nonlinear problems, we need to communicate the average solution, which allows to compute the average
        # Jacobian locally. For linear problems, we do not want the extra communication.
        if problem == 'advection':
            controller_params['average_jacobian'] = False
        elif problem == 'vdp':
            controller_params['average_jacobian'] = True
    else:
        # We do Block-Jacobi multi-step SDC here. It's a bit silly but it's better for comparing "speedup"
        controller_params['mssdc_jac'] = True

    return controller_params


def run_problem(
    n_steps=4,
    problem='advection',
    mode='ParaDiag',
):
    if mode == 'ParaDiag':
        from pySDC.implementations.controller_classes.controller_ParaDiag_nonMPI import (
            controller_ParaDiag_nonMPI as controller_class,
        )
    else:
        from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI as controller_class

    if mode == 'serial':
        num_procs = 1
    else:
        num_procs = n_steps

    description = get_description(problem, mode)
    controller_params = get_controller_params(problem, mode)

    controller = controller_class(num_procs=num_procs, description=description, controller_params=controller_params)

    for S in controller.MS:
        S.levels[0].prob.init = tuple([*S.levels[0].prob.init[:2]] + [np.dtype('complex128')])

    P = controller.MS[0].levels[0].prob

    t0 = 0.0
    uinit = P.u_exact(t0)

    uend, stats = controller.run(u0=uinit, t0=t0, Tend=n_steps * controller.MS[0].levels[0].dt)
    return uend, stats


# %% [markdown]
# The solution becomes complex, because the diagonalization is: `run_problem` switches the data type of the problems
# to complex numbers before the run.
#
# ## The comparison
#
# ParaDiag, PFASST in Jacobi mode and serial time stepping, for 16 steps. Besides the iterations, we compare the work
# of the inner solvers: GMRES iterations for advection, Jacobian solves for van der Pol.


# %%
def compare_ParaDiag_and_PFASST(n_steps, problem):
    print(f'Running {problem} with {n_steps} steps')

    uend_PD, stats_PD = run_problem(n_steps, problem, mode='ParaDiag')
    uend_PF, stats_PF = run_problem(n_steps, problem, mode='PFASST')
    uend_S, stats_S = run_problem(n_steps, problem, mode='serial')

    assert np.allclose(uend_PD, uend_PF)
    assert np.allclose(uend_S, uend_PD)
    assert (
        abs(uend_PD - uend_PF) > 0
    )  # two different iterative methods should not give identical results for non-zero tolerance

    k_PD = get_sorted(stats_PD, type='k')
    k_PF = get_sorted(stats_PF, type='k')

    print(
        f'Needed {max(me[1] for me in k_PD)} ParaDiag iterations and {max(me[1] for me in k_PF)} single-level PFASST iterations'
    )
    if problem == 'advection':
        k_GMRES_PD = get_sorted(stats_PD, type='work_GMRES')
        k_GMRES_PF = get_sorted(stats_PF, type='work_GMRES')
        k_GMRES_S = get_sorted(stats_S, type='work_GMRES')
        print(
            f'Maximum GMRES iterations on each step: {max(me[1] for me in k_GMRES_PD)} in ParaDiag, {max(me[1] for me in k_GMRES_PF)} in single-level PFASST and {sum(me[1] for me in k_GMRES_S)} total GMRES iterations in serial'
        )
    elif problem == 'vdp':
        k_Jac_PD = get_sorted(stats_PD, type='work_jacobian_solves')
        k_Jac_PF = get_sorted(stats_PF, type='work_jacobian_solves')
        k_Jac_S = get_sorted(stats_S, type='work_jacobian_solves')
        print(
            f'Maximum Jacobian solves on each step: {max(me[1] for me in k_Jac_PD)} in ParaDiag, {max(me[1] for me in k_Jac_PF)} in single-level PFASST and {sum(me[1] for me in k_Jac_S)} total Jacobian solves in serial'
        )
    print()
    return max(me[1] for me in k_PD), max(me[1] for me in k_PF)


# %%
iterations = {problem: compare_ParaDiag_and_PFASST(n_steps=16, problem=problem) for problem in ['advection', 'vdp']}

# %% tags=["hide-input"]
fig, ax = plt.subplots(figsize=(6, 3))
x = np.arange(len(iterations))
ax.bar(x - 0.2, [k[0] for k in iterations.values()], width=0.4, label='ParaDiag')
ax.bar(x + 0.2, [k[1] for k in iterations.values()], width=0.4, label='single-level PFASST')
ax.set_xticks(x)
ax.set_xticklabels(['advection', 'van der Pol'])
ax.set_ylabel('iterations')
ax.legend(frameon=False)
fig.tight_layout()

# %% [markdown]
# ParaDiag converges in very few iterations for the hyperbolic advection problem (3), where PFASST struggles (36).
# For van der Pol, ParaDiag needs fewer iterations as well (10 against 24), with a much smaller margin. Remember
# that ParaDiag does only one Newton iteration per ParaDiag iteration, so per node, its number of Newton iterations
# equals the number of ParaDiag iterations, while PFASST solves the systems to some accuracy in every iteration. That
# makes the difference in Jacobian solves per step much larger than the one in iterations: 30 against 143. Again,
# inexactness could improve PFASST.
#
# :::{admonition} Important things to note
# - ParaDiag needs its own sweeper (`QDiagonalization`) and its own controller.
# - The solution becomes complex, because the diagonalization is.
# - ParaDiag converges in very few iterations for the hyperbolic advection example, where PFASST struggles. For the
#   van der Pol oscillator, the gap in iterations is much smaller.
# :::
#
# The checks the tests run are inside `compare_ParaDiag_and_PFASST`: all three methods agree, and the two iterative
# ones are not identical.
