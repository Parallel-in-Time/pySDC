"""
This script shows how to setup ParaDiag in pySDC for two examples and compares performance to single-level PFASST in
Jacobi mode and serial time stepping.
In PFASST, we use a diagonal preconditioner, which allows for the same amount of parallelism as ParaDiag.
We show iteration counts per step here, but both schemes have further concurrency across the nodes.

We have a linear advection example, discretized with finite differences, where ParaDiag converges in very few iterations.
PFASST, on the hand, needs a lot more iterations for this hyperbolic problem.
Note that we did not optimize either setup. With different choice of alpha in ParaDiag, or inexactness and coarsening
in PFASST, both schemes could be improved significantly.

Second is the nonlinear van der Pol oscillator. We choose the mu parameter such that the problem is not overly stiff.
Here, ParaDiag needs many iterations compared to PFASST, but remember that we only perform one Newton iteration per
ParaDiag iteration. So per node, the number of Newton iterations is equal to the number of ParaDiag iterations.
In PFASST, on the other hand, we solve the systems to some accuracy and allow more iterations. Here, ParaDiag needs
fewer Newton iterations per step in total, leaving it with greater speedup. Again, inexactness could improve PFASST.

This script is not meant to show that one parallelization scheme is better than the other. It does, however, demonstrate
that both schemes, without optimization, need fewer iterations per task than serial time stepping. Kindly refrain from
computing parallel efficiency for these examples, however. ;)
"""

import numpy as np
import sys
from pySDC.helpers.stats_helper import get_sorted

# prepare output
out_file = open('data/step_9_C_out.txt', 'w')


def my_print(*args, **kwargs):
    for output in [sys.stdout, out_file]:
        print(*args, **kwargs, file=output)


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
        newton_maxiter = 1
    else:
        from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_class

        newton_maxiter = 99
        # need diagonal preconditioner for same concurrency as ParaDiag
        sweeper_params['QI'] = 'MIN-SR-S'

    if problem == 'advection':
        from pySDC.implementations.problem_classes.AdvectionEquation_ND_FD import advectionNd as problem_class

        problem_params = {'nvars': 64, 'order': 8, 'c': 1, 'solver_type': 'GMRES', 'lintol': 1e-8}
    elif problem == 'vdp':
        from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol as problem_class

        # need to not raise an error when Newton has not converged because we do only one iteration
        problem_params = {'newton_maxiter': newton_maxiter, 'crash_at_maxiter': False, 'mu': 1, 'newton_tol': 1e-9}

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
            controller_params['average_jacobians'] = False
        elif problem == 'vdp':
            controller_params['average_jacobians'] = True
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


def compare_ParaDiag_and_PFASST(n_steps, problem):
    my_print(f'Running {problem} with {n_steps} steps')

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

    my_print(
        f'Needed {max(me[1] for me in k_PD)} ParaDiag iterations and {max(me[1] for me in k_PF)} single-level PFASST iterations'
    )
    if problem == 'advection':
        k_GMRES_PD = get_sorted(stats_PD, type='work_GMRES')
        k_GMRES_PF = get_sorted(stats_PF, type='work_GMRES')
        k_GMRES_S = get_sorted(stats_S, type='work_GMRES')
        my_print(
            f'Maximum GMRES iterations on each step: {max(me[1] for me in k_GMRES_PD)} in ParaDiag, {max(me[1] for me in k_GMRES_PF)} in single-level PFASST and {sum(me[1] for me in k_GMRES_S)} total GMRES iterations in serial'
        )
    elif problem == 'vdp':
        k_Newton_PD = get_sorted(stats_PD, type='work_newton')
        k_Newton_PF = get_sorted(stats_PF, type='work_newton')
        k_Newton_S = get_sorted(stats_S, type='work_newton')
        my_print(
            f'Maximum Newton iterations on each step: {max(me[1] for me in k_Newton_PD)} in ParaDiag, {max(me[1] for me in k_Newton_PF)} in single-level PFASST and {sum(me[1] for me in k_Newton_S)} total Newton iterations in serial'
        )
    my_print()


if __name__ == '__main__':
    out_file = open('data/step_9_C_out.txt', 'w')
    params = {
        'n_steps': 16,
    }
    compare_ParaDiag_and_PFASST(**params, problem='advection')
    compare_ParaDiag_and_PFASST(**params, problem='vdp')
