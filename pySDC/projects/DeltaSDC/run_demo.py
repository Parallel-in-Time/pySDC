r"""
Runnable demonstration of delta-form SDC with a reduced-precision node-local solve.

Shows the two claims the project rests on:

1. the delta-form sweep reproduces :class:`generic_implicit` exactly, and
2. running the node-local solve at ``float32`` neither changes the iteration count nor degrades
   the accuracy, because the solver is handed a *correction*.
"""

import numpy as np

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.AllenCahn_2D_FD import allencahn_fullyimplicit
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.projects.DeltaSDC.precision import tol_floor
from pySDC.projects.DeltaSDC.problems import allencahn_delta
from pySDC.projects.DeltaSDC.sweepers import delta_implicit

SWEEPER_PARAMS = {
    'quad_type': 'RADAU-RIGHT',
    'node_type': 'LEGENDRE',
    'num_nodes': 3,
    'QI': 'LU',
    'initial_guess': 'spread',
}


def run(problem_class, problem_params, sweeper_class, sweeper_params, dt=4e-3, nsteps=2, restol=1e-9):
    """
    Run a short Allen-Cahn simulation and return the end value together with diagnostics.

    Parameters
    ----------
    problem_class : type
        Problem class to integrate.
    problem_params : dict
        Parameters for the problem class.
    sweeper_class : type
        Sweeper class to use.
    sweeper_params : dict
        Parameters for the sweeper.
    dt : float, optional
        Step size.
    nsteps : int, optional
        Number of steps.
    restol : float, optional
        Residual tolerance for the SDC iteration.

    Returns
    -------
    dict
        Keys ``uend``, ``niter`` and ``work``.
    """
    description = {
        'problem_class': problem_class,
        'problem_params': problem_params,
        'sweeper_class': sweeper_class,
        'sweeper_params': sweeper_params,
        'level_params': {'restol': restol, 'dt': dt},
        'step_params': {'maxiter': 30},
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    prob = controller.MS[0].levels[0].prob
    uend, stats = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=nsteps * dt)
    return {
        'uend': uend,
        'niter': sum(value for _, value in get_sorted(stats, type='niter')),
        'work': {key: counter.niter for key, counter in prob.work_counters.items()},
    }


def main():
    """Run the demonstration and print a small comparison table."""
    base_params = {
        'nvars': (64, 64),
        'eps': 0.04,
        'newton_maxiter': 100,
        'newton_tol': 1e-12,
        'lin_tol': 1e-12,
        'lin_maxiter': 500,
        'radius': 0.25,
    }
    tol = tol_floor(np.float32)

    reference = run(allencahn_fullyimplicit, base_params, generic_implicit, SWEEPER_PARAMS)

    results = {'generic_implicit (fp64)': reference}
    for label, precision in [('delta-form (fp64)', None), ('delta-form (fp32 solve)', np.float32)]:
        params = dict(base_params)
        params.update({'solve_precision': precision, 'krylov_tol': tol, 'newton_rtol': tol})
        results[label] = run(allencahn_delta, params, delta_implicit, SWEEPER_PARAMS)

    print(f"{'configuration':>26} | {'sweeps':>7} {'CG':>7} | {'diff to generic_implicit':>25}")
    print('-' * 72)
    for label, result in results.items():
        diff = abs(result['uend'] - reference['uend'])
        print(f"{label:>26} | {result['niter']:>7} {result['work'].get('linear', 0):>7} | {diff:>25.3e}")
    return results


if __name__ == '__main__':
    main()
