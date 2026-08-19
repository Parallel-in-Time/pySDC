r"""
Delta-form SDC on the PETSc Generalized Fisher problem.

Runnable entry point in the style of ``pySDC/tutorial/step_7``: the logic lives here and the
PETSc-marked test simply calls :func:`main`.

Reduced precision is **emulated** -- PETSc fixes its scalar type at build time, so values are
rounded through the working precision while the arithmetic stays at the backend type.
"""

import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.GeneralizedFisher_1D_PETSc import petsc_fisher_fullyimplicit
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.projects.DeltaSDC.problems_petsc import petsc_fisher_delta
from pySDC.projects.DeltaSDC.sweepers import delta_implicit

PROBLEM_PARAMS = {
    'nvars': 127,
    'lambda0': 2.0,
    'nu': 1,
    'interval': (-5.0, 5.0),
    'lsol_tol': 1e-12,
    'nlsol_tol': 1e-12,
    'lsol_maxiter': 200,
    'nlsol_maxiter': 50,
}

SWEEPER_PARAMS = {
    'quad_type': 'RADAU-RIGHT',
    'node_type': 'LEGENDRE',
    'num_nodes': 3,
    'QI': 'LU',
    'initial_guess': 'spread',
}


def run(problem_class, problem_params, sweeper_class, maxiter=10, dt=0.1, nsteps=2):
    """
    Run a short Fisher simulation.

    Parameters
    ----------
    problem_class : type
        Problem class to integrate.
    problem_params : dict
        Parameters for the problem class.
    sweeper_class : type
        Sweeper class to use.
    maxiter : int, optional
        Number of SDC iterations.
    dt : float, optional
        Step size.
    nsteps : int, optional
        Number of steps.

    Returns
    -------
    tuple
        The end value and the problem instance.
    """
    description = {
        'problem_class': problem_class,
        'problem_params': problem_params,
        'sweeper_class': sweeper_class,
        'sweeper_params': SWEEPER_PARAMS,
        'level_params': {'restol': -1, 'dt': dt},
        'step_params': {'maxiter': maxiter},
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    prob = controller.MS[0].levels[0].prob
    uend, _ = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=nsteps * dt)
    return uend, prob


def main():
    """
    Compare the stock and delta-form Fisher paths, at full and emulated reduced precision.

    Returns
    -------
    dict
        The three end values, keyed by configuration.

    Raises
    ------
    AssertionError
        If the delta form deviates from the stock path, or reduced precision moves the answer.
    """
    results = {}
    results['stock'], _ = run(petsc_fisher_fullyimplicit, PROBLEM_PARAMS, generic_implicit)
    results['delta'], _ = run(petsc_fisher_delta, PROBLEM_PARAMS, delta_implicit)
    results['delta_fp32'], prob = run(
        petsc_fisher_delta, dict(PROBLEM_PARAMS, solve_precision=np.float32), delta_implicit
    )

    exact = abs(results['delta'] - results['stock'])
    emulated = abs(results['delta_fp32'] - results['stock'])

    print(f'delta form vs stock                : {exact:.3e}')
    print(f'delta form (fp32 emulated) vs stock: {emulated:.3e}')

    assert prob.solve_precision == np.dtype('float32')
    assert exact < 1e-12, f'delta form deviates from the stock path by {exact:.3e}'
    assert emulated < 1e-6, f'emulated reduced precision moves the answer by {emulated:.3e}'

    return results


if __name__ == '__main__':
    main()
