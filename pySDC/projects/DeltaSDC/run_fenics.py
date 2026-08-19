r"""
Delta-form SDC on a FEniCS problem.

Runnable entry point in the style of ``pySDC/tutorial/step_7/A_pySDC_with_FEniCS.py``: the logic
lives here and the FEniCS-marked test simply calls :func:`main`.

Uses the FEniCS heat equation with the IMEX delta-form sweeper, which reproduces the stock
``imex_1st_order`` path exactly.

Note what this backend showed about ``linear_implicit=True``: that shortcut reuses the stock
``solve_system`` to solve the correction equation directly, which additionally requires the solve to
be **homogeneous in its boundary conditions**. ``fenics_heat.solve_system`` applies inhomogeneous
Dirichlet data (``self.bc.apply(T, b)``) to whatever right-hand side it is handed, so using it for a
correction imposes the wrong boundary values -- the correction must carry *zero* boundary data. The
substitution fallback is used here instead, and is exact.

Reduced-precision *storage* of the corrections is **not** available on this backend: a
``fenics_mesh`` is backed by a DOLFIN function and cannot be built at another precision, so
requesting ``correction_precision`` raises a clear ``NotImplementedError``. That is asserted below
so the limitation stays visible. Reduced precision on a FEniCS backend would have to come from a
single-precision PETSc/DOLFIN build, which is a build-time choice.
"""

import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced import fenics_heat
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.projects.DeltaSDC.sweepers import delta_imex_1st_order

T0 = 0.0

PROBLEM_PARAMS = {
    'nu': 0.1,
    't0': T0,
    'c_nvars': 128,
    'family': 'CG',
    'c': 1.0,
    'order': 4,
    'refinements': 1,
}


def sweeper_params(**extra):
    """Collocation and preconditioner settings, mirroring the step_7 tutorial."""
    params = {
        'quad_type': 'RADAU-RIGHT',
        'node_type': 'LEGENDRE',
        'num_nodes': 3,
        'QI': 'IE',
        'QE': 'EE',
        'initial_guess': 'spread',
    }
    params.update(extra)
    return params


def run(sweeper_class, sweeper_params_, maxiter=6, dt=0.2, nsteps=2):
    """
    Run a short FEniCS heat-equation simulation.

    Parameters
    ----------
    sweeper_class : type
        Sweeper class to use.
    sweeper_params_ : dict
        Parameters for the sweeper.
    maxiter : int, optional
        Number of SDC iterations.
    dt : float, optional
        Step size.
    nsteps : int, optional
        Number of steps.

    Returns
    -------
    dtype_u
        The end value.
    """
    description = {
        'problem_class': fenics_heat,
        'problem_params': PROBLEM_PARAMS,
        'sweeper_class': sweeper_class,
        'sweeper_params': sweeper_params_,
        'level_params': {'restol': -1, 'dt': dt},
        'step_params': {'maxiter': maxiter},
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    prob = controller.MS[0].levels[0].prob
    uend, _ = controller.run(u0=prob.u_exact(T0), t0=T0, Tend=T0 + nsteps * dt)
    return uend


def main():
    """
    Check the delta form against the stock IMEX sweeper on a FEniCS backend.

    Returns
    -------
    dict
        The end values, keyed by configuration.

    Raises
    ------
    AssertionError
        If any delta-form variant deviates from the stock path.
    """
    results = {}
    results['stock'] = run(imex_1st_order, sweeper_params())
    results['delta'] = run(delta_imex_1st_order, sweeper_params())
    tolerances = {'delta': 1e-11}
    for label, tolerance in tolerances.items():
        deviation = abs(results[label] - results['stock'])
        print(f'{label:>22} vs stock: {deviation:.3e} (tol {tolerance:.0e})')
        assert deviation < tolerance, f'{label} deviates from the stock path by {deviation:.3e}'

    # reduced-precision correction storage is unavailable here, and must say so clearly
    try:
        run(delta_imex_1st_order, sweeper_params(correction_precision=np.dtype('float32')))
    except NotImplementedError as error:
        print(f'correction_precision correctly refused: {error}')
    else:  # pragma: no cover - guards against a silent regression
        raise AssertionError('correction_precision should not be silently accepted for fenics_mesh')

    return results


if __name__ == '__main__':
    main()
