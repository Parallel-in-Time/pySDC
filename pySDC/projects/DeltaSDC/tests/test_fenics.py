"""
FEniCS coverage for the delta form.

These tests have never been executed: DOLFIN was unavailable in the environment the module was
written in. They are marked ``fenics`` and run in CI; until they have passed there, treat
``problems_fenics`` as unvalidated.
"""

import pytest

# Skip unless the STOCK Gray-Scott problem can actually be built here. A broken or
# version-incompatible DOLFIN raises RuntimeError or AttributeError rather than ImportError, and if
# the baseline cannot be constructed there is nothing to compare the delta form against.
try:
    import dolfin  # noqa: F401
    from pySDC.implementations.problem_classes.GrayScott_1D_FEniCS_implicit import fenics_grayscott as _probe

    _probe(c_nvars=8, t0=0.0, family='CG', order=1, refinements=0, Du=1.0, Dv=0.01, A=0.09, B=0.086)
except Exception as exc:  # pragma: no cover - environment dependent
    pytest.skip(f'FEniCS Gray-Scott unavailable here: {exc}', allow_module_level=True)

GS_PARAMS = {
    'c_nvars': 128,
    't0': 0.0,
    'family': 'CG',
    'order': 2,
    'refinements': 1,
    'Du': 1.0,
    'Dv': 0.01,
    'A': 0.09,
    'B': 0.086,
}

SWEEPER_PARAMS = {
    'quad_type': 'RADAU-RIGHT',
    'node_type': 'LEGENDRE',
    'num_nodes': 3,
    'QI': 'LU',
    'initial_guess': 'spread',
}


def _run(problem_class, problem_params, sweeper_class, maxiter, dt=1.0, nsteps=2):
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    description = {
        'problem_class': problem_class,
        'problem_params': problem_params,
        'sweeper_class': sweeper_class,
        'sweeper_params': SWEEPER_PARAMS,
        'level_params': {'restol': -1, 'dt': dt},
        'step_params': {'maxiter': maxiter},
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 40}, description=description)
    prob = controller.MS[0].levels[0].prob
    uend, _ = controller.run(u0=prob.u_exact(0.0), t0=0.0, Tend=nsteps * dt)
    return uend, prob


@pytest.mark.fenics
@pytest.mark.parametrize('maxiter', [3, 6])
def test_correction_solve_matches_stock_path(maxiter):
    """The variational correction solve must reproduce the stock Gray-Scott path."""
    from pySDC.implementations.problem_classes.GrayScott_1D_FEniCS_implicit import fenics_grayscott
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.projects.DeltaSDC.problems_fenics import fenics_grayscott_delta
    from pySDC.projects.DeltaSDC.sweepers import delta_implicit

    u_stock, _ = _run(fenics_grayscott, GS_PARAMS, generic_implicit, maxiter)
    u_delta, _ = _run(fenics_grayscott_delta, GS_PARAMS, delta_implicit, maxiter)
    assert abs(u_stock - u_delta) < 1e-10, f'deviates by {abs(u_stock - u_delta):.3e}'


@pytest.mark.fenics
def test_emulated_reduced_precision_retains_accuracy():
    """Capping the correction's information at float32 must not move the answer."""
    import numpy as np
    from pySDC.implementations.problem_classes.GrayScott_1D_FEniCS_implicit import fenics_grayscott
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.projects.DeltaSDC.problems_fenics import fenics_grayscott_delta
    from pySDC.projects.DeltaSDC.sweepers import delta_implicit

    u_stock, _ = _run(fenics_grayscott, GS_PARAMS, generic_implicit, 6)
    u_emulated, prob = _run(fenics_grayscott_delta, dict(GS_PARAMS, solve_precision=np.float32), delta_implicit, 6)

    assert prob.solve_precision == np.dtype('float32')
    assert abs(u_stock - u_emulated) < 1e-6, f'deviates by {abs(u_stock - u_emulated):.3e}'


@pytest.mark.fenics
def test_increment_vanishes_with_the_correction():
    """The analytically expanded increment must be free of cancellation, i.e. vanish with delta."""
    import dolfin as df
    import numpy as np
    from pySDC.projects.DeltaSDC.problems_fenics import fenics_grayscott_delta

    prob = fenics_grayscott_delta(**GS_PARAMS)
    base = prob.u_exact(0.0)
    prob.base.assign(base.values)

    q1, q2 = df.TestFunctions(prob.V)
    magnitudes = []
    for scale in [1e-2, 1e-4, 1e-6]:
        scaled = df.Function(prob.V)
        scaled.vector().set_local(scale * np.ones(scaled.vector().get_local().shape))
        scaled.vector().apply('insert')
        prob.delta.assign(scaled)
        vector = df.assemble(prob._increment_forms((q1, q2)))
        magnitudes.append(float(np.max(np.abs(vector.get_local()))))

    assert magnitudes[0] > magnitudes[1] > magnitudes[2], f'increment did not shrink: {magnitudes}'
    assert magnitudes[2] < 1e-3 * magnitudes[0]
