"""
Regression coverage for the FEniCS Gray-Scott problem.

This problem class had no test and had silently rotted against DOLFIN 2019.1.0 -- the version
``etc/environment-fenics.yml`` pins -- to the point where it could not be constructed at all. These
tests pin the basics so that cannot happen again.
"""

import pytest

PARAMS = {
    'c_nvars': 64,
    't0': 0.0,
    'family': 'CG',
    'order': 2,
    'refinements': 1,
    'Du': 1.0,
    'Dv': 0.01,
    'A': 0.09,
    'B': 0.086,
}


@pytest.mark.fenics
def test_construction_runs_the_base_initialiser():
    """``Problem.__init__`` must actually run, which an unbound ``super()`` call skipped."""
    from pySDC.implementations.problem_classes.GrayScott_1D_FEniCS_implicit import fenics_grayscott

    prob = fenics_grayscott(**PARAMS)

    assert hasattr(prob, 'init'), 'Problem.__init__ did not run'
    assert hasattr(prob, 'logger'), 'Problem.__init__ did not run'
    assert prob.V.num_sub_spaces() == 2, 'expected a mixed space with two components'


@pytest.mark.fenics
def test_basic_operations():
    """Initial value, right-hand side and implicit solve must all work."""
    from pySDC.implementations.problem_classes.GrayScott_1D_FEniCS_implicit import fenics_grayscott

    prob = fenics_grayscott(**PARAMS)

    u = prob.u_exact(0.0)
    assert abs(u) > 0.0

    f = prob.eval_f(u, 0.0)
    assert abs(f) > 0.0

    solution = prob.solve_system(u, 0.01, u, 0.0)
    assert abs(solution) > 0.0

    with pytest.raises(AssertionError):
        prob.u_exact(1.0)


@pytest.mark.fenics
def test_runs_in_sdc():
    """A short SDC run must complete and stay bounded."""
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.problem_classes.GrayScott_1D_FEniCS_implicit import fenics_grayscott
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit

    description = {
        'problem_class': fenics_grayscott,
        'problem_params': PARAMS,
        'sweeper_class': generic_implicit,
        'sweeper_params': {
            'quad_type': 'RADAU-RIGHT',
            'node_type': 'LEGENDRE',
            'num_nodes': 3,
            'QI': 'LU',
            'initial_guess': 'spread',
        },
        'level_params': {'restol': -1, 'dt': 1.0},
        'step_params': {'maxiter': 4},
    }
    controller = controller_nonMPI(
        num_procs=1, controller_params={'logger_level': 30}, description=description
    )
    prob = controller.MS[0].levels[0].prob
    u0 = prob.u_exact(0.0)
    uend, _ = controller.run(u0=u0, t0=0.0, Tend=2.0)

    assert abs(uend) > 0.0
    assert abs(uend - u0) < 1.0, 'solution moved implausibly far in two steps'
