"""
Newton solvers that check ``self.stop_at_nan`` when they hit ``nan``, which these classes never defined.
"""

import pytest


def check_stop_at_nan(make_problem, solve):
    import numpy as np
    from pySDC.core.errors import ProblemError

    prob = make_problem(stop_at_nan=True)
    u0 = prob.u_exact(0)
    rhs = prob.dtype_u(u0)
    rhs[:] = np.nan
    with pytest.raises(ProblemError, match='nan'):
        solve(prob, rhs, u0)

    prob = make_problem(stop_at_nan=False, newton_maxiter=2)
    assert np.all(np.isnan(solve(prob, rhs, u0)))


@pytest.mark.base
def test_battery_implicit():
    from pySDC.implementations.problem_classes.Battery import battery_implicit

    check_stop_at_nan(battery_implicit, lambda prob, rhs, u0: prob.solve_system(rhs, 0.1, u0, 0))


@pytest.mark.mpi4py
@pytest.mark.parametrize('name', ['mi_diffusion', 'mi_linear'])
def test_grayscott(name):
    from pySDC.implementations.problem_classes import GrayScott_MPIFFT

    problem_class = getattr(GrayScott_MPIFFT, f'grayscott_{name}')
    check_stop_at_nan(
        lambda **kwargs: problem_class(nvars=(16, 16), **kwargs),
        lambda prob, rhs, u0: prob.solve_system_2(rhs, 0.1, u0, 0),
    )
