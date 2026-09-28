import pytest


@pytest.mark.base
def test_solve_system_with_default_parameters():
    """The default Newton parameters must actually run Newton, rather than return the initial guess."""
    import numpy as np
    from pySDC.implementations.problem_classes.Auzinger_implicit import auzinger

    prob = auzinger()
    dt = 0.1
    rhs = prob.u_exact(0.0)
    u0 = prob.u_exact(0.0)

    u = prob.solve_system(rhs, dt, u0, 0.0)

    assert not np.allclose(u, u0), 'solve_system returned the initial guess unchanged'
    assert np.allclose(u - dt * prob.eval_f(u, dt), rhs, atol=1e-10), 'solve_system did not solve the system'
