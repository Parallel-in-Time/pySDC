import pytest


@pytest.mark.base
@pytest.mark.parametrize('ndim', [1, 2, 3])
def test_u_exact_solves_semi_discrete_problem(ndim):
    """
    With the second-order stencil, u_exact must be an exact solution of the semi-discrete problem du/dt = A u.
    """
    import numpy as np
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced

    prob = heatNd_unforced(nvars=(16,) * ndim, freq=(2,) * ndim, nu=0.1, order=2, bc='periodic')

    t = 0.3
    u = prob.u_exact(t)
    # u_exact(t) = exp(-nu rho t) u_exact(0), so du/dt = log(u(t)/u(0)) / t * u(t)
    dudt = np.log(np.max(u) / np.max(prob.u_exact(0))) / t * u
    assert np.allclose(prob.eval_f(u, t), dudt, atol=1e-12)
