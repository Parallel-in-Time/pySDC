import numpy as np
import pytest


@pytest.mark.parametrize('t', [0.3, 3.14])
def test_pointwise_BCs_hold(t):
    """A solve at time `t` puts g(t) on the boundary; T_n(1) = 1 and T_n(-1) = (-1)^n read it off the coefficients"""
    from pySDC.playgrounds.time_dep_BCs.heat_time_dep_BCs import Heat1DTimeDepBCs

    P = Heat1DTimeDepBCs(a=1, b=-3.14, nvars=32, spectral_space=True)
    u_hat = P.solve_system(P.u_exact(0), 0.1, None, t)[0]
    left, right = P.g(t)
    assert np.isclose(u_hat.sum(), right)
    assert np.isclose(u_hat[::2].sum() - u_hat[1::2].sum(), left)


def test_three_ways_of_imposing_BCs():
    """
    All three solve the same problem, and lifting and differentiating gain an order over pointwise data.
    The two remedies coincide here because the lift is linear in x; see README.
    """
    from pySDC.playgrounds.time_dep_BCs.run_time_dep_BCs import convergence, orders

    M = 4
    p, u = {}, {}
    for mode in ['pointwise', 'lifted', 'differentiated']:
        _, errors, u[mode] = convergence(M, 1, mode, n_dt=4, Tend=1)
        p[mode] = orders(errors)[-1]

    assert abs(u['lifted'] - u['pointwise']) < 1e-9
    assert abs(u['differentiated'] - u['pointwise']) < 1e-9
    assert p['pointwise'] < M + 1.6, p
    assert p['lifted'] > M + 1.8, p
    assert p['differentiated'] > M + 1.8, p
