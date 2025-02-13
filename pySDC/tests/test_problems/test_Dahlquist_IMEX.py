def test_Dahlquist_IMEX():
    from pySDC.implementations.problem_classes.TestEquation_0D import test_equation_IMEX
    import numpy as np

    N = 1
    dt = 1e-2

    lambdas_implicit = np.ones(N) * -10
    lambdas_explicit = np.ones(N) * -1e-3

    prob = test_equation_IMEX(lambdas_explicit=lambdas_explicit, lambdas_implicit=lambdas_implicit, u0=1)

    u0 = prob.u_exact(0)

    # do IMEX Euler step forward
    f0 = prob.eval_f(u0, 0)
    u1 = prob.solve_system(u0 + dt * f0.expl, dt, u0, 0)

    exact = prob.u_exact(dt)
    error = abs(u1 - exact)
    error0 = abs(u0 - exact)
    assert error < error0 * 1e-1

    # do explicit Euler step backwards
    f = prob.eval_f(u1, dt)
    u02 = u1 - dt * (f.impl + f0.expl)

    assert np.allclose(u0, u02)


if __name__ == '__main__':
    test_Dahlquist_IMEX()
