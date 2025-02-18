import pytest


@pytest.mark.parametrize('c', [0, 3.14])
@pytest.mark.firedrake
def test_solve_system(c):
    from pySDC.implementations.problem_classes.HeatFiredrake import Heat1DForcedFiredrake
    import numpy as np
    import firedrake as fd

    # test we get the initial conditions back when solving with zero step size
    P = Heat1DForcedFiredrake(n=128, c=c)
    u0 = P.u_exact(0)
    un = P.solve_system(u0, 0)
    assert abs(u0 - un) < 1e-8

    # test we get the expected solution to a Poisson problem by setting very large step size
    dt = 1e6
    x = fd.SpatialCoordinate(P.mesh)
    u0 = P.u_init
    u0.interpolate(fd.sin(np.pi * x[0]))
    expect = P.u_init
    expect.interpolate(1 / (P.nu * np.pi**2 * dt) * fd.sin(np.pi * x[0]) + P.c)
    un = P.solve_system(u0, dt)
    error = abs(un - expect) / abs(expect)
    assert error < 1e-4, error

    # test that we arrive back where we started when going forward with IMEX Euler and backward with explicit Euler
    dt = 1e0
    u0 = P.u_exact(0)
    f = P.eval_f(u0, 0)
    un2 = P.solve_system(u0 + dt * f.expl, dt)
    fn2 = P.eval_f(un2, 0)
    u02 = un2 - dt * (fn2.impl + fn2.expl)
    error = abs(u02 - u0) / abs(u02)
    assert error < 1e-8, error


@pytest.mark.firedrake
def test_eval_f():
    from pySDC.implementations.problem_classes.HeatFiredrake import Heat1DForcedFiredrake
    import numpy as np
    import firedrake as fd

    P = Heat1DForcedFiredrake(n=128)

    me = P.u_init
    x = fd.SpatialCoordinate(P.mesh)
    me.interpolate(-fd.sin(np.pi * x[0]))

    expect = P.u_init
    expect.interpolate(P.nu * np.pi**2 * fd.sin(np.pi * x[0]))

    get = P.eval_f(me, 0).impl

    error = abs(expect - get) / abs(expect)
    assert error < 1e-8, error


if __name__ == '__main__':
    test_solve_system(0)
    # test_eval_f()
