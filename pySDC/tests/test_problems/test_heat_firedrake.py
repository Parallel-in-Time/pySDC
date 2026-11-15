import pytest


@pytest.mark.parametrize('c', [0, 3.14])
@pytest.mark.firedrake
def test_solve_system(c):
    """solve_system solves (M + factor nu K) u = rhs for a load vector rhs, with the boundary values c."""
    import firedrake as fd
    import numpy as np
    from pySDC.implementations.problem_classes.HeatFiredrake import Heat1DForcedFiredrake

    # test we get the initial conditions back when solving with zero step size
    P = Heat1DForcedFiredrake(n=128, c=c)
    u0 = P.u_exact(0)
    un = P.solve_system(P.apply_mass_matrix(u0), 0, u0, 0)
    assert abs(u0 - un) < 1e-8

    # test we get the expected solution to a Poisson problem by setting very large step size
    dt = 1e6
    x = fd.SpatialCoordinate(P.mesh)
    u0 = P.u_init
    u0.interpolate(fd.sin(np.pi * x[0]))
    expect = P.u_init
    expect.interpolate(1 / (P.nu * np.pi**2 * dt) * fd.sin(np.pi * x[0]) + P.c)
    un = P.solve_system(P.apply_mass_matrix(u0), dt, u0, 0)
    error = abs(un - expect) / abs(expect)
    assert error < 1e-4, error

    # an IMEX Euler step satisfies its equation away from the boundary, where the boundary values replace it
    dt = 1e0
    u0 = P.u_exact(0)
    f = P.eval_f(u0, 0)
    rhs = P.apply_mass_matrix(u0) + dt * f.expl
    un = P.solve_system(rhs, dt, u0, 0)
    residual = P.apply_mass_matrix(un) - dt * P.eval_f(un, 0).impl - rhs
    P.fix_residual(residual)
    error = abs(residual) / abs(rhs)
    # round-off of the direct solve, 6e-12 to 1e-11 here
    assert error < 1e-10, error


@pytest.mark.firedrake
def test_eval_f():
    """eval_f gives the load vectors -nu K u and M f, the latter for the forcing at the time asked for."""
    import firedrake as fd
    import numpy as np
    from pySDC.implementations.problem_classes.HeatFiredrake import Heat1DForcedFiredrake

    P = Heat1DForcedFiredrake(n=128)
    x = fd.SpatialCoordinate(P.mesh)

    me = P.u_init
    me.interpolate(-fd.sin(np.pi * x[0]))
    expect = P.u_init
    expect.interpolate(P.nu * np.pi**2 * fd.sin(np.pi * x[0]))

    # -nu K u = M nu u'' for u = 0 on the boundary, up to the boundary rows, where integration by parts leaves a term
    difference = P.eval_f(me, 0).impl - P.apply_mass_matrix(expect)
    P.fix_residual(difference)
    error = abs(difference) / abs(P.apply_mass_matrix(expect))
    assert error < 1e-8, error

    # the time-dependent expressions follow the time they are evaluated at
    for t in [0.3, 1.7]:
        expect.interpolate(-(np.sin(t) - P.nu * np.pi**2 * np.cos(t)) * fd.sin(np.pi * x[0]))
        error = abs(P.eval_f(me, t).expl - P.apply_mass_matrix(expect)) / abs(P.apply_mass_matrix(expect))
        assert error < 1e-12, error
        expect.interpolate(np.cos(t) * fd.sin(np.pi * x[0]) + P.c)
        assert abs(P.u_exact(t) - expect) / abs(expect) < 1e-12
