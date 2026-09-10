import numpy as np
import pytest


@pytest.mark.fenics
def test_exact_solution_is_periodic():
    """
    The benchmark compares time-dependent Dirichlet against periodic conditions in x using the
    *same* manufactured solution. That is only meaningful because the solution is genuinely
    one-periodic in x, and constant on the top and bottom boundary. Check both, since every
    conclusion drawn from this benchmark rests on it.
    """
    import dolfin as df
    from pySDC.projects.StroemungsRaum.problem_classes.NavierStokes_2D_TaylorGreen_monolithic_FEniCS import (
        fenics_NSE_2D_TaylorGreen,
    )

    prob = fenics_NSE_2D_TaylorGreen(nelems=8, t0=0.0, order=2, nu=0.05)

    for t in (0.0, 0.13, 0.4):
        prob.u_ex.t = t
        prob.p_ex.t = t
        for y in np.linspace(-0.5, 0.5, 11):
            left = np.hstack([prob.u_ex(-0.5, y), prob.p_ex(-0.5, y)])
            right = np.hstack([prob.u_ex(0.5, y), prob.p_ex(0.5, y)])
            assert np.allclose(left, right, atol=1e-12), f"solution is not periodic in x at t={t}, y={y}"

        for x in np.linspace(-0.5, 0.5, 11):
            for y in (-0.5, 0.5):
                value = np.hstack([prob.u_ex(x, y), prob.p_ex(x, y)])
                assert np.allclose(value, [1.0, 0.0, 1.0], atol=1e-12), f"top/bottom data varies at t={t}"

    # the periodic function space must not lose anything when representing this solution
    prob_periodic = fenics_NSE_2D_TaylorGreen(nelems=8, t0=0.0, order=2, nu=0.05, periodic=True)
    assert prob_periodic.W.dim() < prob.W.dim(), "periodic space should have fewer dofs"

    u, p = prob.u_exact(0.3).values.split(deepcopy=True)
    up, pp = prob_periodic.u_exact(0.3).values.split(deepcopy=True)
    assert abs(df.norm(u, 'L2') - df.norm(up, 'L2')) < 1e-12
    assert abs(df.norm(p, 'L2') - df.norm(pp, 'L2')) < 1e-12


@pytest.mark.fenics
def test_eval_f():
    """
    Anchor ``eval_f`` on the analytical solution: the semi-discrete system is M u' = f(u, t),
    so evaluating f at the exact solution must reproduce M du/dt on the interior dofs, and do
    so with the second order expected from the interpolation of the data.

    Compared against du/dt rather than against ``solve_system`` on purpose -- a sign error
    shared by both would pass a consistency check between them.
    """
    import dolfin as df
    from pySDC.projects.StroemungsRaum.problem_classes.NavierStokes_2D_TaylorGreen_monolithic_FEniCS import (
        fenics_NSE_2D_TaylorGreen,
    )

    t, nu = 0.3, 0.05
    errors = []
    for nelems in (16, 32):
        prob = fenics_NSE_2D_TaylorGreen(nelems=nelems, t0=0.0, order=2, nu=nu)

        dudt = df.Expression(
            (
                '8*pi*pi*nu*exp(-8*pi*pi*nu*t)*sin(2*pi*(x[0] - t))*sin(pi*x[1])*cos(pi*x[1])'
                ' + 2*pi*exp(-8*pi*pi*nu*t)*cos(2*pi*(x[0] - t))*sin(pi*x[1])*cos(pi*x[1])',
                '8*pi*pi*nu*exp(-8*pi*pi*nu*t)*cos(2*pi*(x[0] - t))*cos(pi*x[1])*cos(pi*x[1])'
                ' - 2*pi*exp(-8*pi*pi*nu*t)*sin(2*pi*(x[0] - t))*cos(pi*x[1])*cos(pi*x[1])',
            ),
            pi=np.pi,
            nu=nu,
            t=t,
            degree=prob.order + 2,
        )

        ut = prob.dtype_u(prob.W)
        df.assign(ut.values.sub(0), df.interpolate(dudt, prob.V))
        expected = prob.apply_mass_matrix(ut)
        f = prob.eval_f(prob.u_exact(t), t)

        # eval_f integrates by parts, so the two can only agree away from the boundary
        prob.fix_residual(f)
        prob.fix_residual(expected)

        velocity_dofs = np.array(prob.W.sub(0).dofmap().dofs())
        a = f.values.vector()[velocity_dofs]
        b = expected.values.vector()[velocity_dofs]
        errors.append(np.linalg.norm(a - b) / np.linalg.norm(b))

    assert errors[0] < 2e-2, f"eval_f does not match M du/dt: relative error {errors[0]:.3e}"
    order = np.log2(errors[0] / errors[1])
    assert order > 1.7, f"eval_f converges at order {order:.2f}, expected second order"


@pytest.mark.fenics
def test_order_reduction():
    r"""
    The point of the whole benchmark: the same exact solution, computed with periodic
    conditions in x, reaches the design order 2M-1 of RADAU-RIGHT, while with time-dependent
    Dirichlet conditions in x it drops to the stiff order M+1.

    M = 4 is deliberate. The gap the benchmark can show is (2M-1) - (M+1) = M-2, so it is
    *identically zero for M = 2*, where both orders are 3 -- a setup with two nodes cannot
    exhibit this phenomenon no matter what else is done. At M = 4 the two orders are 7 and 5
    and the separation is unmistakable.
    """
    from pySDC.projects.StroemungsRaum.run_Navier_Stokes_TaylorGreen_FEniCS import (
        order_study,
        observed_order,
    )

    Tend, dts, num_nodes = 0.2, [0.2, 0.1, 0.05, 0.025], 4
    errors, orders = {}, {}
    for periodic in (True, False):
        dts_out, errors_u, errors_p = order_study(dts, Tend, periodic=periodic, num_nodes=num_nodes)
        errors[periodic] = (errors_u, errors_p)
        # the finest pair is the most asymptotic estimate
        orders[periodic] = (observed_order(dts_out, errors_u)[-1], observed_order(dts_out, errors_p)[-1])

    design, stiff = 2 * num_nodes - 1, num_nodes + 1

    # without time-dependent boundary data the method attains its design order
    assert (
        orders[True][0] > design - 1.0
    ), f"periodic velocity order {orders[True][0]:.2f} is not close to the design order {design}"

    # with it, the pressure drops towards the stiff order and stays well clear of the design one
    assert (
        orders[False][1] < (design + stiff) / 2
    ), f"pressure order with Dirichlet data is {orders[False][1]:.2f}, expected near {stiff}"
    gap = orders[True][1] - orders[False][1]
    assert gap > 1.0, f"pressure order gap is only {gap:.2f}, expected close to {design - stiff}"

    # and the accumulated error differs by more than an order of magnitude at the finest step
    ratio = errors[False][1][-1] / errors[True][1][-1]
    assert ratio > 10.0, f"pressure error ratio at the finest step size is only {ratio:.1f}"


@pytest.mark.fenics
@pytest.mark.parametrize("periodic", [False, True])
def test_run_benchmark(periodic):
    """
    End-to-end smoke test: a couple of SDC steps have to land close to the exact solution.
    The tolerance is set by the spatial discretization error on this coarse mesh, not by the
    time integration, so it says nothing about the temporal order -- see test_order_reduction.
    """
    from pySDC.projects.StroemungsRaum.run_Navier_Stokes_TaylorGreen_FEniCS import (
        setup,
        run_simulation,
        run_postprocessing,
    )

    description, controller_params = setup(dt=0.05, periodic=periodic, nelems=16, nu=0.05, num_nodes=2)
    P, _, uend = run_simulation(description, controller_params, Tend=0.1)
    error_u, error_p = run_postprocessing(P, uend, Tend=0.1)

    assert error_u < 1e-3, f"relative velocity error {error_u:.3e} exceeds tolerance"
    assert error_p < 1e-2, f"relative pressure error {error_p:.3e} exceeds tolerance"


@pytest.mark.fenics
@pytest.mark.parametrize("periodic", [False, True])
def test_solve_system(periodic):
    """
    ``solve_system`` solves M w - factor * f(w) = rhs, so feeding it the right-hand side built
    from the exact solution must return the exact solution. Newton is started away from the
    answer so this actually exercises the solve, the Jacobian and the boundary conditions.
    """
    from pySDC.projects.StroemungsRaum.problem_classes.NavierStokes_2D_TaylorGreen_monolithic_FEniCS import (
        fenics_NSE_2D_TaylorGreen,
    )

    t, factor = 0.3, 0.01
    prob = fenics_NSE_2D_TaylorGreen(nelems=16, t0=0.0, order=2, nu=0.05, periodic=periodic)

    uex = prob.u_exact(t)
    rhs = prob.apply_mass_matrix(uex) - factor * prob.eval_f(uex, t)

    w = prob.solve_system(rhs, factor, prob.dtype_u(prob.W), t)

    rel_err = abs(w - uex) / abs(uex)
    assert rel_err < 1e-9, f"solve_system did not recover the exact solution: {rel_err:.3e}"


@pytest.mark.fenics
def test_differentiated_boundary_condition():
    r"""
    Imposing the time-dependent boundary data in differentiated form recovers most of the order
    it otherwise costs.

    Asserted on the error rather than on the observed order: the order estimates are not clean
    enough on a mesh CI can afford. The *periodic* reference itself only reaches about 6.3 here
    instead of its design order 7, and on finer step sizes it collapses to 5 against a solver
    floor near 1e-10, so separating 2M-1 from 2M-2 is beyond what this benchmark resolves. The
    error, on the other hand, is unambiguous: with the differentiated condition it lands close
    to the periodic case, while the pointwise one is an order of magnitude away.
    """
    from pySDC.projects.StroemungsRaum.run_Navier_Stokes_TaylorGreen_FEniCS import order_study

    Tend, dts = 0.4, [0.2, 0.1]
    errors = {}
    for label, kwargs in [
        ('periodic', dict(periodic=True)),
        ('pointwise', dict(periodic=False)),
        ('differentiated', dict(periodic=False, differentiated_bc=True)),
    ]:
        _, _, errors_p = order_study(dts, Tend, num_nodes=4, restol=1e-13, **kwargs)
        errors[label] = errors_p[0]

    # the differentiated condition must be a clear improvement on the pointwise one ...
    gain = errors['pointwise'] / errors['differentiated']
    assert gain > 2.5, f"differentiated boundary condition only improves the error by {gain:.1f}x"

    # ... and land near the periodic case, which is the best this discretization can do
    remaining = errors['differentiated'] / errors['periodic']
    assert remaining < 4.0, f"differentiated error is still {remaining:.1f}x the periodic one"

    # sanity: the pointwise variant is the one that is far off
    assert errors['pointwise'] / errors['periodic'] > 5.0, "pointwise variant unexpectedly accurate"


@pytest.mark.fenics
def test_differentiated_boundary_condition_needs_its_sweeper():
    """
    ``differentiated_bc`` silently doing nothing would be worse than failing, since the run
    would look fine and just be less accurate. Check both guards.
    """
    from pySDC.projects.StroemungsRaum.problem_classes.NavierStokes_2D_TaylorGreen_monolithic_FEniCS import (
        fenics_NSE_2D_TaylorGreen,
    )

    prob = fenics_NSE_2D_TaylorGreen(nelems=8, nu=0.05, differentiated_bc=True)
    with pytest.raises(RuntimeError, match='prepare_step'):
        prob.solve_system(prob.u_exact(0.0), 0.01, prob.dtype_u(prob.W), 0.0)

    with pytest.raises(ValueError, match='time-dependent'):
        fenics_NSE_2D_TaylorGreen(nelems=8, nu=0.05, periodic=True, differentiated_bc=True)
