import pytest

SWEEPER_PARAMS = {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'LU'}


def heat_forms(V, nu, c, t):
    """
    The forced heat equation of `Heat1DForcedFiredrake`, written as forms for `FormFiredrakeIMEX`.

    The forcing is interpolated inside the form, as `Heat1DForcedFiredrake` interpolates it, so both discretise the
    same equation.
    """
    import firedrake as fd
    import numpy as np

    x = fd.SpatialCoordinate(V.mesh())[0]
    forcing = fd.interpolate(-(fd.sin(t) - nu * np.pi**2 * fd.cos(t)) * fd.sin(np.pi * x), V)
    return {
        'V': V,
        'mass': lambda u, v: u * v * fd.dx,
        'rhs': lambda u, v: -nu * fd.inner(fd.grad(u), fd.grad(v)) * fd.dx,
        'rhs_exp': lambda u, v: forcing * v * fd.dx,
        'bcs': tuple(fd.DirichletBC(V, c, area) for area in [1, 2]),
        't': t,
        'u_exact': fd.cos(t) * fd.sin(np.pi * x) + c,
    }


def nonlinear_forms(V, nu, imex):
    """
    Reaction-diffusion u' = nu u'' - u^3 + g with a manufactured solution and time-dependent Dirichlet values.

    The solution is quadratic in space, so it lies in the finite element space and all error is from the time stepping.
    The forcing g comes from differentiating the exact solution with UFL.
    """
    import firedrake as fd
    from pySDC.helpers.ufl_forms import manufactured_forcing, total_rhs

    x = fd.SpatialCoordinate(V.mesh())[0]
    t = fd.Constant(0.0)
    u_exact = fd.cos(t) * (1 + x**2) + t * x

    def mass(u, v):
        return u * v * fd.dx

    def diffusion(u, v):
        return -nu * fd.inner(fd.grad(u), fd.grad(v)) * fd.dx

    def reaction(u, v):
        return -(u**3) * v * fd.dx

    forcing = manufactured_forcing(mass, total_rhs(diffusion, reaction), u_exact, t, fd.TestFunction(V))

    forms = {
        'V': V,
        'mass': mass,
        'bcs': (fd.DirichletBC(V, u_exact, 'on_boundary'),),
        't': t,
        'u_exact': u_exact,
        'solver_parameters': {'snes_rtol': 1e-14, 'snes_atol': 1e-15},
    }
    if imex:
        forms['rhs'] = lambda u, v: diffusion(u, v) + forcing
        forms['rhs_exp'] = reaction
    else:
        forms['rhs'] = lambda u, v: diffusion(u, v) + reaction(u, v) + forcing
    return forms


def run(problem_class, problem_params, sweeper_class, dt, Tend, maxiter, restol=-1.0):
    """Run SDC on one level and return the controller and the end point."""
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    description = {
        'problem_class': problem_class,
        'problem_params': problem_params,
        'sweeper_class': sweeper_class,
        'sweeper_params': SWEEPER_PARAMS,
        'level_params': {'dt': dt, 'restol': restol},
        'step_params': {'maxiter': maxiter},
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    P = controller.MS[0].levels[0].prob
    uend, _ = controller.run(u0=P.u_exact(0), t0=0, Tend=Tend)
    return controller, uend


@pytest.mark.firedrake
@pytest.mark.parametrize('c', [0, 3.14])
def test_heat_reproduces_HeatFiredrake(c):
    """The heat equation as forms with the mass sweeper gives what `Heat1DForcedFiredrake` gives with M^-1."""
    import firedrake as fd
    from pySDC.implementations.problem_classes.HeatFiredrake import Heat1DForcedFiredrake
    from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrakeIMEX
    from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
    from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass

    nu, n, order = 0.1, 16, 4
    _, u_ref = run(Heat1DForcedFiredrake, {'n': n, 'nu': nu, 'c': c, 'order': order}, imex_1st_order, 0.1, 0.5, 4)

    V = fd.FunctionSpace(fd.UnitIntervalMesh(n), 'CG', order)
    controller, u_form = run(
        FormFiredrakeIMEX, heat_forms(V, nu, c, fd.Constant(0.0)), imex_1st_order_mass, 0.1, 0.5, 4
    )

    diff = abs(u_ref.dat.data_ro - u_form.dat.data_ro).max()
    assert diff < 1e-12, f'forms and Heat1DForcedFiredrake differ by {diff:.2e}'

    # it is a real time integration, not the initial condition coming back
    P = controller.MS[0].levels[0].prob
    assert abs(u_form - P.u_exact(0.5)) < 1e-6

    # one solver per distinct LU diagonal entry, all reused across steps and iterations
    assert P.work_counters['solver_setup'].niter == 3
    assert P.work_counters['solver'].niter == 5 * 4 * 3


@pytest.mark.firedrake
@pytest.mark.parametrize('imex', [False, True])
def test_manufactured_solution_converges(imex):
    """Converged collocation on RADAU-RIGHT with 3 nodes is of order 5, including time-dependent boundary values."""
    import firedrake as fd
    import numpy as np
    from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrake, FormFiredrakeIMEX
    from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass
    from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass

    V = fd.FunctionSpace(fd.UnitIntervalMesh(4), 'CG', 2)
    problem_class, sweeper_class = (
        (FormFiredrakeIMEX, imex_1st_order_mass) if imex else (FormFiredrake, generic_implicit_mass)
    )

    dts = [0.125, 0.0625, 0.03125]
    errors = []
    for dt in dts:
        controller, uend = run(problem_class, nonlinear_forms(V, 0.1, imex), sweeper_class, dt, 1.0, 50, 1e-13)
        errors.append(abs(uend - controller.MS[0].levels[0].prob.u_exact(1.0)))

    orders = np.log(np.array(errors[:-1]) / errors[1:]) / np.log(2)
    # the order approaches 5 from below: 4.59, 4.66, 4.74, 4.82 from dt = 0.5 on
    assert np.all(orders > 4.7), f'expected order 5, got {orders} from errors {errors}'


@pytest.mark.firedrake
def test_boundary_rows_of_residual():
    """
    The boundary rows of the mass-form residual carry no equation: they stay O(1) when the interior has converged.
    The solves put the boundary values in, and `fix_residual` keeps the boundary rows out of the residual norm.
    """
    import firedrake as fd
    import numpy as np
    from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrake
    from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass

    c = 3.14
    V = fd.FunctionSpace(fd.UnitIntervalMesh(8), 'CG', 2)
    controller, _ = run(
        FormFiredrake, heat_forms(V, 0.1, c, fd.Constant(0.0)), generic_implicit_mass, 0.5, 0.5, 50, 1e-13
    )
    L = controller.MS[0].levels[0]
    P = L.prob
    nodes = np.concatenate([bc.nodes for bc in P.bcs])
    interior = np.setdiff1d(np.arange(V.dof_count), nodes)

    # the solves put the Dirichlet values in at every node
    for m in range(1, len(L.u)):
        assert np.allclose(L.u[m].dat.data_ro[nodes], c)

    # residual without the fix
    integral = L.sweep.integrate()
    for m in range(len(integral)):
        res = (integral[m] + P.apply_mass_matrix(L.u[0] - L.u[m + 1])).dat.data_ro
        assert abs(res[interior]).max() < 1e-12, abs(res[interior]).max()
        assert abs(res[nodes]).max() > 1e-2, 'expected the boundary rows to hold no equation'

    assert L.status.residual < 1e-13


@pytest.mark.firedrake
def test_solve_system_and_caching():
    """Factor 0 inverts the mass matrix, the solver cache evicts the oldest factor, and load vectors stay dual."""
    import firedrake as fd
    from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrake

    V = fd.FunctionSpace(fd.UnitIntervalMesh(8), 'CG', 2)
    params = nonlinear_forms(V, 0.1, False)
    P = FormFiredrake(**params, LHS_cache_size=2)

    u = P.u_exact(0.3)
    assert abs(P.solve_system(P.apply_mass_matrix(u), 0, P.u_init, 0.3) - u) < 1e-12

    for factor in [0.1, 0.2]:
        P.solve_system(P.apply_mass_matrix(u), factor, u, 0.3)
    assert list(P.solvers) == [0.1, 0.2]
    assert P.work_counters['solver_setup'].niter == 3

    with pytest.raises(NotImplementedError):
        FormFiredrake(V, params['mass'], params['rhs']).u_exact(0)

    # load vectors and solutions are kept apart: R(u) and M u are dual, u is not
    from pySDC.core.errors import DataError

    assert P.eval_f(u, 0.3).is_dual and P.apply_mass_matrix(u).is_dual and not u.is_dual
    with pytest.raises(DataError):
        P.solve_system(u, 0.1, u, 0.3)
    with pytest.raises(DataError):
        P.eval_f(P.apply_mass_matrix(u), 0.3)
