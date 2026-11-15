import pytest

SWEEPER_PARAMS = {'quad_type': 'RADAU-RIGHT', 'num_nodes': 3, 'QI': 'LU'}


def heat_forms(V, nu, c, t):
    """
    The forced heat equation of `Heat1DForcedFiredrake`, written as forms for `FormFiredrakeIMEX`, with the forcing
    as an expression, as there, so that both discretise the same equation.
    """
    import firedrake as fd
    import numpy as np

    x = fd.SpatialCoordinate(V.mesh())[0]
    forcing = -(fd.sin(t) - nu * np.pi**2 * fd.cos(t)) * fd.sin(np.pi * x)
    return {
        'V': V,
        'mass': lambda u, v: u * v * fd.dx,
        'rhs': lambda u, v: -nu * fd.inner(fd.grad(u), fd.grad(v)) * fd.dx,
        'rhs_exp': lambda u, v: forcing * v * fd.dx,
        'bcs': tuple(fd.DirichletBC(V, c, area) for area in [1, 2]),
        't': t,
        'u_exact': fd.cos(t) * fd.sin(np.pi * x) + c,
    }


def nonlinear_forms(V, nu, imex, smooth=True):
    """
    Reaction-diffusion u' = nu u'' - u^3 + g with a manufactured solution and time-dependent Dirichlet values.

    With `smooth`, the solution is quadratic in space, so it lies in the finite element space and all error is from the
    time stepping. Otherwise it is not, so that coarser meshes see a different problem. The forcing g comes from
    differentiating the exact solution with UFL.
    """
    import firedrake as fd
    from pySDC.helpers.ufl_forms import manufactured_forcing, total_rhs

    x = fd.SpatialCoordinate(V.mesh())[0]
    t = fd.Constant(0.0)
    u_exact = fd.cos(t) * (1 + x**2) + t * x if smooth else fd.cos(t) * fd.sin(3 * x) + t * x + 1

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


def run(problem_class, problem_params, sweeper_class, dt, Tend, maxiter, restol=-1.0, num_procs=1, **description):
    """Run SDC, or MLSDC and PFASST with more `description`, and return the controller and the end point."""
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    description = {
        'problem_class': problem_class,
        'problem_params': problem_params,
        'sweeper_class': sweeper_class,
        'sweeper_params': SWEEPER_PARAMS,
        'level_params': {'dt': dt, 'restol': restol},
        'step_params': {'maxiter': maxiter},
        **description,
    }
    controller = controller_nonMPI(num_procs=num_procs, controller_params={'logger_level': 30}, description=description)
    P = controller.MS[0].levels[0].prob
    uend, _ = controller.run(u0=P.u_exact(0), t0=0, Tend=Tend)
    return controller, uend


#: u at POINTS after 5 steps of dt = 0.1 (16 cells, CG4, nu = 0.1, three RADAU-RIGHT nodes, LU, exactly four
#: iterations), recorded with `Heat1DForcedFiredrake` (Firedrake 2026.10) as a check against unintended changes. (The
#: class agreed to 1e-11 with its predecessor that inverted the mass matrix and interpolated its forcing.)
POINTS = [0.1, 0.33, 0.5, 0.77]
HEAT_REFERENCE = {
    0: [0.27118792355851057, 0.7553722034832705, 0.8775825707724594, 0.5803557620288665],
    3.14: [3.411187923558511, 3.8953722034832694, 4.0175825707724595, 3.720355762028867],
}


@pytest.mark.firedrake
@pytest.mark.parametrize('c', [0, 3.14])
def test_heat_forms_match_HeatFiredrake(c):
    """
    The heat equation written as forms gives the recorded values of `Heat1DForcedFiredrake`, and so does the class.
    """
    import firedrake as fd
    import numpy as np
    from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrakeIMEX
    from pySDC.implementations.problem_classes.HeatFiredrake import Heat1DForcedFiredrake
    from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass

    nu, n, order = 0.1, 16, 4
    V = fd.FunctionSpace(fd.UnitIntervalMesh(n), 'CG', order)
    for problem_class, params in [
        (FormFiredrakeIMEX, heat_forms(V, nu, c, fd.Constant(0.0))),
        (Heat1DForcedFiredrake, {'n': n, 'nu': nu, 'c': c, 'order': order}),
    ]:
        controller, uend = run(problem_class, params, imex_1st_order_mass, 0.1, 0.5, 4)
        P = controller.MS[0].levels[0].prob
        values = fd.PointEvaluator(P.V.mesh(), [[x] for x in POINTS]).evaluate(uend.functionspace)
        assert np.allclose(values, HEAT_REFERENCE[c], rtol=0, atol=1e-12), (problem_class.__name__, values)

        # it is a real time integration, not the initial condition coming back
        assert abs(uend - P.u_exact(0.5)) < 1e-6

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


def multilevel(levels, num_nodes, num_procs, maxiter=50):
    """
    Run the nonlinear problem with a solution outside the finite element space on a mesh hierarchy.

    Returns:
        The end point and the iterations per step
    """
    import firedrake as fd
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrake
    from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass
    from pySDC.implementations.transfer_classes.BaseTransfer_mass import base_transfer_mass
    from pySDC.implementations.transfer_classes.TransferFiredrakeMesh import MeshToMeshFiredrakeHierarchy

    hierarchy = fd.MeshHierarchy(fd.UnitIntervalMesh(4), 2)
    per_level = [nonlinear_forms(fd.FunctionSpace(mesh, 'CG', 2), 0.1, False, smooth=False) for mesh in hierarchy]
    # pySDC reads lists in the problem parameters as one entry per level, finest first
    problem_params = {key: [forms[key] for forms in per_level[::-1][:levels]] for key in per_level[0]}

    description = {
        'problem_class': FormFiredrake,
        'problem_params': problem_params,
        'sweeper_class': generic_implicit_mass,
        'sweeper_params': {**SWEEPER_PARAMS, 'num_nodes': num_nodes},
        'level_params': {'dt': 0.125, 'restol': 1e-11},
        'step_params': {'maxiter': maxiter},
        'base_transfer_class': base_transfer_mass,
        'space_transfer_class': MeshToMeshFiredrakeHierarchy,
    }
    controller = controller_nonMPI(num_procs=num_procs, controller_params={'logger_level': 30}, description=description)
    P = controller.MS[0].levels[0].prob
    uend, stats = controller.run(u0=P.u_exact(0), t0=0, Tend=0.5)
    return uend, [me[1] for me in get_sorted(stats, type='niter')]


@pytest.mark.firedrake
@pytest.mark.parametrize(
    'levels, num_nodes, num_procs, max_iter',
    [(2, [3, 3], 1, 9), (2, [3, 2], 1, 13), (3, [3, 3, 2], 1, 9), (2, [3, 2], 4, 16)],
)
def test_multilevel_matches_SDC(levels, num_nodes, num_procs, max_iter):
    """
    MLSDC and PFASST converge to the solution of SDC on the finest level, with time-dependent boundary values, in
    space and in time coarsening. SDC takes 12 iterations per step here.
    """
    import numpy as np

    u_SDC, iterations_SDC = multilevel(1, 3, 1)
    assert iterations_SDC == [12] * 4

    u, iterations = multilevel(levels, num_nodes, num_procs)
    assert np.abs(u.dat.data_ro - u_SDC.dat.data_ro).max() < 1e-10
    assert max(iterations) <= max_iter, iterations


@pytest.mark.firedrake
def test_multilevel_needs_restricted_boundary_values(monkeypatch):
    """
    Without the restricted boundary values in the coarse solves, the coarse correction is not zero on the boundary
    and MLSDC stalls away from SDC, even though L2 projection is the only thing that changes the boundary values.
    """
    import numpy as np
    from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrake

    u_SDC, _ = multilevel(1, 3, 1)
    monkeypatch.setattr(FormFiredrake, 'use_restricted_boundary_values', lambda self, u, t: None)
    u, iterations = multilevel(2, [3, 3], 1, maxiter=20)
    assert iterations == [20] * 4
    assert np.abs(u.dat.data_ro - u_SDC.dat.data_ro).max() > 1e-8


@pytest.mark.firedrake
def test_restricted_boundary_values_match_times_up_to_round_off():
    """
    The restricted boundary values are found by time. The transfer and the sweeper compute the node times separately,
    so they may differ in the last bits, as 0.3 and 0.1 + 0.2 do; the solve must use the restricted values all the same.
    """
    import firedrake as fd
    import numpy as np
    from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrake

    V = fd.FunctionSpace(fd.UnitIntervalMesh(8), 'CG', 2)
    P = FormFiredrake(**nonlinear_forms(V, 0.1, False))
    restricted = P.u_init
    restricted.functionspace.assign(42.0)
    P.use_restricted_boundary_values([restricted], [0.3])

    assert 0.1 + 0.2 != 0.3
    u = P.solve_system(P.apply_mass_matrix(P.u_exact(0.3)), 0.01, P.u_init, 0.1 + 0.2)
    nodes = np.concatenate([bc.nodes for bc in P.bcs])
    assert np.allclose(u.dat.data_ro[nodes], 42.0)

    # at a time with no restricted values, the solve imposes the data
    u = P.solve_system(P.apply_mass_matrix(P.u_exact(0.4)), 0.01, P.u_init, 0.4)
    assert np.allclose(u.dat.data_ro[nodes], P.u_exact(0.4).dat.data_ro[nodes])


@pytest.mark.firedrake
def test_solver_options_follow_the_forms():
    """
    A nonlinear implicit part gets Newton, a linear one a single linear solve, and a linear one that does not depend
    on time keeps its factorization, which gives the same solutions as factorizing every time.
    """
    import firedrake as fd
    import numpy as np
    from pySDC.helpers.ufl_forms import total_rhs
    from pySDC.implementations.problem_classes.FormFiredrake import CONSTANT_JACOBIAN, LINEAR, FormFiredrake

    V = fd.FunctionSpace(fd.UnitIntervalMesh(8), 'CG', 2)
    t = fd.Constant(0.0)
    forms = heat_forms(V, 0.1, 0.0, t)
    forms['rhs'] = total_rhs(forms['rhs'], forms.pop('rhs_exp'))

    assert FormFiredrake(**nonlinear_forms(V, 0.1, False)).default_solver_parameters == {}
    assert FormFiredrake(**forms).default_solver_parameters == {**LINEAR, **CONSTANT_JACOBIAN}
    time_dependent = dict(forms, rhs=lambda u, v: fd.cos(t) * forms['rhs'](u, v))
    assert FormFiredrake(**time_dependent).default_solver_parameters == LINEAR

    kept = FormFiredrake(**forms)
    fresh = FormFiredrake(**forms, solver_parameters={'snes_lag_jacobian': 1, 'snes_lag_preconditioner': 1})
    for time in [0.1, 0.2, 0.3]:
        u, w = (P.solve_system(P.apply_mass_matrix(P.u_exact(time)), 0.05, P.u_init, time) for P in (kept, fresh))
        assert np.allclose(u.dat.data_ro, w.dat.data_ro, rtol=0, atol=1e-13)
    assert kept.work_counters['solver_setup'].niter == 1


@pytest.mark.firedrake
def test_boundary_rows_of_residual():
    """
    The boundary rows of the mass-form residual carry no equation: they stay O(1) when the interior has converged.
    The solves put the boundary values in, and `fix_residual` keeps the boundary rows out of the residual norm.
    """
    import firedrake as fd
    import numpy as np
    from pySDC.helpers.ufl_forms import total_rhs
    from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrake
    from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass

    c = 3.14
    V = fd.FunctionSpace(fd.UnitIntervalMesh(8), 'CG', 2)
    forms = heat_forms(V, 0.1, c, fd.Constant(0.0))
    forms['rhs'] = total_rhs(forms['rhs'], forms.pop('rhs_exp'))
    controller, _ = run(FormFiredrake, forms, generic_implicit_mass, 0.5, 0.5, 50, 1e-13)
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
        P.apply_mass_matrix(P.apply_mass_matrix(u))

    # u0 is a solution on every level, so forms are never evaluated at a load vector
    with pytest.raises(DataError):
        P.eval_f(P.apply_mass_matrix(u), 0.3)


@pytest.mark.firedrake
@pytest.mark.parametrize('stop', [True, False])
def test_newton_divergence(stop):
    """
    A Newton solve that does not converge stops the run, or, with `stop_at_divergence=False`, is logged and its last
    iterate returned.
    """
    import firedrake as fd
    import numpy as np
    from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrake

    V = fd.FunctionSpace(fd.UnitIntervalMesh(8), 'CG', 2)
    params = nonlinear_forms(V, 0.1, False)
    # one Newton step from zero cannot reach these tolerances on the cubic reaction
    params['solver_parameters'] = {'snes_max_it': 1, 'snes_rtol': 1e-14, 'snes_atol': 1e-15}
    P = FormFiredrake(**params, stop_at_divergence=stop)
    rhs = P.apply_mass_matrix(P.u_exact(0.3))

    if stop:
        with pytest.raises(fd.exceptions.ConvergenceError):
            P.solve_system(rhs, 0.5, P.u_init, 0.3)
    else:
        u = P.solve_system(rhs, 0.5, P.u_init, 0.3)
        assert np.isfinite(abs(u)) and abs(u) > 0


@pytest.mark.firedrake
def test_spatial_error_floor_needs_its_parameters():
    """
    The residual floor cannot do without the coarse space and the end time, and says which one is missing. It also
    refuses a relative residual and more than one step at a time.
    """
    import firedrake as fd
    from pySDC.implementations.convergence_controller_classes.spatial_error_floor_firedrake import (
        SpatialErrorFloorFiredrake,
    )
    from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrake
    from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass

    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

    coarse_mesh, fine_mesh = fd.MeshHierarchy(fd.UnitIntervalMesh(4), 1)
    coarse_space = fd.FunctionSpace(coarse_mesh, 'CG', 2)
    for given, missing in [({'coarse_space': coarse_space}, 'Tend'), ({'Tend': 1.0}, 'coarse_space')]:
        with pytest.raises(AssertionError, match=missing):
            run(
                FormFiredrake,
                nonlinear_forms(fd.FunctionSpace(fine_mesh, 'CG', 2), 0.1, False),
                generic_implicit_mass,
                0.1,
                0.1,
                1,
                convergence_controllers={SpatialErrorFloorFiredrake: given},
            )

    # a relative residual is not comparable with the error, and in a PFASST block one step would stop all of them
    def description(residual_type):
        return {
            'problem_class': FormFiredrake,
            'problem_params': nonlinear_forms(fd.FunctionSpace(fine_mesh, 'CG', 2), 0.1, False),
            'sweeper_class': generic_implicit_mass,
            'sweeper_params': SWEEPER_PARAMS,
            'level_params': {'dt': 0.1, 'restol': 1e-8, 'residual_type': residual_type},
            'step_params': {'maxiter': 5},
            'convergence_controllers': {SpatialErrorFloorFiredrake: {'coarse_space': coarse_space, 'Tend': 1.0}},
        }

    with pytest.raises(AssertionError, match='last_rel'):
        controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description('last_rel'))
    with pytest.raises(NotImplementedError, match='one step at a time'):
        controller_nonMPI(num_procs=2, controller_params={'logger_level': 30}, description=description('last_abs'))
    controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description('last_abs'))


@pytest.mark.firedrake
@pytest.mark.parametrize('n_coarse, saves_iterations', [(4, True), (64, False)])
def test_spatial_error_floor(n_coarse, saves_iterations):
    """
    With a large spatial error (8 cells), the residual floor from the spatial error estimate stops the iterations
    early without losing accuracy. With a small one (128 cells), the time error dominates and restol decides alone.
    """
    import firedrake as fd
    import numpy as np
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.convergence_controller_classes.spatial_error_floor_firedrake import (
        SpatialErrorFloorFiredrake,
    )
    from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrake
    from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass

    coarse_mesh, fine_mesh = fd.MeshHierarchy(fd.UnitIntervalMesh(n_coarse), 1)
    coarse_space = fd.FunctionSpace(coarse_mesh, 'CG', 2)
    params = nonlinear_forms(fd.FunctionSpace(fine_mesh, 'CG', 2), 0.1, False, smooth=False)

    def go(convergence_controllers):
        description = {
            'problem_class': FormFiredrake,
            'problem_params': params,
            'sweeper_class': generic_implicit_mass,
            'sweeper_params': SWEEPER_PARAMS,
            'level_params': {'dt': 0.1, 'restol': 1e-8},
            'step_params': {'maxiter': 50},
            'convergence_controllers': convergence_controllers,
        }
        controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
        P = controller.MS[0].levels[0].prob
        uend, stats = controller.run(u0=P.u_exact(0), t0=0, Tend=1.0)
        # the L2 error against the exact solution itself: its interpolant would hide the spatial error
        P.t.assign(1.0)
        error = fd.sqrt(fd.assemble((P.u_exact_expr - uend.functionspace) ** 2 * fd.dx(degree=10)))
        return sum(me[1] for me in get_sorted(stats, type='niter')), error

    iterations_restol, error_restol = go({})
    iterations, error = go({SpatialErrorFloorFiredrake: {'coarse_space': coarse_space, 'Tend': 1.0}})

    assert np.isclose(error, error_restol, rtol=1e-3), (error, error_restol)
    if saves_iterations:
        assert iterations < 0.7 * iterations_restol, (iterations, iterations_restol)
    else:
        assert iterations == iterations_restol, (iterations, iterations_restol)


def two_component_forms(V0, kind):
    """
    A two-component reaction-diffusion system with a Dirichlet condition on the first component only, on a mixed
    space (`kind='mixed'`, the condition on `W.sub(0)`) or a vector space (`kind='vector'`, on `V.sub(0)`). The second
    component has a natural boundary condition. The manufactured solution is quadratic in space, so all error is from
    the time stepping.
    """
    import firedrake as fd
    from pySDC.helpers.ufl_forms import manufactured_forcing

    mesh = V0.mesh()
    V = V0 * V0 if kind == 'mixed' else fd.VectorFunctionSpace(mesh, 'CG', V0.ufl_element().embedded_superdegree, dim=2)
    x = fd.SpatialCoordinate(mesh)[0]
    t = fd.Constant(0.0)
    u_exact = fd.as_vector([fd.cos(t) * (1 + x**2), fd.sin(t) * x**2 + 1])

    def mass(w, q):
        return fd.inner(w, q) * fd.dx

    def rhs(w, q):
        reaction = w[0] * w[1] ** 2
        return (-0.1 * fd.inner(fd.grad(w), fd.grad(q)) - reaction * q[0] + reaction * q[1]) * fd.dx

    forcing = manufactured_forcing(mass, rhs, u_exact, t, fd.TestFunction(V))
    return {
        'V': V,
        'mass': mass,
        'rhs': lambda w, q: rhs(w, q) + forcing,
        'bcs': (fd.DirichletBC(V.sub(0), u_exact[0], 'on_boundary'),),
        't': t,
        'u_exact': u_exact,
        'solver_parameters': {'snes_rtol': 1e-14, 'snes_atol': 1e-15},
    }


@pytest.mark.firedrake
@pytest.mark.parametrize('kind', ['mixed', 'vector'])
def test_dirichlet_on_one_component(kind):
    """
    A Dirichlet condition on one component of a mixed or vector space is imposed on that component only, with its
    time-dependent values, and MLSDC with restricted boundary values on the coarse level converges to SDC.
    """
    import firedrake as fd
    import numpy as np
    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.problem_classes.FormFiredrake import FormFiredrake
    from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass
    from pySDC.implementations.transfer_classes.BaseTransfer_mass import base_transfer_mass
    from pySDC.implementations.transfer_classes.TransferFiredrakeMesh import MeshToMeshFiredrakeHierarchy

    meshes = list(fd.MeshHierarchy(fd.UnitIntervalMesh(4), 1))[::-1]
    forms = [two_component_forms(fd.FunctionSpace(mesh, 'CG', 2), kind) for mesh in meshes]

    def go(levels):
        description = {
            'problem_class': FormFiredrake,
            'problem_params': {key: [level[key] for level in forms[:levels]] for key in forms[0]},
            'sweeper_class': generic_implicit_mass,
            'sweeper_params': SWEEPER_PARAMS,
            'level_params': {'dt': 0.125, 'restol': 1e-11},
            'step_params': {'maxiter': 50},
            'base_transfer_class': base_transfer_mass,
            'space_transfer_class': MeshToMeshFiredrakeHierarchy,
        }
        controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
        P = controller.MS[0].levels[0].prob
        uend, stats = controller.run(u0=P.u_exact(0), t0=0, Tend=0.5)
        return controller, uend, [me[1] for me in get_sorted(stats, type='niter')]

    controller, u_sdc, iterations_sdc = go(1)
    L = controller.MS[0].levels[0]
    P = L.prob

    # the solves impose the data on the first component at every node, and the solution is accurate
    nodes = P.bcs[0].nodes
    for m, tau in enumerate(L.sweep.coll.nodes):
        exact = P.u_exact(L.time + L.dt * tau)
        first = [part.dat.data_ro for part in (L.u[m + 1], exact)]
        if kind == 'mixed':
            first = [data[0] for data in first]
        else:
            first = [data[:, 0] for data in first]
        assert np.allclose(first[0][nodes], first[1][nodes], rtol=0, atol=1e-14)
    assert abs(u_sdc - P.u_exact(0.5)) < 1e-6

    _, u_mlsdc, iterations_mlsdc = go(2)
    difference = max(
        np.abs(np.asarray(a) - np.asarray(b)).max() for a, b in zip(u_mlsdc.dat.data_ro, u_sdc.dat.data_ro, strict=True)
    )
    assert difference < 1e-10, difference
    assert max(iterations_mlsdc) <= max(iterations_sdc), (iterations_mlsdc, iterations_sdc)
