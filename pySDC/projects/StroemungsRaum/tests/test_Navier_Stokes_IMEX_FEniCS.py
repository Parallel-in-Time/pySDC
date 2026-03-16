import pytest
import numpy as np
import dolfin as df

from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh
from pySDC.projects.StroemungsRaum.problem_classes.NavierStokes_2D_FEniCS import fenics_NSE_2D_mass


@pytest.mark.fenics
def test_solve_system():

    # Physical and numerical parameters used in the test
    nu = 0.001
    t = 0.1
    factor = 0.05
    dtau = 0.01

    # Instantiate the convection-diffusion problem
    prob = fenics_NSE_2D_mass(t0=0.0, family='CG', order=2, nu=nu)

    # dummy pressure for compatibility with NSE problem class
    pr = df.Expression('-1/(pi * dtau) * (cos(pi*x[0]) + cos(pi*x[1]))', pi=np.pi, dtau=dtau, degree=2)
    pr = prob.dtype_u(df.interpolate(pr, prob.Q))
    prob.bcp = [df.DirichletBC(prob.Q, pr.values, 'on_boundary')]

    # intermediate velocity for boundary condition
    u = df.Expression(('sin(pi*x[0])', 'sin(pi*x[1])'), pi=np.pi, degree=2)
    u = prob.dtype_u(df.interpolate(u, prob.V))
    prob.bcu = [df.DirichletBC(prob.V, u.values, 'on_boundary')]

    # Define a manufactured right-hand side such that the exact solution satisfies (M - factor * K) u = rhs
    g = df.Expression(
        ('(1 + pow(pi,2) * nu * factor) * sin(pi*x[0])', '(1 + pow(pi,2) * nu * factor) * sin(pi*x[1])'),
        degree=2,
        pi=np.pi,
        nu=nu,
        factor=factor,
    )

    # Apply the mass matrix to the manufactured forcing term to obtain the rhs
    rhs = prob.apply_mass_matrix(prob.dtype_u(df.interpolate(g, prob.V)))

    # Solve the system
    _, pn = prob.solve_system(rhs=rhs, factor=factor, u0=fenics_mesh(prob.V), t=t, dtau=dtau)

    # Relative error between computed solution and exact solution
    rel_err = abs(pr - prob.dtype_u(pn, prob.Q)) / abs(pr)
    assert rel_err < 5e-4, f"Relative error {rel_err} exceeds tolerance"


@pytest.mark.fenics
def test_eval_f():

    # Physical and numerical parameters used in the test
    nu = 0.001
    t = 0.1

    # Instantiate the convection-diffusion problem
    prob = fenics_NSE_2D_mass(t0=0.0, family='CG', order=2, nu=nu)

    # Use a smooth analytical function as test input for eval_f
    uex = prob.dtype_u(df.interpolate(df.Expression(('sin(pi*x[0])', 'sin(pi*x[1])'), degree=2, pi=np.pi), prob.V))

    # Evaluate the split right-hand side at the given state and time
    f = prob.eval_f(u=uex, t=t)

    # ------------------------------------------------------------------
    # Test implicit part: diffusion term
    # f.impl should correspond to M * (nu * Delta u)
    # ------------------------------------------------------------------

    # Analytical Laplacian of the test function multiplied by nu
    lap_u = df.Expression(('-1*pow(pi,2)*nu*sin(pi*x[0])', '-1*pow(pi,2)*nu*sin(pi*x[1])'), degree=2, nu=nu, pi=np.pi)

    # Apply mass matrix to obtain M * (nu * Delta u)
    Mlap_u = prob.apply_mass_matrix(prob.dtype_u(df.interpolate(lap_u, prob.V)))

    # bouandary condition
    bc = df.DirichletBC(prob.V, lap_u, 'on_boundary')
    bc.apply(f.impl.values.vector())
    bc.apply(Mlap_u.values.vector())

    # Relative error between computed implicit term and reference
    err_diff = abs(f.impl - Mlap_u) / abs(Mlap_u)

    assert err_diff < 5e-4, f"Relative error {err_diff} exceeds tolerance"

    # ------------------------------------------------------------------
    # Test explicit part: convection term
    # Since g = 0 in the problem class, f.expl should reduce to the convection contribution
    # ------------------------------------------------------------------

    # Analytical expression for -(u · nabla)u
    Ugrad_u = df.Expression(('-pi*sin(pi*x[0])*cos(pi*x[0])', '-pi*sin(pi*x[1])*cos(pi*x[1])'), degree=2, pi=np.pi)

    # Apply mass matrix to obtain M * (-(u · nabla)u )
    MUgrad_u = prob.apply_mass_matrix(prob.dtype_u(df.interpolate(Ugrad_u, prob.V)))

    # Relative error between computed explicit term and reference
    err_conv = abs(f.expl - MUgrad_u) / abs(MUgrad_u)
    assert err_conv < 5e-4, f"Relative error {err_conv} exceeds tolerance"


@pytest.mark.fenics
def test_problem_class():

    from pySDC.projects.StroemungsRaum.run_Navier_Stokes_equations_FEniCS import setup, run_simulation

    t0 = 3.125e-04
    Tend = 0.005

    # run the setup to get description and controller parameters
    description, controller_params = setup(t0=t0)

    # run the simulation and get the problem, stats and relative error
    prob, stats, uend = run_simulation(description, controller_params, Tend)

    rho = 1

    # normal pointing out of obstacle
    n = -df.FacetNormal(prob.V.mesh())

    # tangential velocity component at the interface of the obstacle
    u_t = df.inner(df.as_vector((n[1], -n[0])), uend.values)

    # compute the drag and lift coefficients
    drag = df.Form(2 / 0.1 * (prob.nu / rho * df.inner(df.grad(u_t), n) * n[1] - prob.pn * n[0]) * prob.dsc)
    lift = df.Form(-2 / 0.1 * (prob.nu / rho * df.inner(df.grad(u_t), n) * n[0] + prob.pn * n[1]) * prob.dsc)

    # assemble the scalar values
    dc = df.assemble(drag)
    Lc = df.assemble(lift)

    # reference values at t = 0.0046875 from FEATFLOW solution
    rdc = 1.4903278329e-01
    rlc = -2.3003337267e-04

    errors = [abs(dc - rdc), abs(Lc - rlc)]

    assert errors[0] < 5e-3, f"Error in drag coefficient {errors[0] / abs(rdc)} exceeds tolerance"
    assert errors[1] < 5e-5, f"Error in lift coefficient {errors[1] / abs(rlc)} exceeds tolerance"
