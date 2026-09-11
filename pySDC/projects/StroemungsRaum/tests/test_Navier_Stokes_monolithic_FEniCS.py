import pytest
import dolfin as df

from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh
from pySDC.projects.StroemungsRaum.problem_classes.NavierStokes_2D_monolithic_FEniCS import fenics_NSE_2D_Monolithic


@pytest.mark.fenics
def test_solve_system():

    # Physical and numerical parameters for the test case
    nu = 0.001
    t = 0.0
    factor = 0.01

    # Create the 2D monolithic Navier-Stokes problem
    prob = fenics_NSE_2D_Monolithic(t0=0.0, order=2, nu=nu)

    # Exact Taylor-Green vortex velocity used as manufactured solution
    u_exact = df.Expression(
        ("-cos(x[0])*sin(x[1])*exp(-2.0*nu*t)", " sin(x[0])*cos(x[1])*exp(-2.0*nu*t)"), degree=prob.order, nu=nu, t=t
    )

    # Exact pressure corresponding to the Taylor-Green vortex
    p_exact = df.Expression("-0.25*(cos(2.0*x[0]) + cos(2.0*x[1]))*exp(-4.0*nu*t)", degree=prob.order - 1, nu=nu, t=t)

    # Interpolate the exact velocity and pressure into the mixed space
    wex = df.Function(prob.W)
    df.assign(wex.sub(0), df.interpolate(u_exact, prob.V))
    df.assign(wex.sub(1), df.interpolate(p_exact, prob.Q))
    wex = prob.dtype_u(wex)

    # Manufactured forcing term chosen so that the exact solution satisfies
    # the discrete Navier-Stokes system used in the problem class
    rhs_u = df.Expression(
        (
            "(-1.0-2.0*nu*factor)*cos(x[0])*sin(x[1])*exp(-2.0*nu*t)",
            "(1.0-2.0*nu*factor)*sin(x[0])*cos(x[1])*exp(-2.0*nu*t)",
        ),
        degree=prob.order,
        nu=nu,
        t=t,
        factor=factor,
    )

    # Zero pressure contribution in the right-hand side
    rhs_p = df.Expression("0.0", degree=prob.order - 1, nu=nu, t=t)

    # Interpolate the manufactured forcing term into the mixed space
    g = df.Function(prob.W)
    df.assign(g.sub(0), df.interpolate(rhs_u, prob.V))
    df.assign(g.sub(1), df.interpolate(rhs_p, prob.Q))

    # Dirichlet boundary conditions taken from the exact solution
    bc_u = df.DirichletBC(prob.W.sub(0), u_exact, 'on_boundary')
    bc_p = df.DirichletBC(prob.W.sub(1), p_exact, 'on_boundary')
    prob.bc = [bc_u, bc_p]

    # Apply the mass matrix to obtain the final right-hand side vector
    rhs = prob.apply_mass_matrix(prob.dtype_u(g))

    # Solve the linear system
    w = prob.solve_system(rhs=rhs, factor=factor, u0=fenics_mesh(prob.W), t=t)

    # Compute the relative error with respect to the exact solution
    rel_err = abs(wex - w) / abs(wex)
    assert rel_err < 5e-5, f"Relative error {rel_err} exceeds tolerance"


@pytest.mark.fenics
def test_eval_f():

    # Physical and numerical parameters for the test case
    nu = 0.1
    t = 0.0

    # Create the 2D monolithic Navier-Stokes problem
    prob = fenics_NSE_2D_Monolithic(t0=0.0, order=2, nu=nu)

    # Exact Taylor-Green vortex velocity used as manufactured solution
    u_exact = df.Expression(
        ("-cos(x[0])*sin(x[1])*exp(-2.0*nu*t)", " sin(x[0])*cos(x[1])*exp(-2.0*nu*t)"), degree=prob.order, nu=nu, t=t
    )

    # Exact pressure corresponding to the Taylor-Green vortex
    p_exact = df.Expression("-0.25*(cos(2.0*x[0]) + cos(2.0*x[1]))*exp(-4.0*nu*t)", degree=prob.order - 1, nu=nu, t=t)

    # Interpolate the exact velocity and pressure into the mixed space
    wex = df.Function(prob.W)
    df.assign(wex.sub(0), df.interpolate(u_exact, prob.V))
    df.assign(wex.sub(1), df.interpolate(p_exact, prob.Q))
    wex = prob.dtype_u(wex)

    # Evaluate the right-hand side for the given state and time
    f = prob.eval_f(w=wex, t=t)

    # Analytical value of the right-hand side for the Taylor-Green vortex
    fu_exact = df.Expression(
        ("2.0*nu*cos(x[0])*sin(x[1])*exp(-2.0*nu*t)", "-2.0*nu*sin(x[0])*cos(x[1])*exp(-2.0*nu*t)"),
        degree=prob.order,
        nu=nu,
        t=t,
    )

    # The pressure component of the right-hand side is zero
    fp_exact = df.Expression("0.0", degree=prob.order - 1, nu=nu, t=t)

    # Interpolate the analytical right-hand side into the mixed space
    fex = df.Function(prob.W)
    df.assign(fex.sub(0), df.interpolate(fu_exact, prob.V))
    df.assign(fex.sub(1), df.interpolate(fp_exact, prob.Q))

    # Apply the mass matrix to obtain the expected right-hand side vector
    fex = prob.apply_mass_matrix(prob.dtype_u(fex))

    # Apply boundary conditions to both vectors before comparison
    bc_u = df.DirichletBC(prob.W.sub(0), fu_exact, 'on_boundary')
    bc_p = df.DirichletBC(prob.W.sub(1), fp_exact, 'on_boundary')
    bc = [bc_u, bc_p]

    [bc.apply(f.values.vector()) for bc in bc]
    [bc.apply(fex.values.vector()) for bc in bc]

    # Compute the relative error between computed and exact right-hand sides
    rel_err = abs(f - fex) / abs(fex)
    assert rel_err < 5e-5, f"Relative error {rel_err} exceeds tolerance"


@pytest.mark.fenics
def test_problem_class():

    from pySDC.projects.StroemungsRaum.run_Navier_Stokes_equations_monolithic_FEniCS import (
        setup,
        run_simulation,
    )

    t0 = 3.125e-04
    Tend = 0.01
    rho = 1

    # Initialize the simulation setup and controller parameters
    description, controller_params = setup(t0=t0)

    # Run the simulation up to Tend and retrieve the final solution
    prob, stats, uend = run_simulation(description, controller_params, Tend)

    # Extract velocity and pressure from the mixed solution
    u, p = df.split(uend.values)

    # Outward normal vector on the obstacle boundary
    n = -df.FacetNormal(prob.V.mesh())

    # Tangential component of the velocity on the obstacle boundary
    u_t = df.inner(df.as_vector((n[1], -n[0])), u)

    # Define the drag coefficient functional
    drag = df.Form(2 / 0.1 * (prob.nu / rho * df.inner(df.grad(u_t), n) * n[1] - p * n[0]) * prob.dsc)

    # Define the lift coefficient functional
    lift = df.Form(-2 / 0.1 * (prob.nu / rho * df.inner(df.grad(u_t), n) * n[0] + p * n[1]) * prob.dsc)

    # Evaluate drag and lift coefficients
    dc = df.assemble(drag)
    lc = df.assemble(lift)

    # Reference drag and lift coefficients at t = 0.0103125
    # taken from the FEATFLOW benchmark solution
    rdc = 1.5689678714e-01
    rlc = -2.2719989899e-04

    # Compute errors in drag and lift coefficients
    errors = [abs(dc - rdc), abs(lc - rlc)]

    # Assert that the computed drag and lift coefficients are within the specified tolerances
    assert errors[0] < 5e-2, f"Error in drag coefficient {errors[0]} exceeds tolerance"
    assert errors[1] < 5e-2, f"Error in lift coefficient {errors[1]} exceeds tolerance"
