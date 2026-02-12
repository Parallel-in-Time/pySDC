import numpy as np
import pytest
import dolfin as df

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pySDC.implementations.datatype_classes.fenics_mesh import fenics_mesh
from pySDC.projects.StroemungsRaum.problem_classes.ConvectionDiffusion_2D_FEniCS import fenics_ConvDiff2D_mass


@pytest.mark.fenics
def test_solve_system():

    # Physical and numerical parameters used in the test
    s = 0.05
    nu = 0.01
    x0 = -0.25
    y0 = 0.0
    t = 0.1
    factor = 0.05

    # Instantiate the convection-diffusion problem
    prob = fenics_ConvDiff2D_mass(c_nvars=64, t0=0.0, family='CG', order=2, nu=nu, sigma=s)

    # Initial guess for the solver (not used by the direct solver, but required by the interface)
    u0 = fenics_mesh(prob.V)

    # Define a manufactured right-hand side such that the exact solution
    # satisfies (M - factor * K) u = rhs
    g = df.Expression(
        'pow(s,2)/(pow(s,2)+4*nu*t)*exp(-( '
        'pow((cos(4*t)*x[0]+sin(4*t)*x[1]-x0),2)'
        '+pow((-sin(4*t)*x[0]+cos(4*t)*x[1]-y0),2)'
        ')/(pow(s,2)+4*nu*t))'
        '*(1 - nu * factor*( '
        '4*(pow((cos(4*t)*x[0]+sin(4*t)*x[1]-x0),2)'
        '+pow((-sin(4*t)*x[0]+cos(4*t)*x[1]-y0),2)'
        ')/pow((pow(s,2)+4*nu*t),2)'
        '-4/(pow(s,2)+4*nu*t)'
        '))',
        degree=2,
        s=s,
        nu=nu,
        t=t,
        x0=x0,
        y0=y0,
        factor=factor,
    )

    # Apply the mass matrix to the manufactured forcing term to obtain the RHS
    rhs = prob.apply_mass_matrix(prob.dtype_u(df.interpolate(g, prob.V)))

    # Solve the linear system (M - factor * K) u = rhs
    u = prob.solve_system(rhs=rhs, factor=factor, u0=u0, t=t)

    # Exact solution evaluated at the final time
    uex = prob.u_exact(t)

    # Relative error between numerical and exact solution
    rel_err = abs(uex - u) / abs(uex)

    assert rel_err < 5e-5, f"Relative error {rel_err} exceeds tolerance"


@pytest.mark.fenics
def test_eval_f():

    # Physical and numerical parameters used in the test
    s = 0.05
    nu = 0.01
    x0 = -0.25
    y0 = 0.0
    t = 0.1
    factor = 0.05

    # Instantiate the convection-diffusion problem
    prob = fenics_ConvDiff2D_mass(c_nvars=64, t0=0.0, family='CG', order=2, nu=nu, sigma=s)

    # Use a smooth analytical function as test input for eval_f
    uex = prob.dtype_u(df.interpolate(df.Expression('sin(pi*x[0])*sin(pi*x[1])', degree=2, pi=np.pi), prob.V))

    # Evaluate the split right-hand side at the given state and time
    f = prob.eval_f(u=uex, t=t)

    # ------------------------------------------------------------------
    # Test implicit part: diffusion term
    # f.impl should correspond to M * (nu * Delta u)
    # ------------------------------------------------------------------

    # Analytical Laplacian of the test function multiplied by nu
    lap_u = df.Expression('-2*pow(pi,2)*nu*sin(pi*x[0])*sin(pi*x[1])', degree=2, nu=nu, pi=np.pi)

    # Apply mass matrix to obtain M * (nu * Delta u)
    Mlap_u = prob.apply_mass_matrix(prob.dtype_u(df.interpolate(lap_u, prob.V)))

    # Relative error between computed implicit term and reference
    err_diff = abs(f.impl - Mlap_u) / abs(Mlap_u)
    assert err_diff < 5e-4, f"Relative error {err_diff} exceeds tolerance"

    # ------------------------------------------------------------------
    # Test explicit part: convection term
    # Since g = 0 in the problem class, f.expl should reduce to the convection contribution
    # ------------------------------------------------------------------

    # Analytical expression for U · grad(u) with U = (-4y, 4x)
    Ugrad_u = df.Expression(
        "-4*pi*( x[0]*sin(pi*x[0])*cos(pi*x[1]) - x[1]*cos(pi*x[0])*sin(pi*x[1]) )", degree=5, pi=np.pi
    )

    # Apply mass matrix to obtain M * (U · grad(u))
    MUgad_u = prob.apply_mass_matrix(prob.dtype_u(df.interpolate(Ugrad_u, prob.V)))

    # Relative error between computed explicit term and reference
    err_conv = abs(f.expl - MUgad_u) / abs(MUgad_u)
    assert err_conv < 5e-4, f"Relative error {err_conv} exceeds tolerance"


@pytest.mark.fenics
def test_problem_class():

    from pySDC.projects.StroemungsRaum.run_convection_diffusion_equation_FEniCS import setup, run_simulation

    t0 = 0.0
    Tend = 0.1

    # run the setup to get description and controller parameters
    description, controller_params = setup(t0=t0)

    # run the simulation and get the problem, stats and relative error
    _, _, rel_err = run_simulation(description, controller_params, Tend)

    assert rel_err <= 5e-4, 'ERROR: Relative error is too high, got rel_err = {}'.format(rel_err)
