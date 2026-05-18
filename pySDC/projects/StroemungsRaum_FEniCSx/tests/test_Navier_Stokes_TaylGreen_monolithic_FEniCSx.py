import pytest
import numpy as np
import ufl
import dolfinx as dfx
import dolfinx.fem.petsc

from pySDC.projects.StroemungsRaum_FEniCSx.problem_classes.NavierStokes_2D_TaylGreen_monolithic_FEniCSx import (
     fenicsx_NSE_mass, 
     fenicsx_NSE_periodic_mass,
     )

@pytest.mark.fenicsx
def test_solve_system():
#
    # Physical and numerical parameters for the test case
    nu = 0.02
    t = 0.0
    factor = 0.01

    # Create the 2D monolithic Navier-Stokes problem for both non-periodic and periodic cases
    prob = fenicsx_NSE_mass(t0=0.0, nelems=32, family='CG', order=2, nu=nu)
    prob_periodic = fenicsx_NSE_periodic_mass(t0=0.0, nelems=32, family='CG', order=2, nu=nu)

    # Evaluate the source term at the given time
    g = prob.source_term(t)

    # get the exact solution at time t 
    wex = dfx.fem.Function(prob.W)
    wex.x.array[:] = prob.u_exact(t)[:]

    # Split the exact solution into velocity and pressure components
    u, p = ufl.split(wex)
    
    # assemble the right-hand side vector
    rhs_f =  ufl.dot(u, prob.v) * ufl.dx
    rhs_f += factor * ufl.dot(ufl.dot(u, ufl.nabla_grad(u)), prob.v) * ufl.dx
    rhs_f += factor * ufl.inner(prob.nu * ufl.nabla_grad(u), ufl.nabla_grad(prob.v)) * ufl.dx
    rhs_f -= factor * ufl.dot(p, ufl.div(prob.v)) * ufl.dx
    rhs_f -= factor * ufl.dot(g, prob.v) * ufl.dx
    rhs_f -= factor * ufl.dot(ufl.div(u), prob.q) * ufl.dx

    b = dolfinx.fem.petsc.assemble_vector(dfx.fem.form(rhs_f))
    b.assemble()
    #b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        

    # Solve the linear system for both non-periodic and periodic cases
    w = prob.solve_system(rhs=b, factor=factor, u0=prob.dtype_u(prob.init), t=t)
    w_periodic = prob_periodic.solve_system(rhs=b, factor=factor, u0=prob.dtype_u(prob.init), t=t)
    
    # get the exact solution at time t for error computation
    wx = prob.dtype_u(prob.init)
    wx[:] = wex.x.array[:]

    # Compute the relative monolithic velocity–pressure error for both non-periodic and periodic cases
    rel_err = abs(wx - w) / abs(wx)
    rel_err_periodic = abs(wx - w_periodic) / abs(wx)
    assert rel_err < 1e-5, (f"Relative monolithic velocity–pressure error {rel_err:.3e} exceeds tolerance")
    assert rel_err_periodic < 1e-5, (
    f"Relative monolithic velocity–pressure error (periodic) {rel_err_periodic:.3e} exceeds tolerance")


@pytest.mark.fenicsx
def test_eval_f():

    # parameters for the test case
    nu = 0.2
    t = 0.0

    # create the 2D monolithic Navier-Stokes problem 
    prob = fenicsx_NSE_mass(t0=0.0, nelems=64, family='CG', order=2, nu=nu)

    # get the exact solution at time t
    wex = dfx.fem.Function(prob.W)
    wex.sub(0).interpolate(lambda x: 
     np.vstack((-1.0 *np.cos(x[0])*np.sin(x[1])*np.exp(-2.0*nu*t),np.sin(x[0])*np.cos(x[1])*np.exp(-2.0*nu*t))))        
    wex.sub(1).interpolate( lambda x: -0.25*(np.cos(2.0*x[0]) + np.cos(2.0*x[1]))*np.exp(-4.0*nu*t))

    # evaluate the right-hand side vector using the problem's eval_f method
    f = prob.eval_f(w=wex.x.array[:], t=t)

    # compute the expected right-hand side vector by applying the mass matrix to the exact solution
    fw = dfx.fem.Function(prob.W)
    fw.sub(0).interpolate(lambda x: np.vstack((
                    2.0*nu*np.cos(x[0])*np.sin(x[1])*np.exp(-2.0*nu*t)
                    + np.pi / 34 * np.exp(-16 * np.pi**2 * nu * t) * np.sin(4 * np.pi * (t - x[0]))
                    * np.cos(np.pi * x[1]) * (32 - 17*np.cos(np.pi*x[1])),
                    #
                    -2.0*nu*np.sin(x[0])*np.cos(x[1])*np.exp(-2.0*nu*t)  
                    + 2 * np.pi**2 * nu * np.exp(-8 * np.pi**2 * nu * t) * np.cos(2 * np.pi * (t - x[0])) 
                    - np.pi * np.exp(-16 * np.pi**2 * nu * t) * np.sin(np.pi * x[1]) * (2 * np.cos(np.pi \
                    * x[1])**3 + 4/17 * np.cos(4 * np.pi * (t - x[0])))
                                            )))
    fw.sub(1).interpolate(lambda x:  np.full(x.shape[1], 0.0, dtype=np.float64))
    
    # apply the mass matrix to obtain the expected right-hand side vector
    Mfw = dfx.fem.Function(prob.W)
    prob.M.mult(fw.x.petsc_vec, Mfw.x.petsc_vec)

    # convert the expected right-hand side vector to a numpy array for comparison
    fwx  = prob.dtype_u(prob.init)
    fwx[:] = Mfw.x.array[:]
    
    # apply boundary conditions to both vectors before comparison
    prob.fix_residual(f)
    prob.fix_residual(fwx)
    
    # compute the relative error between computed and expected right-hand sides
    rel_err = abs(f - fwx) / abs(fwx)
    print(f"Relative error in right-hand side evaluation: {rel_err:.3e}")
    assert rel_err < 1e-5, f"Relative error {rel_err} exceeds tolerance"
    

@pytest.mark.fenicsx
def test_problem_class():

    from pySDC.projects.StroemungsRaum_FEniCSx.run_Navier_Stokes_equations_TaylGreen_monolithic_FEniCSx import (
        setup,
        run_simulation,
        run_postprocessing
    )

    t0 = 0.0
    Tend = 0.1

    # -- Non-periodic case --
    # run the setup to get description and controller parameters
    description, controller_params = setup(t0=t0, periodic=False)

    # run the simulation and get the problem, stats and final solution
    problem, stats, uend = run_simulation(description, controller_params, Tend)

    # run postprocessing to save parameters and solutions for visualization
    rel_error_u, rel_error_p = run_postprocessing(problem, uend, Tend)
    
    print(f"Relative L2 error in velocity: {rel_error_u:.3e}")
    print(f"Relative L2 error in pressure: {rel_error_p:.3e}")
    # assert that the relative errors in velocity and pressure are within acceptable limits
    assert rel_error_u < 1e-2, f"Error in velocity error {rel_error_u} exceeds tolerance"
    assert rel_error_p < 6e-2, f"Error in pressure error {rel_error_p} exceeds tolerance"

    # -- Periodic case --
    # run the setup to get description and controller parameters
    description, controller_params = setup(t0=t0, periodic=True)

    # run the simulation and get the problem, stats and final solution
    problem, stats, uend = run_simulation(description, controller_params, Tend)

    # run postprocessing to save parameters and solutions for visualization
    rel_error_u, rel_error_p = run_postprocessing(problem, uend, Tend)
    
    # assert that the relative errors in velocity and pressure are within acceptable limits
    assert rel_error_u < 1e-2, f"Error in velocity error (periodic bc) {rel_error_u} exceeds tolerance"
    assert rel_error_p < 1e-2, f"Error in pressure error (periodic bc) {rel_error_p} exceeds tolerance"