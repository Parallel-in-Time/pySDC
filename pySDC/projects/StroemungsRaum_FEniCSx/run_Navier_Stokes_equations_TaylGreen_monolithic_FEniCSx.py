import sys
import numpy as np
import dolfinx as dfx
import ufl
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.StroemungsRaum.sweepers.generic_implicit_mass import generic_implicit_mass
from pySDC.projects.StroemungsRaum_FEniCSx.problem_classes.NavierStokes_2D_TaylGreen_monolithic_FEniCSx import (
    fenicsx_NSE_mass, 
    fenicsx_NSE_periodic_mass
    )

def setup(t0=0, periodic=False):
    """
    Helper routine to set up parameters

    Args:
        t0: float,
            initial time
        periodic: bool,
            whether to use periodic boundary conditions or not

    Returns:
        description: dict,
            pySDC description dictionary containing problem and method parameters.
        controller_params: dict,
            Parameters for the pySDC controller.
    """
    # time step size
    dt = 0.1

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-10
    level_params['dt'] = dt

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 15

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = [2]
    sweeper_params['QI'] = ['LU']

    problem_params = dict()
    problem_params['nu'] = 0.2
    problem_params['t0'] = t0  
    problem_params['nelems'] = [64]
    problem_params['family'] = 'CG'
    problem_params['order'] = [2]

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20 

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    if periodic:
        description['problem_class'] = fenicsx_NSE_periodic_mass 
    else:
        description['problem_class'] = fenicsx_NSE_mass 
    description['sweeper_class'] = generic_implicit_mass
    description['problem_params'] = problem_params
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    
    return description, controller_params


def run_simulation(description, controller_params, Tend):
    """
    Run the time integration for the 2D Navier-Stokes equations.

     Args:
        description: dict,
            pySDC problem and method description.
        controller_params: dict,
            Parameters for the pySDC controller.
        Tend: float,
            Final simulation time.

    Returns:
        P: problem instance,
           Problem instance containing the final solution and other problem-related information.
        stats: dict,
           collected runtime statistics,
        uend: FEniCS function,
           Final solution at time Tend.
    """
    # get initial time from description
    t0 = description['problem_params']['t0']

    # quickly generate block of steps
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    
    return P, stats, uend

def run_postprocessing(P, uend, Tend):
    """
    Postprocess and store simulation results for visualization and analysis.

    Args:
        P: Problem instance,
            Problem instance containing the final solution and other problem-related information.
        uend: FEniCS function,
            Final solution at time Tend.
        Tend: float,
            Final simulation time.

    Returns: 
        rel_error_u: float,
            Relative L2 error in velocity compared to the exact solution.
        rel_error_p: float,
            Relative L2 error in pressure compared to the exact solution.
    """
    wx = dfx.fem.Function(P.W)
    wn = dfx.fem.Function(P.W)
    #
    wx.x.array[:] = P.u_exact(Tend)[:]
    wn.x.array[:] = uend[:]
    #
    un, pn = wn.split()
    un_ = un.collapse()
    pn_ = pn.collapse()
    #
    ue, pe = wx.split()
    ue_ = ue.collapse()
    pe_ = pe.collapse()
    # 
    error_u = np.sqrt(dfx.fem.assemble_scalar(dfx.fem.form(ufl.dot(un_ - ue_, un_ - ue_) * ufl.dx)))
    error_p = np.sqrt(dfx.fem.assemble_scalar(dfx.fem.form((pn_ - pe_) * (pn_ - pe_) * ufl.dx)))
    
    # 
    norm_u = np.sqrt(dfx.fem.assemble_scalar(dfx.fem.form(ufl.dot(ue_, ue_) * ufl.dx)))
    norm_p = np.sqrt(dfx.fem.assemble_scalar(dfx.fem.form(pe_ * pe_ * ufl.dx)))

    #
    rel_error_u = error_u / norm_u
    rel_error_p = error_p / norm_p 
    
    #
    print(f"L2 error in velocity : {error_u:.3e} (relative: {rel_error_u:.3e})")
    print(f"L2 error in pressure : {error_p:.3e} (relative: {rel_error_p:.3e})")

    return rel_error_u, rel_error_p
  

if __name__ == "__main__":

    t0 = 0.0
    Tend =0.1

    # run the setup to get description and controller parameters
    description, controller_params = setup(t0=t0, periodic=True)

    # run the simulation and get the problem, stats and final solution
    problem, stats, uend = run_simulation(description, controller_params, Tend)

    # run postprocessing to save parameters and solutions for visualization
    rel_error_u, rel_error_p = run_postprocessing(problem, uend, Tend)