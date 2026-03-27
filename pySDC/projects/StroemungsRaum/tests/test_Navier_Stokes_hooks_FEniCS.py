import pytest
import numpy as np
import dolfin as df

from pySDC.projects.StroemungsRaum.run_Navier_Stokes_equations_FEniCS import setup, run_simulation
from pySDC.helpers.stats_helper import get_sorted


@pytest.mark.fenics
def test_LiftDrag_hook():
    from pySDC.projects.StroemungsRaum.hooks.hooks_NSE_IMEX_FEniCS import LogLiftDrag

    t0 = 3.125e-04
    Tend = 0.005

    # run the setup to get description and controller parameters
    description, controller_params = setup(t0=t0)

    # change the hook class to LogLiftDrag
    controller_params['hook_class'] = [LogLiftDrag]

    # run the simulation and get the problem, stats and relative error
    _, stats, _ = run_simulation(description, controller_params, Tend)

    lift = get_sorted(stats, type='lift_post_step', sortby='time')
    drag = get_sorted(stats, type='drag_post_step', sortby='time')

    # reference values at t = 0.0046875 from FEATFLOW solution
    rdc = 1.4903278329e-01
    rlc = -2.3003337267e-04

    errors = [abs(drag[-1][1] - rdc), abs(lift[-1][1] - rlc)]

    assert errors[0] < 5e-3, f"Error in drag coefficient {errors[0]} exceeds tolerance"
    assert errors[1] < 5e-5, f"Error in lift coefficient {errors[1]} exceeds tolerance"


def test_WriteSolutions_hook():
    import os
    from pySDC.projects.StroemungsRaum.hooks.hooks_NSE_IMEX_FEniCS import LogWriteSolutions

    t0 = 3.125e-04
    Tend = 0.005

    # run the setup to get description and controller parameters
    description, controller_params = setup(t0=t0)

    # change the hook class to LogWriteSolutions
    controller_params['hook_class'] = [LogWriteSolutions]

    # run the simulation and get the problem, stats and relative error
    prob, stats, uend = run_simulation(description, controller_params, Tend)

    # Define variables
    un = df.Function(prob.V)
    pn = df.Function(prob.Q)

    # Open XDMF files for reading the stored solutions
    path = f"{os.path.dirname(__file__)}/../data/navier_stokes/"
    xdmffile_u = df.XDMFFile(path + 'Cylinder_velocity.xdmf')
    xdmffile_p = df.XDMFFile(path + 'Cylinder_pressure.xdmf')

    # number of time steps
    num_steps = int((Tend - t0) / description['level_params']['dt'])

    # Read the last checkpoint from XDMF files
    xdmffile_u.read_checkpoint(un, 'un', num_steps - 1)
    xdmffile_p.read_checkpoint(pn, 'pn', num_steps - 1)

    # normal pointing out of obstacle
    n = -df.FacetNormal(prob.V.mesh())

    # tangential velocity component at the interface of the obstacle
    u_t = df.inner(df.as_vector((n[1], -n[0])), un)

    # compute the drag and lift coefficients
    drag = df.Form(2 / 0.1 * (prob.nu * df.inner(df.grad(u_t), n) * n[1] - pn * n[0]) * prob.dsc)
    lift = df.Form(-2 / 0.1 * (prob.nu * df.inner(df.grad(u_t), n) * n[0] + pn * n[1]) * prob.dsc)

    # assemble the scalar values
    dc = df.assemble(drag)
    Lc = df.assemble(lift)

    # reference values at t = 0.0046875 from FEATFLOW solution
    rdc = 1.4903278329e-01
    rlc = -2.3003337267e-04

    errors = [abs(dc - rdc), abs(Lc - rlc)]

    assert errors[0] < 5e-3, f"Error in drag coefficient {errors[0]} exceeds tolerance"
    assert errors[1] < 5e-5, f"Error in lift coefficient {errors[1]} exceeds tolerance"
