"""
The Taylor-Green order study of :mod:`pySDC.projects.StroemungsRaum.run_Navier_Stokes_TaylorGreen_FEniCS` with the
Firedrake port of the problem class, :class:`NavierStokes2DTaylorGreenFiredrake`.
"""

import firedrake as fd
import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass
from pySDC.projects.StroemungsRaum.convergence_controllers.prepare_step import PrepareStep
from pySDC.projects.StroemungsRaum.problem_classes.NavierStokes_2D_TaylorGreen_monolithic_Firedrake import (
    NavierStokes2DTaylorGreenFiredrake,
    zero_mean_pressure,
)


def setup(
    t0=0.0,
    dt=0.1,
    periodic=False,
    differentiated_bc=False,
    nelems=24,
    nu=0.1,
    num_nodes=4,
    maxiter=40,
    restol=2e-9,
    newton_tol=1e-13,
):
    """
    The parameters of the order study. The residual tolerance is in the norm dual to L2 that the Firedrake data types
    measure residuals in; 2e-9 is well below the time errors the study resolves.

    Args:
        t0 (float): Initial time
        dt (float): Step size
        periodic (bool): Periodic instead of time-dependent Dirichlet boundaries in x
        differentiated_bc (bool): Impose the time-dependent boundary data in differentiated form
        nelems (int): Number of squares per direction
        nu (float): Kinematic viscosity
        num_nodes (int): Number of collocation nodes
        maxiter (int): Maximum number of SDC iterations
        restol (float): Residual tolerance
        newton_tol (float): Absolute and relative tolerance of the Newton solves, below `restol`

    Returns:
        description and controller_params dictionaries, and t0
    """
    description = {
        'problem_class': NavierStokes2DTaylorGreenFiredrake,
        'problem_params': {
            'nelems': nelems,
            'order': 2,
            'nu': nu,
            'periodic': periodic,
            'differentiated_bc': differentiated_bc,
            'solver_parameters': {'snes_atol': newton_tol, 'snes_rtol': newton_tol},
        },
        'sweeper_class': generic_implicit_mass,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': num_nodes, 'QI': 'LU'},
        'level_params': {'restol': restol, 'dt': dt},
        'step_params': {'maxiter': maxiter},
    }
    if differentiated_bc:
        description['convergence_controllers'] = {PrepareStep: {}}
    return description, {'logger_level': 30}, t0


def run_simulation(description, controller_params, t0, Tend):
    """
    Run the simulation.

    Returns:
        The problem, the statistics and the solution at `Tend`
    """
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
    P = controller.MS[0].levels[0].prob
    uend, stats = controller.run(u0=P.u_exact(t0), t0=t0, Tend=Tend)
    return P, stats, uend


def relative_errors(u, uref):
    """
    Relative L2 errors in velocity and pressure between two solutions of identically built problems, with the
    pressures compared up to their means, which the problem leaves free. Separate runs build separate (identical)
    meshes, so the difference is taken on the values.

    Returns:
        tuple of float: relative L2 error in velocity and in pressure
    """
    u, uref = zero_mean_pressure(u), zero_mean_pressure(uref)
    difference = fd.Function(uref.function_space())
    for d, a, b in zip(difference.dat.data_wo, u.dat.data_ro, uref.dat.data_ro, strict=True):
        d[:] = a - b
    pairs = zip(difference.subfunctions, uref.subfunctions, strict=True)
    return tuple(fd.norm(d) / fd.norm(r) for d, r in pairs)


def run_postprocessing(P, uend, Tend):
    """
    Relative L2 errors in velocity and pressure against the exact solution at `Tend`.

    Returns:
        tuple of float: relative L2 error in velocity and in pressure
    """
    return relative_errors(uend, P.u_exact(Tend))


def order_study(dts, Tend, **kwargs):
    """
    Observed temporal order by comparing consecutive step sizes with each other on the same mesh, which cancels the
    spatial error (it dominates the error against the exact solution on any affordable mesh).

    Args:
        dts (list of float): Step sizes, largest first, each half the previous
        Tend (float): Final time, an integer multiple of every step size
        kwargs: Passed on to :func:`setup`, e.g. `periodic` or `differentiated_bc`

    Returns:
        The step sizes the errors belong to (one fewer than `dts`), and the relative velocity and pressure errors
    """
    solutions = []
    for dt in dts:
        description, controller_params, t0 = setup(dt=dt, **kwargs)
        solutions.append(run_simulation(description, controller_params, t0, Tend)[2])
    errors = [relative_errors(u, ref) for u, ref in zip(solutions[:-1], solutions[1:], strict=True)]
    return dts[:-1], [e[0] for e in errors], [e[1] for e in errors]


def observed_order(dts, errors):
    """Observed order of convergence between consecutive step sizes, one fewer than the inputs."""
    return [np.log(errors[i] / errors[i + 1]) / np.log(dts[i] / dts[i + 1]) for i in range(len(dts) - 1)]


def main():
    """
    Run the order study for the three boundary treatments. RADAU-RIGHT with M nodes has design order 2M-1 and drops to
    M+1 with time-dependent boundary data on a stiff problem, so M = 4 shows the gap, which the differentiated
    boundary conditions mostly close.
    """
    Tend, dts = 0.2, [0.2, 0.1, 0.05, 0.025]
    cases = [
        ('periodic', dict(periodic=True)),
        ('time-dependent Dirichlet', dict(periodic=False)),
        ('time-dependent Dirichlet, differentiated', dict(periodic=False, differentiated_bc=True)),
    ]
    for label, kwargs in cases:
        dts_out, errors_u, errors_p = order_study(dts, Tend, **kwargs)
        print(f'\n{label} boundary conditions in x:')
        print(f'{"dt":>10} {"err(u)":>12} {"order(u)":>9} {"err(p)":>12} {"order(p)":>9}')
        orders_u = [None] + observed_order(dts_out, errors_u)
        orders_p = [None] + observed_order(dts_out, errors_p)
        for dt, eu, ou, ep, op in zip(dts_out, errors_u, orders_u, errors_p, orders_p, strict=True):
            su = '     --- ' if ou is None else f'{ou:9.2f}'
            sp = '     --- ' if op is None else f'{op:9.2f}'
            print(f'{dt:10.5f} {eu:12.4e} {su} {ep:12.4e} {sp}')


if __name__ == '__main__':
    main()
