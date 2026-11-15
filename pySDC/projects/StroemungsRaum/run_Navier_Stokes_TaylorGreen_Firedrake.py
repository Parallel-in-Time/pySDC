"""
The Taylor-Green order study of :mod:`pySDC.projects.StroemungsRaum.run_Navier_Stokes_TaylorGreen_FEniCS` with the
Firedrake port of the problem class, :class:`NavierStokes2DTaylorGreenFiredrake`. The ``differentiated_bc`` option of
the FEniCS version is not ported.
"""

import firedrake as fd
import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.generic_implicit_mass import generic_implicit_mass
from pySDC.projects.StroemungsRaum.problem_classes.NavierStokes_2D_TaylorGreen_monolithic_Firedrake import (
    NavierStokes2DTaylorGreenFiredrake,
)


def setup(t0=0.0, dt=0.1, periodic=False, nelems=24, nu=0.1, num_nodes=4, maxiter=40, restol=1e-11, newton_tol=1e-13):
    """
    The parameters of the FEniCS version, with the residual tolerance in the dual norm the Firedrake data types use.

    Args:
        t0 (float): Initial time
        dt (float): Step size
        periodic (bool): Periodic instead of time-dependent Dirichlet boundaries in x
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
            'solver_parameters': {'snes_atol': newton_tol, 'snes_rtol': newton_tol},
        },
        'sweeper_class': generic_implicit_mass,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': num_nodes, 'QI': 'LU'},
        'level_params': {'restol': restol, 'dt': dt},
        'step_params': {'maxiter': maxiter},
    }
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
    Relative L2 errors in velocity and pressure between two solutions of identically built problems. Separate runs
    build separate (identical) meshes, so the difference is taken on the values.

    Returns:
        tuple of float: relative L2 error in velocity and in pressure
    """
    difference = fd.Function(uref.function_space())
    for d, a, b in zip(difference.dat.data_wo, u.dat.data_ro, uref.dat.data_ro, strict=True):
        d[:] = a - b
    pairs = zip(difference.subfunctions, uref.functionspace.subfunctions, strict=True)
    return tuple(fd.norm(d) / fd.norm(r) for d, r in pairs)


def run_postprocessing(P, uend, Tend):
    """
    Relative L2 errors in velocity and pressure against the exact solution at `Tend`.

    Returns:
        tuple of float: relative L2 error in velocity and in pressure
    """
    return relative_errors(uend, P.u_exact(Tend))


def order_study(dts, Tend, periodic=False, **kwargs):
    """
    Observed temporal order by comparing consecutive step sizes with each other on the same mesh, which cancels the
    spatial error (it dominates the error against the exact solution on any affordable mesh).

    Args:
        dts (list of float): Step sizes, largest first, each half the previous
        Tend (float): Final time, an integer multiple of every step size
        periodic (bool): Periodic instead of time-dependent Dirichlet boundaries in x
        kwargs: Passed on to :func:`setup`

    Returns:
        The step sizes the errors belong to (one fewer than `dts`), and the relative velocity and pressure errors
    """
    solutions = []
    for dt in dts:
        description, controller_params, t0 = setup(dt=dt, periodic=periodic, **kwargs)
        solutions.append(run_simulation(description, controller_params, t0, Tend)[2])
    errors = [relative_errors(u, ref) for u, ref in zip(solutions[:-1], solutions[1:], strict=True)]
    return dts[:-1], [e[0] for e in errors], [e[1] for e in errors]


def observed_order(dts, errors):
    """Observed order of convergence between consecutive step sizes, one fewer than the inputs."""
    return [np.log(errors[i] / errors[i + 1]) / np.log(dts[i] / dts[i + 1]) for i in range(len(dts) - 1)]


def main():
    """
    Run the order study for both boundary types. RADAU-RIGHT with M nodes has design order 2M-1 and drops to M+1 with
    time-dependent boundary data on a stiff problem, so M = 4 shows the gap: 7 against 5 in the pressure.
    """
    Tend, dts = 0.2, [0.2, 0.1, 0.05, 0.025]
    for label, periodic in [('periodic', True), ('time-dependent Dirichlet', False)]:
        dts_out, errors_u, errors_p = order_study(dts, Tend, periodic=periodic)
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
