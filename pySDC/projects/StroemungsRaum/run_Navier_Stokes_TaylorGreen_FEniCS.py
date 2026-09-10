import dolfin as df
import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.StroemungsRaum.problem_classes.NavierStokes_2D_TaylorGreen_monolithic_FEniCS import (
    fenics_NSE_2D_TaylorGreen,
)
from pySDC.projects.StroemungsRaum.sweepers.generic_implicit_mass import (
    generic_implicit_mass,
    generic_implicit_mass_diffbc,
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
    restol=1e-12,
):
    """
    Helper routine to set up parameters

    Args:
        t0: float,
            initial time
        dt: float,
            time step size
        periodic: bool,
            use periodic instead of time-dependent Dirichlet conditions in x
        differentiated_bc: bool,
            impose the time-dependent boundary data in differentiated form, which recovers
            the order it otherwise costs; requires periodic=False
        nelems: int,
            number of elements per spatial direction
        nu: float,
            kinematic viscosity
        num_nodes: int,
            number of collocation nodes
        maxiter: int,
            maximum number of SDC iterations
        restol: float,
            residual tolerance

    Returns:
        description: dict,
            pySDC description dictionary containing problem and method parameters.
        controller_params: dict,
            Parameters for the pySDC controller.
    """
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = restol
    level_params['dt'] = dt

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = maxiter

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = num_nodes
    sweeper_params['QI'] = 'LU'

    # initialize problem parameters
    problem_params = dict()
    problem_params['nelems'] = nelems
    problem_params['t0'] = t0
    problem_params['order'] = 2
    problem_params['nu'] = nu
    problem_params['periodic'] = periodic
    problem_params['differentiated_bc'] = differentiated_bc
    problem_params['Sol_tol'] = 1e-13

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30

    # Fill description dictionary
    description = dict()
    description['problem_class'] = fenics_NSE_2D_TaylorGreen
    description['sweeper_class'] = generic_implicit_mass_diffbc if differentiated_bc else generic_implicit_mass
    description['problem_params'] = problem_params
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    return description, controller_params


def run_simulation(description, controller_params, Tend):
    """
    Run the time integration for the 2D Taylor-Green Navier-Stokes benchmark.

    Args:
        description: dict,
            pySDC problem and method description.
        controller_params: dict,
            Parameters for the pySDC controller.
        Tend: float,
            Final simulation time.

    Returns:
        P: problem instance,
            Problem instance holding the function spaces and the exact solution.
        stats: dict,
            Collected runtime statistics.
        uend: dtype_u,
            Final solution at time Tend.
    """
    t0 = description['problem_params']['t0']

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    P = controller.MS[0].levels[0].prob
    uend, stats = controller.run(u0=P.u_exact(t0), t0=t0, Tend=Tend)

    return P, stats, uend


def relative_errors(u, uref):
    """
    Relative L2 errors in velocity and pressure between two solutions on the same space.

    Args:
        u: dtype_u,
            Numerical solution.
        uref: dtype_u,
            Reference solution.

    Returns:
        tuple of float: relative L2 error in velocity and in pressure.
    """
    un, pn = u.values.split(deepcopy=True)
    ur, pr = uref.values.split(deepcopy=True)

    return (
        df.errornorm(ur, un, 'L2') / df.norm(ur, 'L2'),
        df.errornorm(pr, pn, 'L2') / df.norm(pr, 'L2'),
    )


def run_postprocessing(P, uend, Tend):
    """
    Compute relative L2 errors between the numerical and the exact solution at the final time.

    Args:
        P: problem instance,
            Problem instance holding the exact solution.
        uend: dtype_u,
            Final solution at time Tend.
        Tend: float,
            Final simulation time.

    Returns:
        tuple of float: relative L2 error in velocity and in pressure.
    """
    return relative_errors(uend, P.u_exact(Tend))


def order_study(dts, Tend, dt_ref=None, periodic=False, **kwargs):
    r"""
    Measure the observed temporal order of convergence.

    Errors are *not* taken against the exact solution: the spatial discretization error
    dominates it for any affordable mesh, which hides the temporal order completely. Instead
    two variants are offered, both of which cancel the spatial error exactly because every run
    uses the same mesh:

    - ``dt_ref`` given: compare against a reference run with that much smaller step size,
    - ``dt_ref`` omitted: compare consecutive step sizes with each other (Richardson). This
      needs no reference run and is therefore a lot cheaper, at the cost of one order estimate.

    Args:
        dts: list of float,
            Step sizes to run, largest first, each one half of the previous.
        Tend: float,
            Final simulation time; must be an integer multiple of every step size.
        dt_ref: float,
            Step size for the reference run, or ``None`` to compare consecutive step sizes.
        periodic: bool,
            Use periodic instead of time-dependent Dirichlet conditions in x.
        kwargs:
            Passed on to :func:`setup`.

    Returns:
        dts_out: list of float,
            Step sizes the errors belong to; one shorter than ``dts`` without a reference.
        errors_u: list of float,
            Relative L2 velocity error per step size.
        errors_p: list of float,
            Relative L2 pressure error per step size.
    """
    solutions = []
    for dt in dts:
        description, controller_params = setup(dt=dt, periodic=periodic, **kwargs)
        solutions.append(run_simulation(description, controller_params, Tend)[2])

    if dt_ref is None:
        pairs = list(zip(solutions[:-1], solutions[1:], strict=True))
        dts_out = dts[:-1]
    else:
        description, controller_params = setup(dt=dt_ref, periodic=periodic, **kwargs)
        uref = run_simulation(description, controller_params, Tend)[2]
        pairs = [(u, uref) for u in solutions]
        dts_out = list(dts)

    errors = [relative_errors(u, ref) for u, ref in pairs]

    return dts_out, [e[0] for e in errors], [e[1] for e in errors]


def observed_order(dts, errors):
    """
    Observed order of convergence between consecutive step sizes.

    Args:
        dts: list of float,
            Step sizes.
        errors: list of float,
            Corresponding errors.

    Returns:
        list of float: observed orders, one shorter than the inputs.
    """
    return [np.log(errors[i] / errors[i + 1]) / np.log(dts[i] / dts[i + 1]) for i in range(len(dts) - 1)]


def main():
    r"""
    Run the order study for both boundary condition variants and report the observed orders.

    RADAU-RIGHT with M nodes has design order :math:`2M-1` and, on a stiff problem with
    time-dependent boundary data, drops to the stiff order :math:`M+1`. The gap the benchmark
    can show is therefore :math:`M-2`, and **nothing at all is visible for M = 2**, where the
    two coincide at 3. M = 4 is used here because it is the cheapest setting that makes the
    reduction unmistakable: order 7 against 5 in the pressure.
    """
    Tend = 0.2
    dts = [0.2, 0.1, 0.05, 0.025]

    cases = [
        ('periodic', dict(periodic=True)),
        ('time-dependent Dirichlet', dict(periodic=False)),
        ('time-dependent Dirichlet, differentiated', dict(periodic=False, differentiated_bc=True)),
    ]

    results = {}
    for label, kwargs in cases:
        dts_out, errors_u, errors_p = order_study(dts, Tend, **kwargs)
        results[label] = (dts_out, errors_u, errors_p)

        print(f'\n{label} boundary conditions in x:')
        print(f'{"dt":>10} {"err(u)":>12} {"order(u)":>9} {"err(p)":>12} {"order(p)":>9}')
        orders_u = [None] + observed_order(dts_out, errors_u)
        orders_p = [None] + observed_order(dts_out, errors_p)
        for dt, eu, ou, ep, op in zip(dts_out, errors_u, orders_u, errors_p, orders_p, strict=True):
            su = '     --- ' if ou is None else f'{ou:9.2f}'
            sp = '     --- ' if op is None else f'{op:9.2f}'
            print(f'{dt:10.5f} {eu:12.4e} {su} {ep:12.4e} {sp}')

    return results


if __name__ == "__main__":
    main()
