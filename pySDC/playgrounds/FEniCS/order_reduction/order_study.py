"""
Order-reduction study for SDC with time-dependent Dirichlet boundary conditions.

This standalone script runs a careful temporal convergence study for the
FEniCS-based 1D heat equation and reproduces the classic *order reduction*
phenomenon that appears when Dirichlet boundary data depends on time.

Two problems are compared (both driven with a manufactured forcing term so
that the exact solution is known):

* **Sine case** (``fenics_heat_mass``):
  exact solution ``u(x,t) = sin(pi x) cos(t) + c``.
  Because ``sin(pi*0) = sin(pi*1) = 0``, the Dirichlet boundary values are
  *constant in time* (equal to ``c``). SDC converges with the full
  collocation order ``2M-1`` (RADAU-RIGHT).

* **Cosine case** (``fenics_heat_mass_timebc``):
  exact solution ``u(x,t) = cos(pi x) cos(t) + c``.
  Now ``cos(pi*0) = 1`` and ``cos(pi*1) = -1``, so the Dirichlet boundary
  values ``+cos(t)+c`` and ``-cos(t)+c`` *change in time*. Imposing the BC
  directly on the right-hand side inside ``solve_system`` breaks the
  collocation fixed point and the observed order drops below ``2M-1``.

The script writes a JSON file ``order_study_results.json`` with all raw data,
which the companion ``make_report.py`` turns into a standalone HTML report.

Run with a FEniCS-enabled environment, e.g.::

    micromamba run -n pysdc-fenics python -m pySDC.playgrounds.FEniCS.order_reduction.order_study
"""

import json
import time
from pathlib import Path

import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced import (
    fenics_heat_mass,
    fenics_heat_mass_timebc,
)
from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
from pySDC.playgrounds.FEniCS.order_reduction.problem_classes import fenics_heat_mass_timebc_lift_physical


def build_description(problem_class, num_nodes, dt, t0=0.0, c_nvars=128, nu=0.1, c=0.0, order=4, refinements=1):
    """Assemble the pySDC description and controller-params dictionaries."""
    level_params = {
        'restol': 1e-13,  # iterate to (near) the collocation solution
        'dt': dt,
    }
    step_params = {'maxiter': 100}
    sweeper_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': num_nodes,
    }
    problem_params = {
        'nu': nu,
        't0': t0,
        'c_nvars': c_nvars,
        'family': 'CG',
        'order': order,
        'refinements': refinements,
        'c': c,
    }
    controller_params = {'logger_level': 30}
    description = {
        'problem_class': problem_class,
        'problem_params': problem_params,
        'sweeper_class': imex_1st_order_mass,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
    }
    return description, controller_params


def run_sdc(problem_class, dt, num_nodes=3, t0=0.0, Tend=1.0, **kwargs):
    """Run a single SDC solve and return (relative error, mean #iterations)."""
    from pySDC.helpers.stats_helper import get_sorted

    description, controller_params = build_description(problem_class, num_nodes, dt, t0=t0, **kwargs)
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
    P = controller.MS[0].levels[0].prob
    is_physical_lift = isinstance(P, fenics_heat_mass_timebc_lift_physical)
    uinit = P.u_exact_lifted(t0) if is_physical_lift else P.u_exact(t0)
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    uex = P.u_exact(Tend)
    if is_physical_lift:
        uend = uend + P.lift(Tend)
    err = float(abs(uex - uend) / abs(uex))

    niters = [item[1] for item in get_sorted(stats, type='niter', sortby='time')]
    mean_iter = float(np.mean(niters)) if niters else float('nan')
    return err, mean_iter


def local_orders(dts, errors):
    """Order between successive (dt, error) pairs: log(e1/e2)/log(dt1/dt2)."""
    dts = np.asarray(dts, dtype=float)
    errors = np.asarray(errors, dtype=float)
    return (np.log(errors[:-1] / errors[1:]) / np.log(dts[:-1] / dts[1:])).tolist()


def fitted_order(dts, errors):
    """Least-squares slope of log(error) vs log(dt)."""
    return float(np.polyfit(np.log(dts), np.log(errors), 1)[0])


def study_case(problem_class, label, dts, num_nodes, Tend=1.0, **kwargs):
    """Run the full dt-sweep for one problem class / one M and collect data."""
    print(f"\n>>> {label}  (M={num_nodes})")
    errors, iters, times = [], [], []
    for dt in dts:
        t_start = time.perf_counter()
        err, mean_iter = run_sdc(problem_class, dt, num_nodes=num_nodes, Tend=Tend, **kwargs)
        wall = time.perf_counter() - t_start
        errors.append(err)
        iters.append(mean_iter)
        times.append(wall)
        print(f"    dt={dt:8.5f}  rel.err={err:.4e}  mean_iter={mean_iter:5.2f}  ({wall:5.1f}s)")

    result = {
        'label': label,
        'problem_class': problem_class.__name__,
        'num_nodes': num_nodes,
        'expected_order': 2 * num_nodes - 1,
        'dts': list(dts),
        'errors': errors,
        'mean_iters': iters,
        'wall_times': times,
        'local_orders': local_orders(dts, errors),
        'fitted_order': fitted_order(dts, errors),
    }
    print(f"    -> fitted order = {result['fitted_order']:.2f}  (expected {result['expected_order']})")
    return result


def main():
    # Large-dt asymptotic regime so temporal error dominates the spatial floor.
    # Use a fine mesh (order=4, refinements=1, c_nvars=256 -> 1025 DoFs) to push
    # the spatial error floor down to ~1e-10.
    #
    # NB: Tend must be exactly reachable by all dt values (Tend / dt integer) to
    # avoid a spurious final-step error; Tend=1.0 with dyadic dt satisfies this.
    dts = [0.5 / 2**k for k in range(5)]  # 0.5, 0.25, 0.125, 0.0625, 0.03125
    Tend = 1.0
    common = dict(c_nvars=256, nu=0.1, c=1.0, order=4, refinements=1)

    print("=" * 74)
    print("SDC order-reduction study with time-dependent Dirichlet BCs (FEniCS)")
    print(f"  RADAU-RIGHT, Tend={Tend}, dts={[round(d, 4) for d in dts]}")
    print(f"  space: CG order={common['order']}, refinements={common['refinements']}, "
          f"c_nvars={common['c_nvars']}, nu={common['nu']}, c={common['c']}")
    print("=" * 74)

    cases = []
    for num_nodes in (2, 3):
        cases.append(study_case(fenics_heat_mass, "Sine (constant-in-time BCs)", dts, num_nodes, Tend=Tend, **common))
        cases.append(
            study_case(
                fenics_heat_mass_timebc, "Cosine (time-dependent BCs)", dts, num_nodes, Tend=Tend, **common
            )
        )
        cases.append(
            study_case(
                fenics_heat_mass_timebc_lift_physical,
                "Cosine + internal boundary lifting",
                dts,
                num_nodes,
                Tend=Tend,
                **common,
            )
        )

    payload = {
        'meta': {
            'Tend': Tend,
            'dts': list(dts),
            'quad_type': 'RADAU-RIGHT',
            **common,
        },
        'cases': cases,
    }

    out_path = Path(__file__).parent / 'order_study_results.json'
    with open(out_path, 'w') as fh:
        json.dump(payload, fh, indent=2)
    print(f"\nWrote results to {out_path}")


if __name__ == '__main__':
    main()
