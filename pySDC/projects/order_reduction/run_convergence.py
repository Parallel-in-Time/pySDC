"""
Convergence study demonstrating SDC order reduction with time-dependent boundary conditions.

Reproduces the results from the order-reduction document referenced in the pySDC issue:

- **Sine solution** ``u(x,t) = sin(πx) exp(-ρ_FD t)`` with homogeneous Dirichlet BCs:
  no order reduction — each SDC sweep adds one order of accuracy.
- **Cosine solution** (naive) ``u(x,t) = cos(πx) exp(-ν π² t)`` with time-dependent
  Dirichlet BCs and the boundary correction omitted from ``solve_system``:
  severe order reduction (effective order ≈ 0).
- **Cosine solution** (corrected) — boundary correction included in ``solve_system``:
  convergence is restored.

Usage
-----
Run directly::

    python run_convergence.py

The convergence orders and errors are printed to stdout.
"""

import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit

from pySDC.projects.order_reduction.heat_equation import (
    HeatEquation_1D_FD_homogeneous_Dirichlet,
    HeatEquation_1D_FD_time_dependent_Dirichlet,
    HeatEquation_1D_FD_time_dependent_Dirichlet_full,
)


def run_sdc(problem_class, dt, num_nodes=3, num_sweeps=3, nvars=127, nu=0.1, freq=1, t0=0.0, Tend=1.0):
    """
    Run a single SDC solve and return the max-norm error at ``Tend``.

    Parameters
    ----------
    problem_class : type
        One of the problem classes from :mod:`heat_equation`.
    dt : float
        Time-step size.
    num_nodes : int
        Number of collocation nodes (RADAU-RIGHT).
    num_sweeps : int
        Number of SDC sweeps per step.
    nvars : int
        Number of interior spatial degrees of freedom.
    nu : float
        Diffusion coefficient.
    freq : int
        Spatial frequency of the initial condition.
    t0 : float
        Start time.
    Tend : float
        End time.

    Returns
    -------
    float
        Maximum absolute error at ``Tend``.
    """
    description = {
        'problem_class': problem_class,
        'problem_params': {'nvars': nvars, 'nu': nu, 'freq': freq},
        'sweeper_class': generic_implicit,
        'sweeper_params': {
            'quad_type': 'RADAU-RIGHT',
            'num_nodes': num_nodes,
            'QI': 'LU',
            'initial_guess': 'spread',
        },
        'level_params': {'restol': -1.0, 'dt': dt, 'nsweeps': num_sweeps},
        'step_params': {'maxiter': 1},
    }
    controller = controller_nonMPI(num_procs=1, controller_params={'logger_level': 30}, description=description)
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    uend, _ = controller.run(u0=uinit, t0=t0, Tend=Tend)
    uex = P.u_exact(Tend)
    return float(np.max(np.abs(uend - uex)))


def compute_order(problem_class, dts, **kwargs):
    """
    Compute errors and least-squares convergence order over a list of step sizes.

    Parameters
    ----------
    problem_class : type
        Problem class to use.
    dts : list of float
        Step sizes to test.
    **kwargs
        Forwarded to :func:`run_sdc`.

    Returns
    -------
    errors : list of float
        Max-norm errors for each ``dt``.
    order : float
        Least-squares estimate of the convergence order.
    """
    errors = [run_sdc(problem_class, dt, **kwargs) for dt in dts]
    order = float(np.polyfit(np.log(dts), np.log(errors), 1)[0])
    return errors, order


def main():
    """Run the convergence study and print results."""
    dts = [1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32, 1.0 / 64]
    num_nodes = 3
    num_sweeps = 3

    print("=" * 70)
    print("SDC convergence study: order reduction with time-dependent BCs")
    print(f"  num_nodes={num_nodes},  num_sweeps={num_sweeps},  Tend=1.0")
    print("=" * 70)

    for problem_class, label in [
        (HeatEquation_1D_FD_homogeneous_Dirichlet, "Sine  (zero BCs, no order reduction)"),
        (HeatEquation_1D_FD_time_dependent_Dirichlet, "Cosine (time-dep BCs, naive  — order reduction)"),
        (HeatEquation_1D_FD_time_dependent_Dirichlet_full, "Cosine (time-dep BCs, fixed  — order restored)"),
    ]:
        errors, order = compute_order(problem_class, dts, num_nodes=num_nodes, num_sweeps=num_sweeps)
        print(f"\n{label}")
        print(f"  Estimated order : {order:.2f}")
        for dt, err in zip(dts, errors):
            print(f"  dt = {dt:.5f}  error = {err:.3e}")

    print("\n" + "=" * 70)
    print("Summary: sine case shows order ≈ num_sweeps (K=%d);" % num_sweeps)
    print("         cosine naive shows order ≈ 0 (order reduction);")
    print("         cosine fixed shows improved order (> 1).")
    print("=" * 70)


if __name__ == '__main__':
    main()
