"""
Convergence study for SDC applied to the 1D heat equation with and without
time-dependent Dirichlet boundary conditions, using FEM (FEniCS).

This module reproduces the results from the order-reduction document referenced in
the pySDC issue: *SDC with time-dependent boundary conditions*.

Three test cases are compared:

- **Sine solution** (``fenics_heat_mass``):
  :math:`u(x,t) = \\sin(\\pi x)\\cos(t) + c` with homogeneous Dirichlet BCs.
  The right-hand side contains a forcing term that makes this the exact solution
  of the problem. Because the BCs are time-independent, SDC achieves the
  expected full temporal order of accuracy.

- **Cosine solution** (``fenics_heat_mass_timebc``):
  :math:`u(x,t) = \\cos(\\pi x)\\cos(t) + c` with time-dependent Dirichlet BCs
  :math:`u|_{\\partial\\Omega}(t) = \\cos(\\pi x)\\cos(t)`.
  In ``solve_system``, the FEniCS Dirichlet BC is applied directly to the right-hand
  side vector via ``bc.apply(b.values.vector())``.  This is the standard
  FEM/FEniCS way to impose Dirichlet BCs but it causes **order reduction** in
  SDC: the implicit sweeper's fixed point no longer matches the collocation
  solution, resulting in an effective convergence order lower than the theoretical
  SDC order.

- **Cosine solution with boundary lifting** (``fenics_heat_mass_timebc_lift``):
  Uses the same cosine solution but decomposes :math:`u = v + E` where
  :math:`E(x,t) = (1-2x)\\cos(t) + c` is a linear lift satisfying the
  time-dependent BCs. The transformed variable :math:`v = u - E` satisfies
  homogeneous Dirichlet BCs, and SDC is applied to :math:`v` with a corrected
  forcing term. This approach **restores full order** of convergence.

Usage
-----
Run directly::

    python run_convergence.py
"""

import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced import (
    fenics_heat_mass,
    fenics_heat_mass_timebc,
)
from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
from pySDC.playgrounds.FEniCS.order_reduction.problem_classes import fenics_heat_mass_timebc_lift


def build_description(problem_class, num_nodes, dt, t0=0.0, c_nvars=64, nu=0.1, c=0.0):
    """
    Build the pySDC description dictionary for a single run.

    Parameters
    ----------
    problem_class : type
        Either ``fenics_heat_mass``, ``fenics_heat_mass_timebc``, or
        ``fenics_heat_mass_timebc_lift``.
    num_nodes : int
        Number of SDC collocation nodes (RADAU-RIGHT).
    dt : float
        Time-step size.
    t0 : float, optional
        Start time (default 0.0).
    c_nvars : int, optional
        Spatial degrees of freedom (default 64).
    nu : float, optional
        Diffusion coefficient (default 0.1).
    c : float, optional
        Constant Dirichlet boundary offset (default 0.0).

    Returns
    -------
    description : dict
        pySDC description dictionary.
    controller_params : dict
        pySDC controller parameters.
    """
    level_params = {
        'restol': 1e-12,
        'dt': dt,
    }
    step_params = {'maxiter': 50}
    sweeper_params = {
        'quad_type': 'RADAU-RIGHT',
        'num_nodes': num_nodes,
    }
    problem_params = {
        'nu': nu,
        't0': t0,
        'c_nvars': c_nvars,
        'family': 'CG',
        'order': 4,
        'refinements': 1,
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


def run_sdc(problem_class, dt, num_nodes=3, t0=0.0, Tend=1.0, c_nvars=64, nu=0.1, c=0.0):
    """
    Run a single SDC solve and return the relative error at ``Tend``.

    Parameters
    ----------
    problem_class : type
        Problem class to use.
    dt : float
        Time-step size.
    num_nodes : int, optional
        Number of SDC collocation nodes (default 3).
    t0 : float, optional
        Start time (default 0.0).
    Tend : float, optional
        End time (default 1.0).
    c_nvars : int, optional
        Spatial DOFs (default 64).
    nu : float, optional
        Diffusion coefficient (default 0.1).
    c : float, optional
        Constant BC offset (default 0.0).

    Returns
    -------
    float
        Relative error :math:`|u_h - u_{\\text{exact}}| / |u_{\\text{exact}}|`
        at ``Tend``.
    """
    description, controller_params = build_description(
        problem_class, num_nodes, dt, t0=t0, c_nvars=c_nvars, nu=nu, c=c
    )

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    uend, _ = controller.run(u0=uinit, t0=t0, Tend=Tend)
    uex = P.u_exact(Tend)
    return float(abs(uex - uend) / abs(uex))


def compute_order(problem_class, dts, **kwargs):
    """
    Compute errors and least-squares convergence order estimate.

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
        Relative errors for each ``dt``.
    order : float
        Least-squares order estimate.
    """
    errors = [run_sdc(problem_class, dt, **kwargs) for dt in dts]
    order = float(np.polyfit(np.log(dts), np.log(errors), 1)[0])
    return errors, order


def main():
    """Run the convergence study and print results."""
    num_nodes = 3
    dts = [0.2 / 2**k for k in range(4)]
    Tend = 1.0

    print("=" * 70)
    print("SDC convergence study: order reduction with time-dependent BCs")
    print(f"  FEniCS FEM in space, RADAU-RIGHT with M={num_nodes} nodes, Tend={Tend}")
    print("=" * 70)

    for problem_class, label in [
        (fenics_heat_mass, "Sine  (homogeneous BCs, no order reduction)"),
        (fenics_heat_mass_timebc, "Cosine (time-dependent BCs, order reduction)"),
        (fenics_heat_mass_timebc_lift, "Cosine + lifting (boundary lifting, full order restored)"),
    ]:
        errors, order = compute_order(problem_class, dts, num_nodes=num_nodes, Tend=Tend)
        print(f"\n{label}")
        print(f"  Estimated order : {order:.2f}  (expected ≈ {2*num_nodes-1} without reduction)")
        for dt, err in zip(dts, errors):
            print(f"  dt = {dt:.5f}  rel. error = {err:.3e}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
