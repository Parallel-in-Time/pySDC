"""
Convergence study for SDC applied to the 1D Allen-Cahn equation with
inhomogeneous time-dependent Dirichlet boundary conditions, using FEM (FEniCS).

Two test cases are compared:

- **Naive** (``fenics_allencahn_imex_timebc``):
  :math:`u(x,t) = \\tfrac{1}{2}(1 + \\tanh((x - 0.5 - vt)/(\\sqrt{2}\\,\\varepsilon)))`
  with time-dependent Dirichlet BCs enforced via ``bc.apply(b.values.vector())``
  inside ``solve_system``.  This is the standard FEniCS mechanism but it
  **causes order reduction**: the observed convergence order is lower than the
  theoretical SDC order :math:`2M - 1`.

- **Lifted** (``fenics_allencahn_imex_timebc_lift``):
  The solution is decomposed as :math:`u = v + E` where
  :math:`E(x,t) = u_L(t)(1-x) + u_R(t)\\,x` is a linear lift matching the
  time-dependent BCs.  The transformed variable :math:`v` satisfies
  **homogeneous** BCs, and ``solve_system`` only applies ``bc_hom``.  This
  **restores the full SDC order**.

Usage
-----
Run directly::

    python run_convergence.py
"""

import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.imex_1st_order_mass import imex_1st_order_mass
from pySDC.playgrounds.FEniCS.allen_cahn_1d.problem_classes import (
    fenics_allencahn_imex_timebc,
    fenics_allencahn_imex_timebc_lift,
)


def build_description(problem_class, num_nodes, dt, t0=0.0, c_nvars=64, eps=0.3, dw=-0.04):
    """
    Build the pySDC description dictionary for a single run.

    Parameters
    ----------
    problem_class : type
        Either :class:`fenics_allencahn_imex_timebc` or
        :class:`fenics_allencahn_imex_timebc_lift`.
    num_nodes : int
        Number of SDC collocation nodes (RADAU-RIGHT).
    dt : float
        Time-step size.
    t0 : float, optional
        Start time. Default ``0.0``.
    c_nvars : int, optional
        Spatial degrees of freedom. Default ``64``.
    eps : float, optional
        Interface parameter :math:`\\varepsilon`. Default ``0.3``.
    dw : float, optional
        Driving force. Default ``-0.04``.

    Returns
    -------
    description : dict
    controller_params : dict
    """
    level_params = {'restol': 1e-12, 'dt': dt}
    step_params = {'maxiter': 50}
    sweeper_params = {'quad_type': 'RADAU-RIGHT', 'num_nodes': num_nodes}
    problem_params = {
        'c_nvars': c_nvars,
        't0': t0,
        'family': 'CG',
        'order': 4,
        'refinements': 1,
        'eps': eps,
        'dw': dw,
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


def run_sdc(problem_class, dt, num_nodes=3, t0=0.0, Tend=1.0, c_nvars=64, eps=0.3, dw=-0.04):
    """
    Run a single SDC solve and return the relative :math:`L^2` error at ``Tend``.

    Parameters
    ----------
    problem_class : type
    dt : float
    num_nodes : int, optional
    t0 : float, optional
    Tend : float, optional
    c_nvars : int, optional
    eps : float, optional
    dw : float, optional

    Returns
    -------
    float
        Relative error :math:`\\|u_h - u_{\\text{exact}}\\| / \\|u_{\\text{exact}}\\|`
        at ``Tend``.
    """
    description, controller_params = build_description(
        problem_class, num_nodes, dt, t0=t0, c_nvars=c_nvars, eps=eps, dw=dw
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
    dts : list of float
    **kwargs
        Forwarded to :func:`run_sdc`.

    Returns
    -------
    errors : list of float
    order : float
    """
    errors = [run_sdc(problem_class, dt, **kwargs) for dt in dts]
    order = float(np.polyfit(np.log(dts), np.log(errors), 1)[0])
    return errors, order


def main():
    """Run the convergence study and print results."""
    num_nodes = 3
    # Large-dt regime where temporal errors dominate over the FEM spatial error
    dts = [0.5 / 2**k for k in range(3)]
    Tend = 1.0

    print("=" * 70)
    print("SDC convergence study: order reduction with time-dependent BCs")
    print("  Allen-Cahn equation in FEniCS (IMEX, diffusion implicit,")
    print(f"  reaction explicit), RADAU-RIGHT M={num_nodes} nodes, Tend={Tend}")
    print("=" * 70)

    for problem_class, label in [
        (fenics_allencahn_imex_timebc, "Naive  (time-dependent BCs, order reduction)"),
        (fenics_allencahn_imex_timebc_lift, "Lifted (boundary lifting, full order restored)"),
    ]:
        errors, order = compute_order(problem_class, dts, num_nodes=num_nodes, Tend=Tend)
        print(f"\n{label}")
        print(f"  Estimated order : {order:.2f}  (expected ~{2*num_nodes-1} without reduction)")
        for dt, err in zip(dts, errors):
            print(f"  dt = {dt:.5f}  rel. error = {err:.3e}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()
