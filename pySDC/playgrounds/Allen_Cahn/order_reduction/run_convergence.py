"""
Convergence study for SDC applied to the 1D Allen-Cahn equation with and
without boundary lifting for time-dependent Dirichlet BCs, using the
finite-difference fully-implicit solver.

Three variants of the problem are compared:

- **Original** (:class:`allencahn_front_fullyimplicit`):
  Correct time-dependent BCs are imposed in *both* ``eval_f`` and
  ``solve_system``.  This is the reference; SDC achieves the expected
  full temporal convergence order :math:`2M - 1`.

- **Naive** (:class:`allencahn_front_fullyimplicit_naive`):
  ``eval_f`` uses the correct time-dependent BCs, but ``solve_system``
  imposes *zero* BCs.  The mismatch causes **order reduction**: the SDC
  fixed point no longer matches the collocation solution.

- **Lifted** (:class:`allencahn_front_fullyimplicit_lift`):
  Solves for the lifted variable :math:`w = u - E`, where
  :math:`E(x, t)` is a linear interpolant of the boundary data.
  The lifted variable satisfies homogeneous BCs, so ``solve_system``
  correctly uses zero BCs throughout, restoring the **full convergence
  order**.

Parameter choice
----------------
The Allen-Cahn parameter :math:`\\varepsilon` sets the interface width and
the natural stiffness scale.  A semiimplicit (IMEX) scheme requires
:math:`\\Delta t \\lesssim \\varepsilon^2` for stability, so
:math:`\\varepsilon^2` is an upper bound on the useful step-size range.

Here we use :math:`\\varepsilon = 0.5`, giving :math:`\\Delta t_{\\max} =
\\varepsilon^2 = 0.25`.  The three dt values
:math:`[\\varepsilon^2,\\, \\varepsilon^2/2,\\, \\varepsilon^2/4]` lie in the
temporal-error-dominated regime and yield clear convergence behaviour.

The parameter :math:`\\delta w = -1` is chosen so that the travelling wave
moves at speed :math:`v = 3\\sqrt{2}\\,\\varepsilon\\,\\delta w \\approx -2.12`,
giving large enough temporal derivatives for the error to be visible at these
step sizes.

Usage
-----
Run directly::

    python run_convergence.py
"""

import numpy as np

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.AllenCahn_1D_FD import allencahn_front_fullyimplicit
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.playgrounds.Allen_Cahn.order_reduction.problem_classes import (
    allencahn_front_fullyimplicit_lift,
    allencahn_front_fullyimplicit_naive,
)

# Default parameters
_EPS = 0.5  # interface width; eps^2 = 0.25 gives the dt upper bound
_DW = -1.0  # driving force; |v| = 3*sqrt(2)*eps*|dw| ≈ 2.12
_NVARS = 127  # interior grid points (must satisfy nvars+1 = 2^p)
_TEND = 5.0 * _EPS**2  # = 1.25; enough steps to observe temporal convergence
_DTS = [_EPS**2 / 2**k for k in range(3)]  # [0.25, 0.125, 0.0625]


def build_description(problem_class, num_nodes, dt, eps=_EPS, dw=_DW, nvars=_NVARS):
    """
    Build a pySDC description dictionary for a single Allen-Cahn FD run.

    Parameters
    ----------
    problem_class : type
        One of ``allencahn_front_fullyimplicit``,
        ``allencahn_front_fullyimplicit_naive``, or
        ``allencahn_front_fullyimplicit_lift``.
    num_nodes : int
        Number of SDC collocation nodes (RADAU-RIGHT).
    dt : float
        Time-step size.
    eps : float, optional
        Allen-Cahn interface-width parameter (default :data:`_EPS`).
    dw : float, optional
        Driving-force parameter (default :data:`_DW`).
    nvars : int, optional
        Number of interior FD grid points (default :data:`_NVARS`).

    Returns
    -------
    description : dict
        pySDC description dictionary.
    controller_params : dict
        pySDC controller parameters.
    """
    level_params = {'restol': 1e-12, 'dt': dt}
    step_params = {'maxiter': 50}
    sweeper_params = {'quad_type': 'RADAU-RIGHT', 'num_nodes': num_nodes, 'QI': 'IE'}
    problem_params = {
        'nvars': nvars,
        'eps': eps,
        'dw': dw,
        'interval': (-0.5, 0.5),
        'newton_maxiter': 100,
        'newton_tol': 1e-12,
        'stop_at_nan': False,
    }
    controller_params = {'logger_level': 30}
    description = {
        'problem_class': problem_class,
        'problem_params': problem_params,
        'sweeper_class': generic_implicit,
        'sweeper_params': sweeper_params,
        'level_params': level_params,
        'step_params': step_params,
    }
    return description, controller_params


def run_sdc(problem_class, dt, num_nodes=3, t0=0.0, Tend=_TEND, eps=_EPS, dw=_DW, nvars=_NVARS):
    """
    Run a single SDC solve and return the relative error at ``Tend``.

    Parameters
    ----------
    problem_class : type
        Problem class to use.
    dt : float
        Time-step size.
    num_nodes : int, optional
        Number of SDC collocation nodes (default ``3``).
    t0 : float, optional
        Start time (default ``0.0``).
    Tend : float, optional
        End time (default :data:`_TEND`).
    eps : float, optional
        Allen-Cahn interface width (default :data:`_EPS`).
    dw : float, optional
        Driving force (default :data:`_DW`).
    nvars : int, optional
        Number of interior grid points (default :data:`_NVARS`).

    Returns
    -------
    float
        Relative error :math:`\\|u_h(T) - u^*(T)\\| / \\|u^*(T)\\|`.
    """
    description, controller_params = build_description(problem_class, num_nodes, dt, eps=eps, dw=dw, nvars=nvars)
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)
    uend, _ = controller.run(u0=uinit, t0=t0, Tend=Tend)
    uex = P.u_exact(Tend)
    return float(abs(uex - uend) / abs(uex))


def compute_order(problem_class, dts, **kwargs):
    """
    Compute errors and a least-squares convergence order estimate.

    Parameters
    ----------
    problem_class : type
        Problem class to use.
    dts : list of float
        Step sizes to test (should be decreasing).
    **kwargs
        Forwarded to :func:`run_sdc`.

    Returns
    -------
    errors : list of float
        Relative errors for each ``dt`` in ``dts``.
    order : float
        Least-squares estimate of the convergence order.
    """
    errors = [run_sdc(problem_class, dt, **kwargs) for dt in dts]
    order = float(np.polyfit(np.log(dts), np.log(errors), 1)[0])
    return errors, order


def main():
    """Run the convergence study and print results."""
    num_nodes = 3
    dts = _DTS
    Tend = _TEND

    print('=' * 70)
    print('SDC convergence study: order reduction with time-dependent BCs')
    print(f'  Allen-Cahn 1D FD, RADAU-RIGHT M={num_nodes} nodes, Tend={Tend}')
    print(f'  eps={_EPS}, dw={_DW}, dt in {dts}  (dt <= eps^2={_EPS**2})')
    print('=' * 70)

    for problem_class, label in [
        (allencahn_front_fullyimplicit, 'Original  (correct BCs, no order reduction)'),
        (allencahn_front_fullyimplicit_naive, 'Naive     (zero BCs in solve_system, order reduction)'),
        (allencahn_front_fullyimplicit_lift, 'Lifted    (boundary lifting, full order restored)'),
    ]:
        errors, order = compute_order(problem_class, dts, num_nodes=num_nodes, Tend=Tend)
        print(f'\n{label}')
        print(f'  Estimated order : {order:.2f}  (expected ≈ {2 * num_nodes - 1} without reduction)')
        for dt, err in zip(dts, errors):
            print(f'  dt = {dt:.4f}  rel. error = {err:.3e}')

    print('\n' + '=' * 70)


if __name__ == '__main__':
    main()
