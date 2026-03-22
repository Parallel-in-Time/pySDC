"""
Tests verifying the SDC order-reduction behaviour for the heat equation
with time-dependent Dirichlet boundary conditions.

The test cases check three scenarios:

1. **Sine solution** (homogeneous BCs): SDC achieves the expected convergence
   order (≈ ``num_sweeps``).
2. **Cosine solution — naive** (time-dependent BCs, boundary correction omitted
   from ``solve_system``): severe order reduction (effective order close to 0).
3. **Cosine solution — corrected** (boundary correction included in
   ``solve_system``): convergence is restored to an order strictly greater
   than the naive case.
"""

import numpy as np
import pytest

from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit

from pySDC.projects.order_reduction.heat_equation import (
    HeatEquation_1D_FD_homogeneous_Dirichlet,
    HeatEquation_1D_FD_time_dependent_Dirichlet,
    HeatEquation_1D_FD_time_dependent_Dirichlet_full,
)


def run_sdc(problem_class, dt, num_nodes=3, num_sweeps=3, nvars=63, nu=0.1, freq=1, t0=0.0, Tend=1.0):
    """Run one SDC solve and return the max-norm error."""
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


def estimate_order(problem_class, dts, **kwargs):
    """Return the least-squares order estimate over ``dts``."""
    errors = [run_sdc(problem_class, dt, **kwargs) for dt in dts]
    return float(np.polyfit(np.log(dts), np.log(errors), 1)[0])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_sweeps", [2, 3])
def test_sine_no_order_reduction(num_sweeps):
    """
    Sine solution (homogeneous BCs) must converge at the expected SDC order.

    For K sweeps with RADAU-RIGHT nodes, each additional sweep adds one order
    of accuracy (order ≈ K).  The measured order must be at least K - 0.5.
    """
    dts = [1.0 / 8, 1.0 / 16, 1.0 / 32, 1.0 / 64]
    order = estimate_order(
        HeatEquation_1D_FD_homogeneous_Dirichlet,
        dts,
        num_nodes=3,
        num_sweeps=num_sweeps,
    )
    assert order >= num_sweeps - 0.5, (
        f"Sine case: expected order ≥ {num_sweeps - 0.5:.1f}, got {order:.2f}"
    )


def test_cosine_naive_order_reduction():
    """
    Cosine solution (time-dependent BCs) with the naive ``solve_system``
    must exhibit order reduction: the observed order must be less than 1.
    """
    dts = [1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32, 1.0 / 64]
    order = estimate_order(
        HeatEquation_1D_FD_time_dependent_Dirichlet,
        dts,
        num_nodes=3,
        num_sweeps=3,
    )
    assert order < 1.0, (
        f"Cosine naive case: expected order < 1 (order reduction), got {order:.2f}"
    )


def test_cosine_corrected_better_than_naive():
    """
    Cosine solution with the corrected ``solve_system`` (BC correction included)
    must converge strictly faster than the naive implementation.
    """
    dts = [1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32, 1.0 / 64]
    kwargs = {'num_nodes': 3, 'num_sweeps': 3}
    order_naive = estimate_order(HeatEquation_1D_FD_time_dependent_Dirichlet, dts, **kwargs)
    order_fixed = estimate_order(HeatEquation_1D_FD_time_dependent_Dirichlet_full, dts, **kwargs)
    assert order_fixed > order_naive + 0.5, (
        f"Corrected (order {order_fixed:.2f}) must beat naive (order {order_naive:.2f}) by ≥ 0.5"
    )


def test_problem_classes_instantiate():
    """Smoke test: all three problem classes must instantiate without error."""
    for cls in [
        HeatEquation_1D_FD_homogeneous_Dirichlet,
        HeatEquation_1D_FD_time_dependent_Dirichlet,
        HeatEquation_1D_FD_time_dependent_Dirichlet_full,
    ]:
        P = cls(nvars=31, nu=0.1, freq=1)
        u0 = P.u_exact(0.0)
        assert u0.shape == (31,)
        f = P.eval_f(u0, 0.0)
        assert f.shape == (31,)
