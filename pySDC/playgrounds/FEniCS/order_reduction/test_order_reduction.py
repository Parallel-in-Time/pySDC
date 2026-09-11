"""
Tests verifying the SDC order-reduction behaviour with time-dependent Dirichlet
boundary conditions, using the FEniCS-based 1D heat equation problem classes.

Three scenarios are tested:

1. **Sine solution** (``fenics_heat_mass``, homogeneous BCs):
   SDC achieves the expected collocation order (≈ 2M − 1 for RADAU-RIGHT with M nodes).

2. **Cosine solution** (``fenics_heat_mass_timebc``, time-dependent BCs):
   The standard FEniCS BC imposition via ``bc.apply(b.values.vector())`` in
   ``solve_system`` causes order reduction — the observed convergence order is
   lower than the theoretical SDC order.

3. **Cosine solution with boundary lifting** (``fenics_heat_mass_timebc_lift``):
   The boundary lifting approach decomposes ``u = v + E``, where ``E`` is a
   linear lift satisfying the time-dependent BCs. The transformed variable ``v``
   satisfies homogeneous BCs, and SDC applied to ``v`` **restores the full
   convergence order**.

All tests are marked ``@pytest.mark.fenics`` because they require the FEniCS/dolfin
library.
"""

import numpy as np
import pytest

# Common parameters used across all convergence tests.
# Three dt values in the large-dt asymptotic regime: [0.5, 0.25, 0.125].
# Smaller dt values cause the temporal error to fall below the spatial
# discretization floor (high-order FEM with order=4, refinements=1), which
# distorts the global convergence order estimate for M=3 SDC nodes.
_DTS = [0.5 / 2**k for k in range(3)]
_TEND = 1.0
_C_NVARS = 32  # coarse spatial resolution keeps tests fast while preserving temporal convergence


@pytest.mark.fenics
@pytest.mark.parametrize("num_nodes", [2, 3])
def test_sine_no_order_reduction(num_nodes):
    """
    Sine solution (homogeneous BCs) must converge at the expected SDC order.

    For RADAU-RIGHT collocation with M nodes and a fully converged SDC iteration,
    the expected order is 2M − 1. The measured order must be at least 2M − 2.
    """
    from pySDC.playgrounds.FEniCS.order_reduction.run_convergence import compute_order
    from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced import fenics_heat_mass

    errors, order = compute_order(fenics_heat_mass, _DTS, num_nodes=num_nodes, Tend=_TEND, c_nvars=_C_NVARS)

    expected = 2 * num_nodes - 1
    assert order >= expected - 1.0, f"Sine case (M={num_nodes}): expected order >= {expected - 1.0:.1f}, got {order:.2f}"


@pytest.mark.fenics
def test_cosine_order_reduction():
    """
    Cosine solution (time-dependent BCs) must exhibit order reduction.

    The observed convergence order with ``fenics_heat_mass_timebc`` must be
    strictly less than the theoretical SDC order minus 0.3, confirming that
    the naive time-dependent BC imposition causes a measurable loss of accuracy.
    For RADAU-RIGHT with M=3 nodes the theoretical order is 2M − 1 = 5; the
    observed order is around 4.4 – 4.5, well below 5 − 0.3 = 4.7.
    """
    from pySDC.playgrounds.FEniCS.order_reduction.run_convergence import compute_order
    from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced import fenics_heat_mass_timebc

    num_nodes = 3
    errors, order = compute_order(fenics_heat_mass_timebc, _DTS, num_nodes=num_nodes, Tend=_TEND, c_nvars=_C_NVARS)

    theoretical_order = 2 * num_nodes - 1  # = 5 for M=3
    assert order < theoretical_order - 0.3, (
        f"Cosine timebc case: expected order < {theoretical_order - 0.3:.1f} (order reduction), got {order:.2f}"
    )


@pytest.mark.fenics
def test_cosine_has_lower_order_than_sine():
    """
    The cosine solution (time-dependent BCs) must converge slower than the
    sine solution (homogeneous BCs) — i.e., order reduction is present.

    With the IMEX mass-matrix formulation the order reduction is mild (about
    0.4 – 0.5 orders for M=3). The sine order is therefore required to exceed
    the cosine order by at least 0.3.
    """
    from pySDC.playgrounds.FEniCS.order_reduction.run_convergence import compute_order
    from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced import (
        fenics_heat_mass,
        fenics_heat_mass_timebc,
    )

    num_nodes = 3
    _, order_sine = compute_order(fenics_heat_mass, _DTS, num_nodes=num_nodes, Tend=_TEND, c_nvars=_C_NVARS)
    _, order_cosine = compute_order(fenics_heat_mass_timebc, _DTS, num_nodes=num_nodes, Tend=_TEND, c_nvars=_C_NVARS)

    assert order_sine > order_cosine + 0.3, (
        f"Sine order ({order_sine:.2f}) should be > cosine timebc order ({order_cosine:.2f}) + 0.3"
    )


@pytest.mark.fenics
@pytest.mark.parametrize("num_nodes", [2, 3])
def test_lifting_restores_full_order(num_nodes):
    """
    Cosine solution with boundary lifting must recover the full SDC order.

    The boundary lifting approach (``fenics_heat_mass_timebc_lift``) transforms
    the problem so that the solver sees homogeneous BCs. This eliminates the
    source of order reduction and the observed convergence order must be at
    least 2M − 2.
    """
    from pySDC.playgrounds.FEniCS.order_reduction.run_convergence import compute_order
    from pySDC.playgrounds.FEniCS.order_reduction.problem_classes import fenics_heat_mass_timebc_lift

    errors, order = compute_order(fenics_heat_mass_timebc_lift, _DTS, num_nodes=num_nodes, Tend=_TEND, c_nvars=_C_NVARS)

    expected = 2 * num_nodes - 1
    assert order >= expected - 1.0, (
        f"Lifting case (M={num_nodes}): expected order >= {expected - 1.0:.1f} (full order restored), got {order:.2f}"
    )


@pytest.mark.fenics
def test_lifting_has_higher_order_than_timebc():
    """
    The boundary lifting case must converge faster than the plain time-dependent BC case.

    This verifies that the lifting correction actually fixes the order reduction.
    """
    from pySDC.playgrounds.FEniCS.order_reduction.run_convergence import compute_order
    from pySDC.implementations.problem_classes.HeatEquation_1D_FEniCS_matrix_forced import fenics_heat_mass_timebc
    from pySDC.playgrounds.FEniCS.order_reduction.problem_classes import fenics_heat_mass_timebc_lift

    num_nodes = 3
    _, order_timebc = compute_order(fenics_heat_mass_timebc, _DTS, num_nodes=num_nodes, Tend=_TEND, c_nvars=_C_NVARS)
    _, order_lift = compute_order(
        fenics_heat_mass_timebc_lift, _DTS, num_nodes=num_nodes, Tend=_TEND, c_nvars=_C_NVARS
    )

    assert order_lift > order_timebc + 0.3, (
        f"Lifting order ({order_lift:.2f}) should be > timebc order ({order_timebc:.2f}) + 0.3"
    )
