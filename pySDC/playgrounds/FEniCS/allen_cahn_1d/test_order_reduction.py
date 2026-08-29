"""
Tests verifying SDC order-reduction behaviour with inhomogeneous time-dependent
Dirichlet boundary conditions for the 1D Allen-Cahn equation, using FEniCS.

Two scenarios are tested:

1. **Naive** (``fenics_allencahn_imex_timebc``, time-dependent BCs):
   The standard FEniCS BC imposition via ``bc.apply(b.values.vector())`` in
   ``solve_system`` causes order reduction — the observed convergence order
   is lower than the theoretical SDC order :math:`2M - 1`.

2. **Lifted** (``fenics_allencahn_imex_timebc_lift``):
   Boundary lifting decomposes :math:`u = v + E`, where :math:`E` is a
   linear lift matching the time-dependent BCs.  The transformed variable
   :math:`v` satisfies homogeneous BCs and SDC applied to :math:`v`
   **restores the full convergence order**.

All tests are marked ``@pytest.mark.fenics`` because they require FEniCS/dolfin.

Parameters
----------
_DTS : list of float
    Three step sizes in the large-dt asymptotic regime ``[0.5, 0.25, 0.125]``.
    At these step sizes temporal errors dominate over the high-order FEM
    spatial discretisation error (CG-4 on a refined mesh).
_TEND : float
    End time ``1.0``.
_C_NVARS : int
    Coarse spatial resolution ``32`` — keeps tests fast while preserving
    the temporal convergence signal.
"""

import numpy as np
import pytest

# dt values in the large-dt asymptotic regime where temporal errors dominate
_DTS = [0.5 / 2**k for k in range(3)]
_TEND = 1.0
_C_NVARS = 32  # coarse mesh keeps tests fast; CG-4 spatial error is negligible


@pytest.mark.fenics
def test_allencahn_order_reduction():
    """
    Naive Allen-Cahn (time-dependent BCs) must exhibit order reduction.

    The observed convergence order with ``fenics_allencahn_imex_timebc`` must
    be strictly less than the theoretical SDC order :math:`2M - 1 - 0.3`,
    confirming that the standard FEniCS BC imposition causes a measurable
    loss of temporal accuracy.

    For RADAU-RIGHT with M=3 nodes the theoretical order is 5; the observed
    order should be noticeably lower.
    """
    from pySDC.playgrounds.FEniCS.allen_cahn_1d.run_convergence import compute_order
    from pySDC.playgrounds.FEniCS.allen_cahn_1d.problem_classes import fenics_allencahn_imex_timebc

    num_nodes = 3
    _, order = compute_order(fenics_allencahn_imex_timebc, _DTS, num_nodes=num_nodes, Tend=_TEND, c_nvars=_C_NVARS)

    theoretical_order = 2 * num_nodes - 1  # = 5 for M=3
    assert order < theoretical_order - 0.3, (
        f"Naive timebc case (M={num_nodes}): expected order < {theoretical_order - 0.3:.1f} "
        f"(order reduction), got {order:.2f}"
    )


@pytest.mark.fenics
@pytest.mark.parametrize("num_nodes", [2, 3])
def test_lifting_restores_full_order(num_nodes):
    """
    Allen-Cahn with boundary lifting must recover the full SDC order.

    The lifting approach (``fenics_allencahn_imex_timebc_lift``) transforms
    the problem so that the solver always sees homogeneous BCs.  This
    eliminates the source of order reduction and the observed convergence
    order must be at least :math:`2M - 2`.
    """
    from pySDC.playgrounds.FEniCS.allen_cahn_1d.run_convergence import compute_order
    from pySDC.playgrounds.FEniCS.allen_cahn_1d.problem_classes import fenics_allencahn_imex_timebc_lift

    _, order = compute_order(
        fenics_allencahn_imex_timebc_lift, _DTS, num_nodes=num_nodes, Tend=_TEND, c_nvars=_C_NVARS
    )

    expected = 2 * num_nodes - 1
    assert order >= expected - 1.0, (
        f"Lifting case (M={num_nodes}): expected order >= {expected - 1.0:.1f} "
        f"(full order restored), got {order:.2f}"
    )


@pytest.mark.fenics
def test_lifting_has_higher_order_than_naive():
    """
    The boundary lifting case must converge faster than the naive case.

    This verifies that the lifting correction actually fixes the order
    reduction.  With RADAU-RIGHT M=3 nodes, the lifted order must exceed
    the naive order by at least 0.3.
    """
    from pySDC.playgrounds.FEniCS.allen_cahn_1d.run_convergence import compute_order
    from pySDC.playgrounds.FEniCS.allen_cahn_1d.problem_classes import (
        fenics_allencahn_imex_timebc,
        fenics_allencahn_imex_timebc_lift,
    )

    num_nodes = 3
    _, order_naive = compute_order(
        fenics_allencahn_imex_timebc, _DTS, num_nodes=num_nodes, Tend=_TEND, c_nvars=_C_NVARS
    )
    _, order_lift = compute_order(
        fenics_allencahn_imex_timebc_lift, _DTS, num_nodes=num_nodes, Tend=_TEND, c_nvars=_C_NVARS
    )

    assert order_lift > order_naive + 0.3, (
        f"Lifting order ({order_lift:.2f}) should be > naive timebc order ({order_naive:.2f}) + 0.3"
    )
