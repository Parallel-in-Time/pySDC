"""
Tests for SDC order reduction and boundary lifting in the 1D Allen-Cahn FD
problem with time-dependent Dirichlet boundary conditions.

Three scenarios are tested:

1. **Original** (``allencahn_front_fullyimplicit``, correct BCs):
   SDC achieves the expected collocation order :math:`\\approx 2M - 1` for
   RADAU-RIGHT with :math:`M` nodes.

2. **Naive** (``allencahn_front_fullyimplicit_naive``, zero BCs in solve):
   Imposing zero BCs inside ``solve_system`` while ``eval_f`` uses the correct
   time-dependent BCs causes **order reduction** — the observed convergence
   order is far below the theoretical SDC order.

3. **Lifted** (``allencahn_front_fullyimplicit_lift``, boundary lifting):
   Reformulating in terms of :math:`w = u - E`, where :math:`E` is a linear
   lift satisfying the time-dependent BCs, gives a problem with homogeneous
   BCs.  SDC applied to :math:`w` **restores the full convergence order**.

Parameter rationale
-------------------
``eps = 0.5`` sets the interface width; ``eps^2 = 0.25`` is the natural
upper bound on the step size (semiimplicit stability constraint).  The test
uses ``dts = [eps^2, eps^2/2, eps^2/4] = [0.25, 0.125, 0.0625]`` so that
temporal errors dominate over the spatial (FD) discretisation error.
``dw = -1.0`` drives the front fast enough (speed ``v ≈ -2.12``) to produce
visible temporal errors in this dt range.
"""

import numpy as np
import pytest

from pySDC.implementations.problem_classes.AllenCahn_1D_FD import allencahn_front_fullyimplicit
from pySDC.playgrounds.Allen_Cahn.order_reduction.problem_classes import (
    allencahn_front_fullyimplicit_lift,
    allencahn_front_fullyimplicit_naive,
)
from pySDC.playgrounds.Allen_Cahn.order_reduction.run_convergence import compute_order

# Common parameters — all tests use dt <= eps^2 to stay in the temporal-error regime
_EPS = 0.5
_DW = -1.0
_DTS = [_EPS**2 / 2**k for k in range(3)]  # [0.25, 0.125, 0.0625]
_TEND = 5.0 * _EPS**2  # = 1.25
_NVARS = 127


@pytest.mark.parametrize('num_nodes', [2, 3])
def test_original_full_order(num_nodes):
    """
    Original problem (correct BCs) must achieve the expected SDC order.

    For RADAU-RIGHT with :math:`M` nodes the collocation order is
    :math:`2M - 1`.  We require the measured order to be at least :math:`M`
    (conservative, allowing for the spatial FD discretisation floor that
    limits the measurable order at the smallest dt).
    """
    _, order = compute_order(
        allencahn_front_fullyimplicit,
        _DTS,
        num_nodes=num_nodes,
        Tend=_TEND,
        eps=_EPS,
        dw=_DW,
        nvars=_NVARS,
    )
    assert order >= num_nodes, (
        f'Original (M={num_nodes}): expected order >= {num_nodes}, got {order:.2f}'
    )


@pytest.mark.parametrize('num_nodes', [2, 3])
def test_naive_order_reduction(num_nodes):
    """
    Naive problem (zero BCs in solve_system) must exhibit order reduction.

    The observed convergence order must be strictly less than :math:`M - 0.5`,
    confirming that the zero-BC mismatch causes a measurable loss of accuracy.
    """
    _, order = compute_order(
        allencahn_front_fullyimplicit_naive,
        _DTS,
        num_nodes=num_nodes,
        Tend=_TEND,
        eps=_EPS,
        dw=_DW,
        nvars=_NVARS,
    )
    assert order < num_nodes - 0.5, (
        f'Naive (M={num_nodes}): expected order < {num_nodes - 0.5} (order reduction), got {order:.2f}'
    )


@pytest.mark.parametrize('num_nodes', [2, 3])
def test_lifting_restores_order(num_nodes):
    """
    Lifted problem (boundary lifting) must recover the expected SDC order.

    The lifted variable :math:`w = u - E` satisfies homogeneous BCs, so
    the Newton solve is consistent with the SDC collocation equations.
    The observed order must be at least :math:`M`.
    """
    _, order = compute_order(
        allencahn_front_fullyimplicit_lift,
        _DTS,
        num_nodes=num_nodes,
        Tend=_TEND,
        eps=_EPS,
        dw=_DW,
        nvars=_NVARS,
    )
    assert order >= num_nodes, (
        f'Lifted (M={num_nodes}): expected order >= {num_nodes} (full order restored), got {order:.2f}'
    )


@pytest.mark.parametrize('num_nodes', [2, 3])
def test_lifting_beats_naive(num_nodes):
    """
    Lifting must converge at a strictly higher order than the naive approach.

    This verifies that the lifting correction actually fixes the order
    reduction: the lifted order must exceed the naive order by at least 1.
    """
    _, order_naive = compute_order(
        allencahn_front_fullyimplicit_naive,
        _DTS,
        num_nodes=num_nodes,
        Tend=_TEND,
        eps=_EPS,
        dw=_DW,
        nvars=_NVARS,
    )
    _, order_lift = compute_order(
        allencahn_front_fullyimplicit_lift,
        _DTS,
        num_nodes=num_nodes,
        Tend=_TEND,
        eps=_EPS,
        dw=_DW,
        nvars=_NVARS,
    )
    assert order_lift > order_naive + 1.0, (
        f'Lifted order ({order_lift:.2f}) should be > naive order ({order_naive:.2f}) + 1.0 (M={num_nodes})'
    )
