import pytest
from pySDC.projects.Resilience.extrapolation_within_Q import check_order
from pySDC.projects.Resilience.advection import run_advection
from pySDC.projects.Resilience.piline import run_piline


@pytest.mark.base
@pytest.mark.parametrize("prob", [run_advection, run_piline])
@pytest.mark.parametrize('num_nodes', [2, 3])
@pytest.mark.parametrize('quad_type', ['RADAU-RIGHT', 'GAUSS'])
def test_order_extrapolation_estimate_within_Q(prob, num_nodes, quad_type):
    check_order(None, prob=prob, dts=[1e-1, 5e-2, 1e-2], Tend=5e-1, num_nodes=num_nodes, quad_type=quad_type)
