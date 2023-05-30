import pytest


@pytest.mark.base
@pytest.mark.parametrize("prob_name", ['advection', 'piline'])
@pytest.mark.parametrize('num_nodes', [2, 3])
@pytest.mark.parametrize('quad_type', ['RADAU-RIGHT', 'GAUSS'])
def test_order_extrapolation_estimate_within_Q(prob_name, num_nodes, quad_type):
    from pySDC.projects.Resilience.extrapolation_within_Q import check_order

    if prob_name == 'advection':
        from pySDC.projects.Resilience.advection import run_advection

        prob = run_advection
    elif prob_name == 'piline':
        from pySDC.projects.Resilience.piline import run_piline

        prob = run_piline

    else:
        raise NotImplementedError(f'Problem \"{prob_name}\" not implemented in this test!')

    check_order(None, prob=prob, dts=[5e-1, 1e-1, 5e-2, 1e-2], num_nodes=num_nodes, quad_type=quad_type)
