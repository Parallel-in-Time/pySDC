import pytest


@pytest.mark.base
def test_adaptivity_collocation():
    from pySDC.projects.Resilience.collocation_adaptivity import adaptivity_collocation

    adaptivity_collocation(plotting=False)


@pytest.mark.base
def test_error_estimate_order():
    from pySDC.projects.Resilience.collocation_adaptivity import order_stuff, run_advection

    order_stuff(run_advection)


@pytest.mark.base
def test_adaptive_collocation():
    from pySDC.projects.Resilience.collocation_adaptivity import compare_adaptive_collocation, run_vdp

    compare_adaptive_collocation(run_vdp)
