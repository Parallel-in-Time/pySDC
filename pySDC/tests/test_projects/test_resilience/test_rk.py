import pytest


@pytest.mark.base
def test_main():
    from pySDC.projects.Resilience.test_Runge_Kutta_sweeper import test_vdp, test_advection, test_embedded_method

    test_vdp()
    test_advection()
    test_embedded_method()
