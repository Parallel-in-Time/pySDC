import pytest


@pytest.mark.base
def test_main():
    from pySDC.projects.Resilience.test_Runge_Kutta_sweeper import (
        test_vdp,
        test_advection,
        test_embedded_method,
        test_embedded_estimate_order,
        Heun_Euler,
    )

    test_vdp()
    test_advection()
    test_embedded_method()
    test_embedded_estimate_order(Heun_Euler)
