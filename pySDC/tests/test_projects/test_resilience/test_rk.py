import pytest

from pySDC.projects.Resilience.test_Runge_Kutta_sweeper import test_vdp, test_advection, test_embedded_method


def test_main():
    test_vdp()
    test_advection()
    test_embedded_method()
