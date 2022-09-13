import pytest

from pySDC.playgrounds.Runge_Kutta.test_order import test_vdp, test_advection

def test_main():
    test_vdp()
    test_advection()
