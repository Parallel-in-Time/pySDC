import pytest


@pytest.mark.base
def test_main():
    from pySDC.projects.PinTSimE.battery_2capacitors_model import run

    run()
