import pytest


@pytest.mark.base
def test_main():
    from pySDC.projects.PinTSimE.battery_implicit import run_check

    run_check()
