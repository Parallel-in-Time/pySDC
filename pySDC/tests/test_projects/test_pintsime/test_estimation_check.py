import pytest


@pytest.mark.PinTSimE
def test_main():
    from pySDC.projects.PinTSimE.estimation_check import run_estimation_check

    run_estimation_check()
