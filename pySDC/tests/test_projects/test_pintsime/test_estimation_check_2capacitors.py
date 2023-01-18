import pytest


@pytest.mark.base
def test_main():
    from pySDC.projects.PinTSimE.estimation_check_2capacitors import check

    check()
