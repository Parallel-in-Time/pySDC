import pytest


@pytest.mark.base
def test_main():
    from pySDC.projects.PinTSimE.estimation_check_implicit.py import check

    check()
