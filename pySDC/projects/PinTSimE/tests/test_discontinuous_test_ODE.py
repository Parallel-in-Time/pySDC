import pytest


@pytest.mark.base
def test_main():
    from pySDC.projects.PinTSimE.discontinuous_test_ODE import main

    main()
