import pytest


@pytest.mark.PinTSimE
def test_main():
    from pySDC.projects.PinTSimE.battery_model import main

    main()
