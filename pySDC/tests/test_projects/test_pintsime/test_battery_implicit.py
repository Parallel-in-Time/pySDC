import pytest


@pytest.mark.base
def test_main():
    from pySDC.projects.PinTSimE.battery_implicit import main

    main()
