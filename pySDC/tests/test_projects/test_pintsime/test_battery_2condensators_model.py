import pytest


@pytest.mark.base
def test_main():
    from pySDC.projects.PinTSimE.battery_2condensators_model import main
    
    main()
