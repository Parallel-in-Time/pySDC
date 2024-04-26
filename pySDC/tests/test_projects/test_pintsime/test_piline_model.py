import pytest


@pytest.mark.PinTSimE
def test_main():
    from pySDC.projects.PinTSimE.piline_model import main

    main()
