import pytest


@pytest.mark.base
def test_main():
    from pySDC.projects.PinTSimE.piline_model import main

    main()
