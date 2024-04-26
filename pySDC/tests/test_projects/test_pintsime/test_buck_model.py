import pytest


@pytest.mark.PinTSimE
def test_main():
    from pySDC.projects.PinTSimE.buck_model import main

    main()
