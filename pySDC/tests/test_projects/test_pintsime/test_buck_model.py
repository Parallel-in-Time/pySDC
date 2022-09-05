import pytest


@pytest.mark.base
def test_main():
    from pySDC.projects.PinTSimE.buck_model import main

    main()
