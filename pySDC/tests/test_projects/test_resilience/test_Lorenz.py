import pytest


@pytest.mark.base
def test_main():
    from pySDC.projects.Resilience.Lorentz import main

    main(plotting=False)
