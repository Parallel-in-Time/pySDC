import pytest


@pytest.mark.base
def test_main():
    from pySDC.projects.Resilience.Lorenz import main

    main(plotting=False)
