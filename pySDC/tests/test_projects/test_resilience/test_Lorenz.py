import pytest


@pytest.mark.Resilience
def test_main():
    from pySDC.projects.Resilience.Lorenz import main

    main(plotting=False)
