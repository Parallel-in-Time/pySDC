import pytest


@pytest.mark.base
def test_main():
    from pySDC.projects.Resilience.piline import main

    main()


@pytest.mark.base
def test_residual_adaptivity():
    from pySDC.projects.Resilience.piline import residual_adaptivity

    residual_adaptivity()
