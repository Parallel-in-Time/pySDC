import pytest


@pytest.mark.base
def test_main():
    from pySDC.projects.Resilience.collocation_adaptivity import main

    main()
