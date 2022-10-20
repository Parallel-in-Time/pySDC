import pytest


@pytest.mark.base
def test_main():
    from pySDC.projects.Resilience.accuracy_check import main

    main()
