import pytest


@pytest.mark.base
def test_main():
    from pySDC.projects.Resilience.piline import main

    main()
