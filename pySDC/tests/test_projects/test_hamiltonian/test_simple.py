import pytest


@pytest.mark.base
@pytest.mark.slow
def test_main():
    from pySDC.projects.Hamiltonian.simple_problems import main

    main()
