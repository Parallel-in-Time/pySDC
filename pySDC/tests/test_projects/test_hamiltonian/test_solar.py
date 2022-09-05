import pytest


@pytest.mark.base
def test_main():
    from pySDC.projects.Hamiltonian.solar_system import main

    main()
