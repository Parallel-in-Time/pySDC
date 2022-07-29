import pytest

from pySDC.projects.Hamiltonian.simple_problems import main


@pytest.mark.slow
def test_main():
    main()
