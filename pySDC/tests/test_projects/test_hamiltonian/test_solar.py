import pytest


@pytest.mark.base
@pytest.mark.parametrize("prob", ['outer_solar_system', 'full_solar_system'])
def test_main(prob):
    from pySDC.projects.Hamiltonian.solar_system import run_simulation, show_results

    run_simulation(prob)
    show_results(prob)
