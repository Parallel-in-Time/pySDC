import pytest


@pytest.mark.base
@pytest.mark.parametrize('leak_type', ['linear', 'exponential'])
def test_imex_vs_fully_implicit_leaky_superconductor(leak_type):
    """
    Test if the IMEX and fully implicit schemes get the same solution and that the runaway process has started.
    """
    from pySDC.projects.Resilience.leaky_superconductor import compare_imex_full

    compare_imex_full(plotting=False, leak_type=leak_type)


@pytest.mark.base
def test_crossing_time_computation():
    from pySDC.projects.Resilience.leaky_superconductor import run_leaky_superconductor, get_crossing_time

    controller_params = {'logger_level': 30}
    description = {'level_params': {'dt': 2.5e1}, 'step_params': {'maxiter': 5}}
    stats, controller, _ = run_leaky_superconductor(
        custom_controller_params=controller_params,
        custom_description=description,
        Tend=400,
    )
    _ = get_crossing_time(stats, controller, num_points=5, inter_points=155)
