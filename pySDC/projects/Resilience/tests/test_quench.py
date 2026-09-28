import pytest


@pytest.mark.base
@pytest.mark.parametrize('leak_type', ['linear', 'exponential'])
def test_imex_vs_fully_implicit_quench(leak_type):
    """
    Test if the IMEX and fully implicit schemes get the same solution and that the runaway process has started.
    """
    from pySDC.projects.Resilience.quench import compare_imex_full

    compare_imex_full(plotting=False, leak_type=leak_type)


@pytest.mark.base
def test_crossing_time_computation():
    import numpy as np
    from pySDC.projects.Resilience.quench import run_quench, get_crossing_time
    from pySDC.helpers.stats_helper import get_sorted

    controller_params = {'logger_level': 30}
    description = {'level_params': {'dt': 2.5e1}, 'step_params': {'maxiter': 5}}
    stats, controller, _ = run_quench(
        custom_controller_params=controller_params,
        custom_description=description,
        Tend=400,
    )
    t_cross = get_crossing_time(stats, controller, num_points=5, inter_points=155)

    u = get_sorted(stats, type='u', recomputed=False)
    t = np.array([me[0] for me in u])
    temp = np.array([np.mean(me[1]) for me in u])
    first_above = np.argmax(temp > controller.MS[0].levels[0].prob.u_thresh)
    assert t[first_above - 1] < t_cross <= t[first_above], 'Crossing time is not between the steps that cross'

    # a run with dt=5 crosses at 322.53, this one at 322.73; the step size is 25
    assert np.isclose(t_cross, 322.5, atol=0.5), f'Unexpected crossing time {t_cross}'
