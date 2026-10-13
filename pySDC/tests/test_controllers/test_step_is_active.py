import pytest


@pytest.mark.base
@pytest.mark.parametrize('num_steps', [91, 181])
@pytest.mark.parametrize('num_procs', [1, 4])
def test_no_extra_step_past_Tend(num_steps, num_procs):
    """
    Accumulating `t += 2 / 91` 91 times gives 1.999999999999997, which the old absolute tolerance
    of 10 eps took as a start before Tend = 2, running a spurious 92nd step to 2 + dt.
    """
    import numpy as np

    from pySDC.helpers.stats_helper import get_sorted
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit

    Tend = 2.0
    dt = Tend / num_steps
    description = {
        'problem_class': testequation0d,
        'problem_params': {'lambdas': np.array([-1.0]), 'u0': 1.0},
        'sweeper_class': generic_implicit,
        'sweeper_params': {'quad_type': 'RADAU-RIGHT', 'num_nodes': 2, 'QI': 'LU'},
        'level_params': {'dt': dt},
        'step_params': {'maxiter': 1},
    }
    controller = controller_nonMPI(num_procs, {'logger_level': 30}, description)
    _, stats = controller.run(controller.MS[0].levels[0].prob.u_exact(0.0), 0.0, Tend)

    starts = [me[0] for me in get_sorted(stats, type='niter')]
    assert len(starts) == num_steps
    assert np.isclose(starts[-1] + dt, Tend, rtol=0, atol=1e-13)
