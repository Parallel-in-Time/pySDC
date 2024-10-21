import pytest


@pytest.mark.mpi4py
@pytest.mark.parametrize('Tend', [1.2345, 1, 4.0 / 3.0])
@pytest.mark.parametrize('dt', [0.1, 1.0 / 3.0, 0.999999])
def test_ReachTendExactly(Tend, dt, num_procs=1):
    from pySDC.projects.Resilience.RBC import ReachTendExactly
    from pySDC.implementations.hooks.log_solution import LogSolution
    from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.helpers.stats_helper import get_sorted
    import numpy as np

    level_params = {}
    level_params['dt'] = dt

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 2
    sweeper_params['QI'] = 'LU'

    problem_params = {
        'mu': 0.0,
        'newton_tol': 1e-1,
        'newton_maxiter': 9,
        'u0': np.array([2.0, 0.0]),
        'relative_tolerance': True,
    }

    step_params = {}
    step_params['maxiter'] = 2

    convergence_controllers = {ReachTendExactly: {'Tend': Tend}}

    controller_params = {}
    controller_params['logger_level'] = 15
    controller_params['hook_class'] = LogSolution
    controller_params['mssdc_jac'] = False

    description = {}
    description['problem_class'] = vanderpol
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['convergence_controllers'] = convergence_controllers

    t0 = 0.0

    controller = controller_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)

    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    u = get_sorted(stats, type='u')
    t_last = u[-1][0]

    assert np.isclose(
        t_last, Tend, atol=1e-10
    ), f'Expected {Tend=}, but got {t_last}, which is off by {t_last - Tend:.8e}'


if __name__ == '__main__':
    test_ReachTendExactly(1.2345, 0.1)
