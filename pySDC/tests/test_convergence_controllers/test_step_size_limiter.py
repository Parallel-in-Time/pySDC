import pytest


def get_controller(step_size_limier_params):
    """
    Runs a single advection problem with certain parameters

    Args:
        step_size_limier_params (dict): Parameters for convergence controller

    Returns:
       (pySDC.Controller.controller): Controller used in the run
    """
    from pySDC.implementations.problem_classes.polynomial_test_problem import polynomial_testequation
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.convergence_controller_classes.step_size_limiter import StepSizeLimiter

    level_params = {}
    level_params['dt'] = 1.0
    level_params['restol'] = 1.0

    sweeper_params = {}
    sweeper_params['quad_type'] = 'GAUSS'
    sweeper_params['num_nodes'] = 1
    sweeper_params['do_coll_update'] = True

    problem_params = {'degree': 10}

    step_params = {}
    step_params['maxiter'] = 0

    controller_params = {}
    controller_params['logger_level'] = 30

    description = {}
    description['problem_class'] = polynomial_testequation
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['convergence_controllers'] = {StepSizeLimiter: step_size_limier_params}

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    controller.add_convergence_controller(StepSizeLimiter, description, step_size_limier_params)

    return controller


@pytest.mark.base
def test_step_size_slope_limiter():
    from pySDC.implementations.convergence_controller_classes.step_size_limiter import StepSizeSlopeLimiter

    params = {'dt_slope_max': 2, 'dt_slope_min': 1e-3, 'dt_rel_min_slope': 1e-1}
    controller = get_controller(params)

    limiter = controller.convergence_controllers[
        [type(me) for me in controller.convergence_controllers].index(StepSizeSlopeLimiter)
    ]

    S = controller.MS[0]
    S.status.slot = 0
    L = S.levels[0]
    L.status.time = 0

    L.params.dt = 1
    L.status.dt_new = 3
    limiter.get_new_step_size(controller, S)
    assert L.status.dt_new == 2

    L.params.dt = 1
    L.status.dt_new = 0
    limiter.get_new_step_size(controller, S)
    assert L.status.dt_new == 1e-3

    L.params.dt = 1
    L.status.dt_new = 1 + 1e-3
    limiter.get_new_step_size(controller, S)
    assert L.status.dt_new == 1


@pytest.mark.base
def test_step_size_limiter():
    from pySDC.implementations.convergence_controller_classes.step_size_limiter import StepSizeLimiter

    params = {'dt_max': 2, 'dt_min': 0.5}
    controller = get_controller(params)

    limiter = controller.convergence_controllers[
        [type(me) for me in controller.convergence_controllers].index(StepSizeLimiter)
    ]

    S = controller.MS[0]
    S.status.slot = 0
    L = S.levels[0]
    L.status.time = 0

    L.params.dt = 1
    L.status.dt_new = 3
    limiter.get_new_step_size(controller, S)
    assert L.status.dt_new == 2

    L.params.dt = 1
    L.status.dt_new = 0
    limiter.get_new_step_size(controller, S)
    assert L.status.dt_new == 0.5


@pytest.mark.base
@pytest.mark.parametrize('dt', [1 / 3, 2 / 30])
def test_step_size_rounding(dt):
    from pySDC.implementations.convergence_controller_classes.step_size_limiter import StepSizeRounding

    expect = {
        1 / 3: 0.3,
        2 / 30: 0.065,
    }
    assert StepSizeRounding._round_step_size(dt, 5, 1) == expect[dt]


if __name__ == '__main__':
    test_step_size_slope_limiter()
