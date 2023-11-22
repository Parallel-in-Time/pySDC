import pytest


def run_problem(maxiter=1, num_procs=1, n_steps=1, error_estimator=None, params=None, restol=-1):
    import numpy as np
    from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.hooks.log_errors import (
        LogLocalErrorPostIter,
        LogGlobalErrorPostIter,
        LogLocalErrorPostStep,
    )

    # initialize level parameters
    level_params = {}
    level_params['dt'] = 6e-3
    level_params['restol'] = restol

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'IE'
    # sweeper_params['initial_guess'] = 'random'

    # build lambdas
    re = np.linspace(-30, -1, 10)
    im = np.linspace(-50, 50, 11)
    lambdas = np.array([[complex(re[i], im[j]) for i in range(len(re))] for j in range(len(im))]).reshape(
        (len(re) * len(im))
    )

    problem_params = {
        'lambdas': lambdas,
        'u0': 1.0 + 0.0j,
    }

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = maxiter

    # convergence controllers
    convergence_controllers = {error_estimator: params}

    # initialize controller parameters
    controller_params = {}
    controller_params['logger_level'] = 15
    controller_params['hook_class'] = [LogLocalErrorPostIter, LogGlobalErrorPostIter, LogLocalErrorPostStep]
    controller_params['mssdc_jac'] = False

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_class'] = testequation0d
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['convergence_controllers'] = convergence_controllers

    # set time parameters
    t0 = 0.0

    # instantiate controller
    controller = controller_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=n_steps * level_params['dt'])
    return stats


@pytest.mark.base
def test_EstimateExtrapolationErrorNonMPI_serial(order_time_marching=2, n_steps=3, thresh=0.15):
    from pySDC.implementations.convergence_controller_classes.estimate_extrapolation_error import (
        EstimateExtrapolationErrorNonMPI,
    )
    from pySDC.helpers.stats_helper import get_sorted, filter_stats, sort_stats

    params = {
        'no_storage': False,
    }
    preperatory_steps = (order_time_marching + 3) // 2

    stats = run_problem(
        maxiter=order_time_marching,
        n_steps=n_steps + preperatory_steps,
        error_estimator=EstimateExtrapolationErrorNonMPI,
        params=params,
        num_procs=1,
    )

    e_local = sort_stats(filter_stats(stats, type='e_local_post_iteration', iter=order_time_marching), sortby='time')
    e_estimated = get_sorted(stats, type='error_extrapolation_estimate')

    rel_diff = [
        abs(e_local[i][1] - e_estimated[i][1]) / e_estimated[i][1]
        for i in range(len(e_estimated))
        if e_estimated[i][1] is not None
    ]
    assert all(
        me < thresh for me in rel_diff
    ), f'Extrapolated error estimate failed! Relative difference to true error: {rel_diff}'


@pytest.mark.base
@pytest.mark.parametrize('no_storage', [True, False])
def test_EstimateExtrapolationErrorNonMPI_parallel(
    no_storage, order_time_marching=4, n_steps=3, num_procs=3, thresh=0.50
):
    from pySDC.implementations.convergence_controller_classes.estimate_extrapolation_error import (
        EstimateExtrapolationErrorNonMPI,
    )
    from pySDC.helpers.stats_helper import get_sorted, filter_stats, sort_stats

    params = {
        'no_storage': no_storage,
    }
    preperatory_steps = (order_time_marching + 3) // 2

    if no_storage:
        num_procs = max(num_procs, preperatory_steps + 1)

    stats = run_problem(
        maxiter=order_time_marching,
        n_steps=n_steps + preperatory_steps,
        error_estimator=EstimateExtrapolationErrorNonMPI,
        params=params,
        num_procs=num_procs,
    )

    e_local = sort_stats(filter_stats(stats, type='e_local_post_iteration', iter=order_time_marching), sortby='time')
    e_estimated = get_sorted(stats, type='error_extrapolation_estimate')

    rel_diff = [
        abs(e_local[i][1] - e_estimated[i][1]) / e_local[i][1]
        for i in range(len(e_estimated))
        if e_estimated[i][1] is not None
    ]
    assert all(
        me < thresh for me in rel_diff
    ), f'Extrapolated error estimate failed! Relative difference to true error: {rel_diff}'


@pytest.mark.base
def test_EstimateEmbeddedErrorSerial(order_time_marching=3, n_steps=6, thresh=0.05):
    from pySDC.implementations.convergence_controller_classes.estimate_embedded_error import EstimateEmbeddedError
    from pySDC.helpers.stats_helper import get_sorted, filter_stats, sort_stats

    params = {}

    stats = run_problem(
        maxiter=order_time_marching, n_steps=n_steps, error_estimator=EstimateEmbeddedError, params=params, num_procs=1
    )

    e_local = sort_stats(
        filter_stats(stats, type='e_local_post_iteration', iter=order_time_marching - 1), sortby='time'
    )
    e_estimated = get_sorted(stats, type='error_embedded_estimate')

    rel_diff = [abs(e_local[i][1] - e_estimated[i][1]) / e_local[i][1] for i in range(len(e_estimated))]

    assert all(
        me < thresh for me in rel_diff
    ), f'Embedded error estimate failed! Relative difference to true error: {rel_diff}'


@pytest.mark.base
def test_EstimateEmbeddedErrorParallel(order_time_marching=3, num_procs=3, thresh=0.10):
    from pySDC.implementations.convergence_controller_classes.estimate_embedded_error import EstimateEmbeddedError
    from pySDC.helpers.stats_helper import get_sorted, filter_stats, sort_stats

    params = {}

    stats = run_problem(
        maxiter=order_time_marching,
        n_steps=num_procs,
        error_estimator=EstimateEmbeddedError,
        params=params,
        num_procs=num_procs,
    )

    e_global = sort_stats(
        filter_stats(stats, type='e_global_post_iteration', iter=order_time_marching - 1), sortby='time'
    )
    e_estimated = get_sorted(stats, type='error_embedded_estimate')

    rel_diff = [abs(e_global[i][1] - e_estimated[i][1]) / e_global[i][1] for i in range(len(e_estimated))]

    assert all(
        me < thresh for me in rel_diff
    ), f'Embedded error estimate failed! Relative difference to true error: {rel_diff}'


@pytest.mark.base
def test_EstimateEmbeddedErrorCollocation(n_steps=6, thresh=0.01):
    from pySDC.implementations.convergence_controller_classes.estimate_embedded_error import (
        EstimateEmbeddedErrorCollocation,
    )
    from pySDC.helpers.stats_helper import get_sorted, filter_stats, sort_stats

    adaptive_coll_params = {
        'num_nodes': [3, 2],
    }
    params = {'adaptive_coll_params': adaptive_coll_params}

    stats = run_problem(
        maxiter=99,
        n_steps=n_steps,
        error_estimator=EstimateEmbeddedErrorCollocation,
        params=params,
        num_procs=1,
        restol=1e-13,
    )

    e_estimated = get_sorted(stats, type='error_embedded_estimate_collocation')
    e_local = sort_stats(filter_stats(stats, type='e_local_post_step'), sortby='time')

    rel_diff = [abs(e_local[i][1] - e_estimated[i][1]) / e_local[i][1] for i in range(len(e_estimated))]

    assert all(
        me < thresh for me in rel_diff
    ), f'Embedded error estimate failed! Relative difference to true error: {rel_diff}'


if __name__ == '__main__':
    test_EstimateEmbeddedErrorCollocation()
