import pytest


def get_composite_collocation_problem(L, M, N, alpha=0, dt=1e-1, problem='Dahlquist', ParaDiag=True):
    import numpy as np
    from pySDC.implementations.hooks.log_errors import (
        LogGlobalErrorPostRun,
        LogGlobalErrorPostStep,
        LogGlobalErrorPostIter,
    )

    if ParaDiag:
        from pySDC.implementations.controller_classes.controller_ParaDiag_nonMPI import (
            controller_ParaDiag_nonMPI as controller_class,
        )
    else:
        from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI as controller_class

    average_jacobian = False
    restol = 1e-8

    if problem == 'Dahlquist':
        from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d as problem_class

        if ParaDiag:
            from pySDC.implementations.sweeper_classes.ParaDiagSweepers import QDiagonalization as sweeper_class
        else:
            from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_class

        problem_params = {'lambdas': -1.0 * np.ones(shape=(N)), 'u0': 1}
    elif problem == 'Dahlquist_IMEX':
        from pySDC.implementations.problem_classes.TestEquation_0D import test_equation_IMEX as problem_class

        if ParaDiag:
            from pySDC.implementations.sweeper_classes.ParaDiagSweepers import QDiagonalizationIMEX as sweeper_class
        else:
            from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order as sweeper_class

        problem_params = {
            'lambdas_implicit': -1.0 * np.ones(shape=(N)),
            'lambdas_explicit': -1.0e-1 * np.ones(shape=(N)),
            'u0': 1.0,
        }
    elif problem == 'heat':
        from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_forced as problem_class
        from pySDC.implementations.sweeper_classes.ParaDiagSweepers import QDiagonalizationIMEX as sweeper_class

        problem_params = {'nvars': N}
    elif problem == 'vdp':
        from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol as problem_class

        if ParaDiag:
            from pySDC.implementations.sweeper_classes.ParaDiagSweepers import QDiagonalization as sweeper_class

            problem_params = {'newton_maxiter': 1, 'mu': 1e0, 'crash_at_maxiter': False}
            average_jacobian = True
        else:
            from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_class

            problem_params = {'newton_maxiter': 99, 'mu': 1e0, 'crash_at_maxiter': True}
    else:
        raise NotImplementedError()

    level_params = {}
    level_params['dt'] = dt
    level_params['restol'] = restol

    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = M
    sweeper_params['initial_guess'] = 'spread'

    step_params = {}
    step_params['maxiter'] = 99

    controller_params = {}
    controller_params['logger_level'] = 15
    controller_params['hook_class'] = [LogGlobalErrorPostRun, LogGlobalErrorPostStep, LogGlobalErrorPostIter]
    controller_params['mssdc_jac'] = False
    controller_params['alpha'] = alpha
    controller_params['average_jacobian'] = average_jacobian

    description = {}
    description['problem_class'] = problem_class
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper_class
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    controller_args = {
        'controller_params': controller_params,
        'description': description,
    }
    controller = controller_class(**controller_args, num_procs=L)
    P = controller.MS[0].levels[0].prob

    for prob in [S.levels[0].prob for S in controller.MS]:
        prob.init = tuple([*prob.init[:2]] + [np.dtype('complex128')])

    return controller, P


@pytest.mark.base
@pytest.mark.parametrize('L', [1, 4])
@pytest.mark.parametrize('M', [2, 3])
@pytest.mark.parametrize('N', [2])
@pytest.mark.parametrize('alpha', [1e-4, 1e-2])
@pytest.mark.parametrize('problem', ['Dahlquist', 'Dahlquist_IMEX', 'vdp'])
def test_ParaDiag_convergence(L, M, N, alpha, problem):
    from pySDC.helpers.stats_helper import get_sorted

    controller, prob = get_composite_collocation_problem(L, M, N, alpha, problem=problem)
    level = controller.MS[0].levels[0]

    # setup initial conditions
    u0 = prob.u_exact(0)

    uend, stats = controller.run(u0=u0, t0=0, Tend=L * level.dt * 2)

    # make some tests
    error = get_sorted(stats, type='e_global_post_step')
    k = get_sorted(stats, type='niter')
    assert max(me[1] for me in k) < 90, 'ParaDiag did not converge'
    if problem in ['Dahlquist', 'Dahlquist_IMEX']:
        assert max(me[1] for me in error) < 1e-5, 'Error with ParaDiag too large'


@pytest.mark.base
@pytest.mark.parametrize('L', [1, 4])
@pytest.mark.parametrize('M', [2, 3])
@pytest.mark.parametrize('N', [64])
@pytest.mark.parametrize('alpha', [1e-4, 1e-2])
def test_IMEX_ParaDiag_convergence(L, M, N, alpha):
    from pySDC.helpers.stats_helper import get_sorted

    controller, prob = get_composite_collocation_problem(L, M, N, alpha, problem='heat', dt=1e-3)
    level = controller.MS[0].levels[0]

    # setup initial conditions
    u0 = prob.u_exact(0)

    uend, stats = controller.run(u0=u0, t0=0, Tend=L * level.dt * 2)

    # make some tests
    error = get_sorted(stats, type='e_global_post_step')
    k = get_sorted(stats, type='niter')
    assert max(me[1] for me in k) < 9, 'ParaDiag did not converge'
    assert max(me[1] for me in error) < 1e-4, 'Error with ParaDiag too large'


@pytest.mark.base
@pytest.mark.parametrize('L', [1, 4])
@pytest.mark.parametrize('M', [1, 2])
@pytest.mark.parametrize('N', [2])
@pytest.mark.parametrize('problem', ['Dahlquist', 'Dahlquist_IMEX', 'vdp'])
def test_ParaDiag_vs_PFASST(L, M, N, problem):
    import numpy as np

    alpha = 1e-4

    # setup the same composite collocation problem with different solvers
    controllerParaDiag, prob = get_composite_collocation_problem(L, M, N, alpha, problem=problem, ParaDiag=True)
    controllerPFASST, _ = get_composite_collocation_problem(L, M, N, alpha, problem=problem, ParaDiag=False)
    level = controllerParaDiag.MS[0].levels[0]

    # setup initial conditions
    u0 = prob.u_exact(0)
    Tend = L * 2 * level.dt

    # run the two different solvers for the composite collocation problem
    uendParaDiag, _ = controllerParaDiag.run(u0=u0, t0=0, Tend=Tend)
    uendPFASST, _ = controllerPFASST.run(u0=u0, t0=0, Tend=Tend)

    assert np.allclose(
        uendParaDiag, uendPFASST
    ), f'Got different solutions between single-level PFASST and ParaDiag with {problem=}'
    # make sure we didn't trick ourselves with a bug in the test...
    assert (
        abs(uendParaDiag - uendPFASST) > 0
    ), 'The solutions with PFASST and ParaDiag are unexpectedly exactly the same!'


@pytest.mark.base
@pytest.mark.parametrize('L', [4])
@pytest.mark.parametrize('M', [2, 3])
@pytest.mark.parametrize('N', [1])
@pytest.mark.parametrize('alpha', [1e-4, 1e-2])
def test_ParaDiag_order(L, M, N, alpha):
    import numpy as np
    from pySDC.helpers.stats_helper import get_sorted

    errors = []
    if M == 3:
        dts = [0.8 * 2 ** (-x) for x in range(7, 9)]
    elif M == 2:
        dts = [2 ** (-x) for x in range(5, 9)]
    else:
        raise NotImplementedError
    Tend = max(dts) * L * 2

    for dt in dts:
        controller, prob = get_composite_collocation_problem(L, M, N, alpha, dt=dt)
        level = controller.MS[0].levels[0]

        # setup initial conditions
        u0 = prob.u_init
        u0[:] = 1

        uend, stats = controller.run(u0=u0, t0=0, Tend=Tend)

        # make some tests
        errors.append(get_sorted(stats, type='e_global_post_run')[-1][1])

        expected_order = level.sweep.coll.order

    errors = np.array(errors)
    dts = np.array(dts)
    order = np.log(abs(errors[1:] - errors[:-1])) / np.log(abs(dts[1:] - dts[:-1]))
    num_order = np.mean(order)

    assert (
        expected_order + 1 > num_order > expected_order
    ), f'Got unexpected numerical order {num_order} instead of {expected_order} in ParaDiag'


@pytest.mark.base
@pytest.mark.parametrize('L', [4, 12])
@pytest.mark.parametrize('M', [2, 3])
@pytest.mark.parametrize('N', [1])
@pytest.mark.parametrize('alpha', [1e-4, 1e-2])
def test_ParaDiag_convergence_rate(L, M, N, alpha):
    r"""
    Test that the error in ParaDiag contracts as fast as expected.

    The upper bound is \|u^{k+1} - u^*\| / \|u^k - u^*\| < \alpha / (1-\alpha).
    Here, we compare to the exact solution to the continuous problem rather than the exact solution of the collocation
    problem, which means the error stalls at time-discretization level. Therefore, we only check the contraction in the
    first ParaDiag iteration.
    """
    import numpy as np
    from pySDC.helpers.stats_helper import get_sorted

    dt = 1e-2
    controller, prob = get_composite_collocation_problem(L, M, N, alpha, dt=dt, problem='Dahlquist')

    # setup initial conditions
    u0 = prob.u_exact(0)

    uend, stats = controller.run(u0=u0, t0=0, Tend=L * dt)

    # test that the convergence rate in the first iteration is sufficiently small.
    t_last = max([me[0] for me in get_sorted(stats, type='e_global_post_iteration')])
    errors = get_sorted(stats, type='e_global_post_iteration', sortby='iter', time=t_last)
    convergence_rates = [errors[i + 1][1] / errors[i][1] for i in range(len(errors) - 1)]
    convergence_rate = convergence_rates[0]
    convergence_bound = alpha / (1 - alpha)

    assert (
        convergence_rate < convergence_bound
    ), f'Convergence rate {convergence_rate} exceeds upper bound of {convergence_bound}!'


if __name__ == '__main__':
    test_ParaDiag_convergence_rate(4, 3, 1, 1e-4)
    # test_ParaDiag_vs_PFASST(4, 3, 2, 'Dahlquist')
    # test_ParaDiag_convergence(4, 3, 1, 1e-4, 'vdp')
    # test_IMEX_ParaDiag_convergence(4, 3, 64, 1e-4)
    # test_ParaDiag_order(3, 3, 1, 1e-4)
