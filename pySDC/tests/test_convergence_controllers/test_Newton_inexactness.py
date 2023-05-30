import pytest


@pytest.mark.base
def test_Newton_inexactness(ratio=1e-2, min_tol=1e-11, max_tol=1e-6):
    import numpy as np
    from pySDC.implementations.convergence_controller_classes.inexactness import NewtonInexactness
    from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.helpers.stats_helper import get_sorted, filter_stats
    from pySDC.core.Hooks import hooks

    class log_newton_tol(hooks):
        def pre_iteration(self, step, level_number):
            lvl = step.levels[level_number]
            self.add_to_stats(
                process=step.status.slot,
                time=step.time,
                level=level_number,
                iter=step.status.iter,
                sweep=lvl.status.sweep,
                type='newton_tol_post_spread',
                value=lvl.prob.newton_tol,
            )

        def post_iteration(self, step, level_number):
            lvl = step.levels[level_number]
            self.add_to_stats(
                process=step.status.slot,
                time=step.time,
                level=level_number,
                iter=step.status.iter,
                sweep=lvl.status.sweep,
                type='newton_tol',
                value=lvl.prob.newton_tol,
            )

    # initialize level parameters
    level_params = {}
    level_params['dt'] = 1e-2
    level_params['restol'] = 1e-10

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'LU'

    problem_params = {
        'mu': 5.0,
        'newton_tol': 1e-9,
        'newton_maxiter': 99,
        'u0': np.array([2.0, 0.0]),
    }

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 99

    # initialize controller parameters
    controller_params = {}
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = log_newton_tol
    controller_params['mssdc_jac'] = False

    convergence_controllers = {}
    convergence_controllers[NewtonInexactness] = {'ratio': ratio, 'min_tol': min_tol, 'max_tol': max_tol}

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_class'] = vanderpol
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['convergence_controllers'] = convergence_controllers

    # set time parameters
    t0 = 0.0

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=0, Tend=2 * level_params['dt'])

    for me in get_sorted(stats, type='newton_tol'):
        stats_now = filter_stats(stats, time=me[0])
        tols = get_sorted(stats_now, type='newton_tol', sortby='iter')
        res = get_sorted(stats_now, type='residual_post_iteration', sortby='iter')

        for i in range(len(tols) - 1):
            expect = res[i][1] * ratio
            assert (
                tols[i + 1][1] <= expect or expect < min_tol
            ), f'Expected Newton tolerance smaller {expect:.2e}, but got {tols[i+1][1]:.2e} in iteration {i+1}!'
            assert (
                tols[i + 1][1] <= max_tol
            ), f'Exceeded maximal allowed Newton tolerance {max_tol:.2e} in iteration {i+1} with {tols[i+1][1]:.2e}!'


if __name__ == "__main__":
    test_Newton_inexactness()
