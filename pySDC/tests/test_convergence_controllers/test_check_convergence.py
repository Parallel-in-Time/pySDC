import pytest


def run_heat(maxiter=99, restol=-1, e_tol=-1):
    from pySDC.implementations.problem_classes.HeatEquation_ND_FD import heatNd_unforced
    from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.hooks.log_embedded_error_estimate import LogEmbeddedErrorEstimatePostIter

    # initialize level parameters
    level_params = dict()
    level_params['dt'] = 0.05
    level_params['restol'] = restol
    level_params['e_tol'] = e_tol

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'IE'

    problem_params = {
        'freq': 2,
        'nvars': 2**9,
        'nu': 1.0,
        'stencil_type': 'center',
        'order': 6,
        'bc': 'periodic',
        'solver_type': 'direct',
        'lintol': None,
        'liniter': None,
    }

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = maxiter

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = LogEmbeddedErrorEstimatePostIter

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = heatNd_unforced
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(0.0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=0.0, Tend=level_params['dt'])

    from pySDC.helpers.stats_helper import get_list_of_types

    # residual = np.max([me[1] for me in get_sorted(stats, type='residual_post_step')
    return stats, controller


@pytest.mark.base
@pytest.mark.parametrize("maxiter", [1, 5, 50])
def test_convergence_by_iter(maxiter):
    import numpy as np
    from pySDC.helpers.stats_helper import get_sorted

    stats, _ = run_heat(maxiter=maxiter)
    niter = np.mean([me[1] for me in get_sorted(stats, type='niter')])
    assert niter == maxiter, f"Wrong number of iterations! Expected {maxiter}, but got {niter}!"


@pytest.mark.base
@pytest.mark.parametrize("e_tol", [1e-3, 1e-5, 1e-10])
def test_convergence_by_increment(e_tol):
    import numpy as np
    from pySDC.helpers.stats_helper import get_sorted

    stats, _ = run_heat(e_tol=e_tol)

    e_em = [me[1] for me in get_sorted(stats, type='error_embedded_estimate_post_iteration', sortby='iter')]

    e_em_before_convergence = np.min(e_em[:-1])
    e_em_at_convergence = e_em[-1]

    assert e_em_before_convergence > e_tol, "Embedded error estimate was below threshold before convergence!"
    assert e_em_at_convergence <= e_tol, "Step terminated before convergence by increment was achieved!"


@pytest.mark.base
@pytest.mark.parametrize("restol", [1e-3, 1e-5, 1e-10])
def test_convergence_by_residual(restol):
    import numpy as np
    from pySDC.helpers.stats_helper import get_sorted

    stats, _ = run_heat(restol=restol)

    res = [me[1] for me in get_sorted(stats, type='residual_post_iteration', sortby='iter')]

    res_before_convergence = np.min(res[:-1])
    res_at_convergence = res[-1]

    assert res_before_convergence > restol, "Residual was below threshold before convergence!"
    assert res_at_convergence <= restol, "Step terminated before convergence by residual was achieved!"
