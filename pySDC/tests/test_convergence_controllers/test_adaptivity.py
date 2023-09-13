import pytest


def get_controller(dt, num_nodes, useMPI, adaptivity, adaptivity_params, **kwargs):
    """
    Runs a single advection problem with certain parameters

    Args:
        dt (float): Step size
        num_nodes (int): Number of nodes
        useMPI (bool): Whether or not to use MPI
        adaptivity (pySDC.ConvergenceController): Adaptivity convergence controller
        adaptivity_params (dict): Parameters for convergence controller

    Returns:
       (dict): Stats object generated during the run
       (pySDC.Controller.controller): Controller used in the run
    """
    from pySDC.implementations.problem_classes.polynomial_test_problem import polynomial_testequation
    from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
    from pySDC.implementations.convergence_controller_classes.estimate_extrapolation_error import (
        EstimateExtrapolationErrorWithinQ,
    )

    if useMPI:
        from pySDC.implementations.sweeper_classes.generic_implicit_MPI import generic_implicit_MPI as sweeper_class
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
    else:
        from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit as sweeper_class

        comm = None

    # initialize level parameters
    level_params = {}
    level_params['dt'] = dt
    level_params['restol'] = 1.0

    # initialize sweeper parameters
    sweeper_params = {}
    sweeper_params['quad_type'] = 'GAUSS'
    sweeper_params['num_nodes'] = num_nodes
    sweeper_params['do_coll_update'] = True
    sweeper_params['comm'] = comm

    problem_params = {'degree': 10}

    # initialize step parameters
    step_params = {}
    step_params['maxiter'] = 0

    # initialize controller parameters
    controller_params = {}
    controller_params['logger_level'] = 30
    controller_params['mssdc_jac'] = False

    # fill description dictionary for easy step instantiation
    description = {}
    description['problem_class'] = polynomial_testequation
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper_class
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params
    description['convergence_controllers'] = {adaptivity: adaptivity_params}

    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    controller.add_convergence_controller(adaptivity, description, adaptivity_params)

    try:
        [
            me.reset_status_variables(controller, MS=controller.MS)
            for me in controller.convergence_controllers
            if type(me) not in controller.base_convergence_controllers
        ]
    except ValueError:
        pass

    return controller


def single_test(adaptivity, num_experiments=3, **kwargs):
    """
    Run a single test where the solution is replaced by a polynomial and the nodes are changed.
    Because we know the polynomial going in, we can check if the interpolation based change was
    exact. If the solution is not a polynomial or a polynomial of higher degree then the number
    of nodes, the change in nodes does add some error, of course, but here it is on the order of
    machine precision.
    """
    import numpy as np

    args = {
        'num_nodes': 3,
        'useMPI': False,
        'dt': 1.0,
        'adaptivity': adaptivity,
        'adaptivity_params': {},
        **kwargs,
    }
    error_estimate = []
    error = []
    dts = [args['dt']]

    for _n in range(num_experiments):
        # prepare variables
        controller = get_controller(**{**args, 'dt': dts[-1]})
        step = controller.MS[0]
        level = step.levels[0]
        prob = level.prob
        cont = controller.convergence_controllers[
            np.arange(len(controller.convergence_controllers))[
                [type(me).__name__ == adaptivity.__name__ for me in controller.convergence_controllers]
            ][0]
        ]

        nodes = np.append([0], level.sweep.coll.nodes)

        # initialize variables
        step.status.slot = 0
        step.status.iter = 1
        level.status.time = 0.0
        level.status.residual = 0.0
        level.u[0] = prob.u_exact(t=0)
        level.sweep.predict()

        for i in range(len(level.u)):
            if level.u[i] is not None:
                level.u[i][:] = prob.u_exact(nodes[i] * level.dt)

        level.sweep.compute_end_point()

        # perform the interpolation
        for me in controller.convergence_controllers:
            me.post_iteration_processing(controller, step)
        cont.get_new_step_size(controller, step)

        error_estimate += [
            level.status.get('error_extrapolation_estimate', level.status.get('error_embedded_estimate', None))
        ]
        error += [abs(prob.u_exact(level.dt) - level.uend)]
        dts += [level.status.dt_new]

    return {'est': error_estimate, 'e': error, 'dt': dts}


def multiple_tests(adaptivity, e_tol_range, num_nodes, **kwargs):
    import numpy as np

    e_tol_range = np.asarray(e_tol_range)
    res = []
    for e_tol in e_tol_range:
        res += [single_test(adaptivity, num_nodes=num_nodes, adaptivity_params={'e_tol': e_tol}, **kwargs)]

    # check the final results
    error_estimates = np.array([me['est'][-1] for me in res])
    errors = np.array([me['e'][-1] for me in res])
    dts = np.array([me['dt'][-2] for me in res])

    not_passed = []

    if not all(
        e_tol_range[i] > error_estimates[i] > e_tol_range[i] * 1e-1
        for i in range(len(e_tol_range))
        if error_estimates[i] < 1e-1
    ):
        not_passed += [(0, f'Error estimates don\'t fall in expected range after adaptivity! Got {error_estimates}')]

    if not all(error_estimates[i] > errors[i] for i in range(len(e_tol_range)) if 4e-16 < errors[i] < 1e-2):
        not_passed += [(1, f'Errors larger than estimates! Got {errors}')]

    # check orders relative to step size
    expected_order_estimate = num_nodes
    expected_order_error = 2 * num_nodes + 1

    mask = np.logical_and(errors < 1e-0, errors > 3e-16)
    order = np.log(errors[mask][1:] / errors[mask][:-1]) / np.log(dts[mask][1:] / dts[mask][:-1])
    order_estimate = np.log(error_estimates[1:] / error_estimates[:-1]) / np.log(dts[1:] / dts[:-1])

    if not np.isclose(np.median(order), expected_order_error, atol=0.4):
        not_passed += [(2, f'Expected order {expected_order_error}, but got {order}!')]
    if not np.isclose(np.median(order_estimate), expected_order_estimate, atol=0.6):
        not_passed += [
            (3, f'Expected order {expected_order_estimate} in the estimate, but got {np.median(order_estimate)}!')
        ]

    # check orders relative to e_tol
    expected_order_estimate_e_tol = 1.0
    expected_order_error_e_tol = expected_order_error / expected_order_estimate

    order_e_tol = np.log(errors[mask][1:] / errors[mask][:-1]) / np.log(e_tol_range[mask][1:] / e_tol_range[mask][:-1])
    order_estimate_e_tol = np.log(error_estimates[1:] / error_estimates[:-1]) / np.log(
        e_tol_range[1:] / e_tol_range[:-1]
    )

    if not np.isclose(np.median(order_e_tol), expected_order_error_e_tol, atol=0.4):
        not_passed += [(4, f'Expected order wrt e_tol {expected_order_error_e_tol}, but got {order_e_tol}!')]
    if not np.isclose(np.median(order_estimate_e_tol), expected_order_estimate_e_tol, atol=0.4):
        not_passed += [
            (
                f'Expected order wrt e_tol {expected_order_estimate_e_tol} in the estimate, but got {order_estimate_e_tol}!'
            )
        ]

    return not_passed


@pytest.mark.base
@pytest.mark.parametrize('num_nodes', [2, 3, 4])
def test_AdaptivityInterpolationError(num_nodes):
    from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityInterpolationError
    import numpy as np

    e_tol_range = np.logspace(-1, -8, 2**3)
    not_passed = multiple_tests(
        AdaptivityInterpolationError, e_tol_range=e_tol_range, num_experiments=4, num_nodes=num_nodes
    )
    assert not_passed == [], not_passed


if __name__ == '__main__':
    test_AdaptivityInterpolationError(3)
