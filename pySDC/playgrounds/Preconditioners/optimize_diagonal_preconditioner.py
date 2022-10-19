import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from pySDC.core.Hooks import hooks
from pySDC.helpers.stats_helper import get_sorted

from pySDC.playgrounds.Preconditioners.configs import (
    get_params,
    store_precon,
    store_serial_precon,
    get_collocation_nodes,
)

print_status = False


class log_cost(hooks):
    '''
    This class stores all relevant information for the cost function
    '''

    def post_step(self, step, level_number):

        super(log_cost, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        self.increment_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='u_old',
            value=L.uold[-1],
        )
        self.increment_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='u',
            value=L.u[-1],
        )
        self.increment_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='e_em',
            value=L.status.error_embedded_estimate,
        )
        self.increment_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='k',
            value=step.status.iter,
        )
        self.increment_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='restarts',
            value=int(step.status.restart),
        )


def prepare_sweeper(x, params, use_first_row=False, normalize=False, **kwargs):
    """
    Prepare the sweeper with diagonal elements before running the problem

    Args:
        x (numpy.ndarray): The entries of the preconditioner
        params (dict): Parameters for setting up the run
        use_first_row (bool): Use the first row of the preconditioner or not
        normalize (bool) Normalize the quadrature weights or not

    Returns
        dict: Sweeper parameters
    """
    if use_first_row:
        diags = np.array(x[0 : len(x) // 2])
        first_row = np.array(x[len(x) // 2 : :])
        num_nodes = len(x) // 2 - 1
    else:
        diags = np.array(x)
        first_row = np.zeros_like(diags)
        num_nodes = len(x) - 1

    if normalize:
        raise NotImplementedError

    # setup the sweeper
    if None not in x:
        sweeper_params = {
            'num_nodes': num_nodes,
            'diagonal_elements': diags,
            'first_row': first_row,
            'QI': params.get('QI', 'LU'),
            'quad_type': params.get('quad_type', 'RADAU-RIGHT'),
        }
    else:
        sweeper_params = {}

    return sweeper_params, params['sweeper']


def single_run(x, params, *args, **kwargs):
    '''
    This function takes as input the diagonal preconditioner entries and runs a problem.
    The args should contain the problem to run in position 0

    Args:
        x (numpy.ndarray): The entries of the preconditioner
        params (dict): Parameters for setting up the run
        convergence_controllers (dict): Convergence controllers to use

    Returns:
        dict: Stats of the run
        pySDC.controller: The controller used in the run
    '''

    # setup adaptivity and problem parameters
    custom_description = {
        'convergence_controllers': params.get('convergence_controllers', {}),
    }

    problem_params = params['problem_params']

    controller_params = params.get('controller_params', {})

    sweeper_params, sweeper = prepare_sweeper(x, params, **kwargs)
    custom_description['sweeper_params'] = sweeper_params
    custom_description['sweeper_class'] = sweeper

    stats, controller, _ = params['prob'](
        custom_description=custom_description,
        hook_class=log_cost,
        custom_problem_params=problem_params,
        custom_controller_params=controller_params,
    )
    return stats, controller


def objective_function_k_only(x, *args):
    '''
    This function takes as input the diagonal preconditioner entries and runs a problem and then returns the number of
    iterations.

    The args should contain the params for a problem in position 0

    Args:
        x (numpy.ndarray): The entries of the preconditioner

    Returns:
        int: Number of iterations
    '''
    params = args[0]
    kwargs = args[1]

    stats, controller = single_run(x, params, *args, **kwargs)

    # check how many iterations we needed
    k = sum([me[1] for me in get_sorted(stats, type='k')])

    # get error
    e = get_error(stats, controller)

    # return the score
    score = k
    if print_status:
        print(f's={score:7.0f} | e={e:.2e} | k: {k - params["k"]:5} | sum(x)={sum(x):.2f}', x)

    return score


def get_error(stats, controller):
    """
    Get the error at the end of a pySDC run

    Args:
        stats (pySDC.stats): Stats object generated by a pySDC run
        controller (pySDC.controller): Controller used for the run

    Returns:
        float: Error at the end of the run
    """
    u_end = get_sorted(stats, type='u')[-1]
    exact = controller.MS[0].levels[0].prob.u_exact(t=u_end[0])
    return abs(u_end[1] - exact)


def optimize(params, initial_guess, num_nodes, objective_function, tol=1e-16, **kwargs):
    """
    Run a single optimization run and store the result

    Args:
        params (dict): Parameters for running the problem
        initial_guess (numpy.ndarray): Initial guess to start the minimization problem
        num_nodes (int): Number of collocation nodes
        objective_function (function): Objective function for the minimizaiton alogrithm

    Returns:
        None
    """
    opt = minimize(objective_function, initial_guess, args=(params, kwargs), tol=tol, method='nelder-mead')
    store_precon(params, opt.x, initial_guess, **kwargs)


def objective_function_diagonal_adaptivity_embedded_normalized(x, *args):
    '''
    The same as objective_function_diagonal_residual_embedded, but with sum(x) = 1
    '''
    return objective_function_k_only(np.append(x, -sum(x) + 1), *args)


def objective_function_k_and_e(x, *args):
    '''
    This function takes as input the diagonal preconditioner entries and runs a problem and then returns the number of
    iterations.

    The args should contain the problem parameters in position 0

    Args:
        x (numpy.ndarray): The entries of the preconditioner

    Returns:
        int: Number of iterations
    '''
    raise NotImplementedError
    params = args[0]

    stats, controller = single_run(x, params)

    # check how many iterations we needed
    k = sum([me[1] for me in get_sorted(stats, type='k')])

    # check if we solved the problem correctly
    u_end = get_sorted(stats, type='u')[-1]
    exact = controller.MS[0].levels[0].prob.u_exact(t=u_end[0])
    e = abs(exact - u_end[1])
    e_em = max([me[1] for me in get_sorted(stats, type='e_em', recomputed=False)])
    raise NotImplementedError('Please fix the next couple of lines')

    # return the score
    score = k * e / args[6]
    print(f's={score:7.0f} | k: {k - args[5]:5} | e: {e / args[6]:.2e} | e_em: {e_em / args[7]:.2e}', x)
    # print(x, k, f'e={e:.2e}', f'e_em={e_em:.2e}')

    return score


def plot_errors(stats, u_end, exact):
    plt.plot(np.arange(len(u_end[1])), u_end[1])
    u_old = get_sorted(stats, type='u_old')[-1]
    plt.plot(np.arange(len(exact)), exact, ls='--')
    error = np.abs(exact - u_end[1])
    plt.plot(np.arange(len(exact)), error, label=f'e_max={error.max():.2e}')
    plt.plot(np.arange(len(u_old[1])), np.abs(u_old[1] - u_end[1]), label='e_em')
    # plt.yscale('log')
    plt.legend(frameon=False)
    plt.pause(1e-9)
    plt.cla()


def optimize_with_sum(params, num_nodes):
    initial_guess = (np.arange(num_nodes - 1) + 1) / (num_nodes + 2)
    tol = 1e-16
    minimize(
        objective_function_diagonal_adaptivity_embedded_normalized,
        initial_guess,
        args=params,
        tol=tol,
        method='nelder-mead',
    )


def optimize_without_sum(params, num_nodes, **kwargs):
    initial_guess = np.array(get_collocation_nodes(params, num_nodes)) / 2.0
    optimize(params, initial_guess, num_nodes, objective_function_k_only, **kwargs)


def optimize_with_first_row(params, num_nodes, **kwargs):
    i0 = np.array(get_collocation_nodes(params, num_nodes)) / 2.0
    initial_guess = np.append(i0, i0)
    kwargs['use_first_row'] = True
    optimize(params, initial_guess, num_nodes, objective_function_k_only, **kwargs)


if __name__ == '__main__':
    print_status = True

    kwargs = {'adaptivity': True}

    params = get_params('advection', **kwargs)
    num_nodes = 3

    optimize_without_sum(params, num_nodes, **kwargs)
    optimize_with_first_row(params, num_nodes, **kwargs)
    store_serial_precon('advection', num_nodes, LU=True, **kwargs)
    store_serial_precon('advection', num_nodes, IE=True, **kwargs)
