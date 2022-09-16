import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from pySDC.core.Hooks import hooks
from pySDC.helpers.stats_helper import get_sorted

from pySDC.projects.Resilience.vdp import run_vdp
from pySDC.projects.Resilience.piline import run_piline
from pySDC.projects.Resilience.advection import run_advection
from pySDC.playgrounds.Preconditioners.diagonal_precon_sweeper import DiagPrecon, DiagPreconIMEX
from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity, AdaptivityResidual

print_status = False


class log_cost(hooks):
    '''
    This class stores all relevant information and allows fault injection
    '''

    def post_step(self, step, level_number):

        super(log_cost, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        self.increment_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=0,
                             sweep=L.status.sweep, type='u_old', value=L.uold[-1])
        self.increment_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=0,
                             sweep=L.status.sweep, type='u', value=L.u[-1])
        self.increment_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=0,
                             sweep=L.status.sweep, type='e_em', value=L.status.error_embedded_estimate)
        self.increment_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=0,
                             sweep=L.status.sweep, type='k', value=step.status.iter)
        self.increment_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=0,
                             sweep=L.status.sweep, type='restarts', value=int(step.status.restart))


def single_run(x, params, convergence_controllers, *args):
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
        'convergence_controllers': convergence_controllers,
    }
    problem_params = params['problem_params']

    # setup the sweeper
    if None not in x:
        sweeper_params = {
            'num_nodes': len(x) - 1,
            'diagonal_elements': np.array(x),
        }

        custom_description['sweeper_params'] = sweeper_params
        custom_description['sweeper_class'] = params['sweeper']

    stats, controller, _ = params['prob'](custom_description=custom_description, hook_class=log_cost,
                                          custom_problem_params=problem_params)
    return stats, controller


def objective_function_diagonal_adaptivity_embedded2(x, *args):
    '''
    The same as objective_function_diagonal_residual_embedded, but with sum(x) = 1
    '''
    return objective_function_diagonal_adaptivity_embedded(np.append(x, - sum(x) + 1), *args)


def objective_function_diagonal_adaptivity_embedded(x, *args):
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

    # setup adaptivity and problem parameters
    convergence_controllers = {
        Adaptivity: {'e_tol': params['e_tol']},
    }

    stats, controller = single_run(x, params, convergence_controllers)

    # check how many iterations we needed
    k = sum([me[1] for me in get_sorted(stats, type='k')])

    # get error
    e = get_error(stats, controller)

    # return the score
    score = k
    if print_status:
        print(f's={score:7.0f} | e={e:.2e} | k: {k - params["k"]:5} | sum(x)={sum(x):.2f}', x)

    return score


def objective_function_diagonal_residual_embedded(x, *args):
    '''
    This function takes as input the diagonal preconditioner entries and runs a problem and then returns the number of
    iterations.

    The args should contain the problem parameters in position 0

    Args:
        x (numpy.ndarray): The entries of the preconditioner

    Returns:
        int: Number of iterations
    '''
    params = args[0]

    # setup adaptivity and problem parameters
    convergence_controllers = {
        Adaptivity: {'e_tol': params['e_tol']},
        AdaptivityResidual: {'e_tol': params['r_tol'], 'max_restarts': 99}
    }

    stats, controller = single_run(x, params, convergence_controllers)

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


def get_error(stats, controller):
    u_end = get_sorted(stats, type='u')[-1]
    exact = controller.MS[0].levels[0].prob.u_exact(t=u_end[0])
    return abs(u_end[1] - exact)


vdp_params = {
    'prob': run_vdp,
    'sweeper': DiagPrecon,
    'e_tol': 2e-5,
    'problem_params': None,
    'r_tol': 1.,
    'k': 1,
    'e': 1,
    'e_em': 1.,
}


args_piline = {
    'prob': run_piline,
    'sweeper': DiagPreconIMEX,
    'e_tol': 1e-7,
    'problem_params': None,
    'r_tol': np.inf,
    'k': 2461,
    'e': 4.14e-8,
    'e_em': 7.27e-8,
}


args_advection = {
    'prob': run_advection,
    'sweeper': DiagPrecon,
    'e_tol': 1e-9,
    'problem_params': {'freq': -1, 'sigma': 6e-2},
    'r_tol': 2e-11,
    'k': 475,
    'e': 5.98e-8,
    'e_em': 5.91e-10,
}


def optimize_with_sum(args, num_nodes):
    initial_guess = (np.arange(num_nodes - 1) + 1) / (num_nodes + 2)
    tol = 1e-16
    minimize(objective_function_diagonal_adaptivity_embedded2, initial_guess, args=args_advection, tol=tol,
             method='nelder-mead')


def optimize_without_sum(args, num_nodes):
    initial_guess = (np.arange(num_nodes) + 1) / (num_nodes + 1)
    tol = 1e-16
    minimize(objective_function_diagonal_adaptivity_embedded, initial_guess, args=args_advection, tol=tol,
             method='nelder-mead')


if __name__ == '__main__':
    print_status = True

    args = args_advection
    num_nodes = 3
    initial_guess = np.arange(num_nodes) + 1 / num_nodes
    tol = 1e-16
    # minimize(objective_function_diagonal_adaptivity_embedded, initial_guess, args=args_advection, tol=tol,
    #          method='nelder-mead')
    # objective_function_diagonal([None], *args)
    # plt.show()
    # objective_function_diagonal([0.19915688, 0.27594549, 0.27594545], *args)
    # plt.show()

    # optimize_with_sum(args, num_nodes)
    optimize_without_sum(args, num_nodes)
