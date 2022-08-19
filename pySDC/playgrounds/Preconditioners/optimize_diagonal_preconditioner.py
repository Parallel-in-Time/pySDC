import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from pySDC.core.Hooks import hooks
from pySDC.helpers.stats_helper import get_sorted

from pySDC.projects.Resilience.vdp import run_vdp
from pySDC.projects.Resilience.piline import run_piline
from pySDC.projects.Resilience.advection import run_advection
from pySDC.playgrounds.Preconditioners.diagonal_precon_sweeper import DiagPrecon, DiagPreconIMEX
from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity  # , AdaptivityResidual


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


def objective_function_diagonal(x, *args):
    '''
    This function takes as input the diagonal preconditioner entries and runs a problem and then returns the number of
    iterations.

    The args should contain the problem to run in position 0

    Args:
        x (numpy.ndarray): The entries of the preconditioner

    Returns:
        int: Number of iterations
    '''

    # setup adaptivity and problem parameters
    custom_description = {
        'convergence_controllers': {
            Adaptivity: {'e_tol': args[2]},
            # AdaptivityResidual: {'e_tol': args[4], 'max_restarts': 99}
        }
    }
    problem_params = args[3]

    # setup the sweeper
    if None not in x:
        sweeper_params = {
            'num_nodes': len(x) - 1,
            'diagonal_elements': np.array(x),
        }

        custom_description['sweeper_params'] = sweeper_params
        custom_description['sweeper_class'] = args[1]

    prob = args[0]
    stats, controller, _ = prob(custom_description=custom_description, hook_class=log_cost,
                                custom_problem_params=problem_params)

    # check how many iterations we needed
    k = sum([me[1] for me in get_sorted(stats, type='k')])

    # check if we solved the problem correctly
    u_end = get_sorted(stats, type='u')[-1]
    exact = controller.MS[0].levels[0].prob.u_exact(t=u_end[0])
    e = abs(exact - u_end[1])
    e_em = max([me[1] for me in get_sorted(stats, type='e_em', recomputed=False)])

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


if __name__ == '__main__':
    # parameters are: problem, sweeper, adaptivity tolerance, problem_params, adaptivity residual tolerance, k, e, e_em
    args_vdp = (run_vdp, DiagPrecon, 2e-5, None, 1., 1, 1, 1)
    args_piline = (run_piline, DiagPreconIMEX, 1e-7, None, np.inf, 2416, 4.14e-8, 7.27e-8)
    args_advection = (run_advection, DiagPrecon, 1e-9, {'freq': -1, 'sigma': 6e-2}, 2e-11, 475, 5.98e-8, 5.91e-10)

    args = args_piline
    num_nodes = 3
    # initial_guess = np.random.rand(num_nodes)
    initial_guess = np.ones(num_nodes) * 0.5
    # initial_guess = [0.73583259, 0.14629964, 0.44240191] #  320 iterations for vdp (local minimum)
    # initial_guess = [0.53941499, 0.47673377, 0.47673299] #  20 iterations for advection (local minimum)
    bounds = [(0, 1) for i in initial_guess]
    tol = 1e-16
    print(minimize(objective_function_diagonal, initial_guess, args=args, tol=tol, bounds=bounds, method='nelder-mead'))
    objective_function_diagonal([None], *args)
    # plt.show()
    # objective_function_diagonal([0.19915688, 0.27594549, 0.27594545], *args)
    # plt.show()
