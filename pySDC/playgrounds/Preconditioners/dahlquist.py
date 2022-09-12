# script to run a simple advection problem
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.core.Hooks import hooks
from pySDC.helpers.stats_helper import get_sorted
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl


class log_data(hooks):

    def post_iteration(self, step, level_number):

        super(log_data, self).post_iteration(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        self.add_to_stats(process=step.status.slot, time=L.time + L.dt, level=L.level_index, iter=step.status.iter,
                          sweep=L.status.sweep, type='u', value=L.uend)
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='dt', value=L.dt)

    def pre_run(self, step, level_number):
        super(log_data, self).pre_run(step, level_number)
        L = step.levels[level_number]
        self.add_to_stats(process=0, time=0, level=0, iter=0, sweep=0, type='lambdas', value=L.prob.params.lambdas)


def run_dahlquist(custom_description=None, num_procs=1, Tend=1., hook_class=log_data, fault_stuff=None,
                  custom_controller_params=None, custom_problem_params=None, **kwargs):

    # initialize level parameters
    level_params = dict()
    level_params['dt'] = 1.

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'IE'

    # build lambdas
    if 'lim' not in kwargs:
        lim = ((-30, 30), (-50, 50))
    else:
        lim = kwargs['lim']
    if 'res' not in kwargs:
        res = (400, 400)
    else:
        res = kwargs['res']
    re = np.linspace(lim[0][0], lim[0][1], res[0])
    im = np.linspace(lim[1][0], lim[1][1], res[1])
    lambdas = np.array([[complex(re[i], im[j]) for i in range(len(re))] for j in range(len(im))]).\
        reshape((len(re) * len(im)))

    problem_params = {
        'lambdas': lambdas,
        'u0': 1.,
    }

    if custom_problem_params is not None:
        problem_params = {**problem_params, **custom_problem_params}

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 5

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = hook_class
    controller_params['mssdc_jac'] = False

    if custom_controller_params is not None:
        controller_params = {**controller_params, **custom_controller_params}

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = testequation0d  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = generic_implicit  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params

    if custom_description is not None:
        for k in custom_description.keys():
            if k == 'sweeper_class':
                description[k] = custom_description[k]
                continue
            description[k] = {**description.get(k, {}), **custom_description.get(k, {})}

    # set time parameters
    t0 = 0.0

    # instantiate controller
    controller = controller_nonMPI(num_procs=num_procs, controller_params=controller_params,
                                   description=description)

    # insert faults
    if fault_stuff is not None:
        raise NotImplementedError('No fault stuff here...')

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    return stats, controller, Tend


def plot_stability(stats, ax=None, iter=None):
    lambdas = get_sorted(stats, type='lambdas')[0][1]
    u = get_sorted(stats, type='u', sortby='iter')

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    iter = [1] if iter is None else iter
    colors = ['blue', 'red', 'violet', 'green']

    for i in iter:
        # isolate the solutions from the iteration you want
        U = np.reshape([me[1] for me in u if me[0] == i], (len(np.unique(lambdas.real)), len(np.unique(lambdas.imag))))

        # get a grid for plotting
        X, Y = np.meshgrid(np.unique(lambdas.real), np.unique(lambdas.imag))
        ax.contour(X, Y, U, levels=[1], colors=colors[i - 1])
        ax.plot([None], [None], color=colors[i - 1], label=f'k={i}')

    # decorate
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.legend(frameon=False)


def plot_contraction(stats, fig=None, ax=None, iter=None, plot_increase=False, cbar=True, **kwargs):
    lambdas = get_sorted(stats, type='lambdas')[0][1]
    u = get_sorted(stats, type='u', sortby='iter')
    u_exact = np.exp(lambdas)

    kwargs['cmap'] = kwargs.get('cmap', 'jet')

    # get error for each iteration
    iter = [1] if iter is None else iter
    us = np.reshape(np.append(np.ones_like(lambdas), [me[1] for me in u if me[0] in iter]),
                    (len(iter) + 1, len(lambdas)))
    e = abs(us - u_exact)

    # get contraction rates for each iteration
    rho = e[1:, :] / e[:-1, :]
    rho_avg = np.mean(rho, axis=0)
    rho_log = np.log(np.reshape(rho_avg, (len(np.unique(lambdas.real)), len(np.unique(lambdas.imag)))))

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    # get a grid for plotting
    X, Y = np.meshgrid(np.unique(lambdas.real), np.unique(lambdas.imag))
    if plot_increase:
        cs = ax.contourf(X, Y, rho_log, **kwargs)
        ax.contour(X, Y, rho_log, levels=[0.])
    else:
        cs = ax.contourf(X, Y, np.where(rho_log < 0, rho_log, None), levels=500, **kwargs)

    # decorate
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    if cbar:
        cb = fig.colorbar(mpl.cm.ScalarMappable(**kwargs))
        cb.set_label(r'$\rho$')


def compare_contraction():
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={'width_ratios': [0.8, 1]})
    precons = ['TRAP', 'TAYLOR']
    norm = Normalize(vmin=-7, vmax=0)
    cbar=True
    for i in range(len(precons)):
        custom_description = {'sweeper_params': {'QI': precons[i]}}
        stats, controller, Tend = run_dahlquist(custom_description=custom_description, res=(400, 400))
        plot_contraction(stats, fig=fig, ax=axs[i], cbar=cbar, norm=norm, cmap='jet')
        cbar=False
        axs[i].set_title(precons[i])
    fig.tight_layout()


if __name__ == '__main__':
    compare_contraction()
    plt.show()

    #custom_description = {'sweeper_params': {'QI': 'LU'}}
    #stats, controller, Tend = run_dahlquist(custom_description=custom_description, res=(500, 500))

    #fig, axs = plt.subplots(1, 2, figsize=(12, 5.5))
    #plot_contraction(stats, fig=fig, ax=axs[1])
    #plot_stability(stats, iter=[1, 2, 3], ax=axs[0])
    #axs[0].set_title('Stability')
    #axs[1].set_title('Contraction')
    #fig.tight_layout()
    #plt.show()
