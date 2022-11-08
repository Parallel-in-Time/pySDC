# script to run a simple advection problem
from pySDC.implementations.problem_classes.TestEquation_0D import testequation0d
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.core.Hooks import hooks
from pySDC.helpers.stats_helper import get_sorted
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable


class log_data(hooks):
    def post_iteration(self, step, level_number):

        super(log_data, self).post_iteration(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=step.status.iter,
            sweep=L.status.sweep,
            type='u',
            value=L.uend,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='dt',
            value=L.dt,
        )

    def pre_run(self, step, level_number):
        super(log_data, self).pre_run(step, level_number)
        L = step.levels[level_number]
        self.add_to_stats(process=0, time=0, level=0, iter=0, sweep=0, type='lambdas', value=L.prob.params.lambdas)


def run_dahlquist(
    custom_description=None,
    num_procs=1,
    Tend=1.0,
    hook_class=log_data,
    fault_stuff=None,
    custom_controller_params=None,
    custom_problem_params=None,
    **kwargs,
):

    # initialize level parameters
    level_params = dict()
    level_params['dt'] = 1.0

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'RADAU-RIGHT'
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'IE'
    sweeper_params['initial_guess'] = 'random'

    # build lambdas
    re = np.linspace(-30, 30, 400)
    im = np.linspace(-50, 50, 400)
    lambdas = np.array([[complex(re[i], im[j]) for i in range(len(re))] for j in range(len(im))]).reshape(
        (len(re) * len(im))
    )

    problem_params = {
        'lambdas': lambdas,
        'u0': 1.0 + 0.0j,
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
    controller = controller_nonMPI(num_procs=num_procs, controller_params=controller_params, description=description)

    # insert faults
    if fault_stuff is not None:
        raise NotImplementedError('No fault stuff here...')

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
    return stats, controller, Tend


def plot_stability(stats, ax=None, iter=None, colors=None, crosshair=True, fill=False, **kwargs):
    lambdas = get_sorted(stats, type='lambdas')[0][1]
    u = get_sorted(stats, type='u', sortby='iter')

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    # decorate
    if crosshair:
        ax.axhline(0, color='black', alpha=1.0)
        ax.axvline(0, color='black', alpha=1.0)

    iter = [1] if iter is None else iter
    colors = ['blue', 'red', 'violet', 'green'] if colors is None else colors

    for i in iter:
        # isolate the solutions from the iteration you want
        U = np.reshape([me[1] for me in u if me[0] == i], (len(np.unique(lambdas.real)), len(np.unique(lambdas.imag))))

        # get a grid for plotting
        X, Y = np.meshgrid(np.unique(lambdas.real), np.unique(lambdas.imag))
        if fill:
            ax.contourf(X, Y, abs(U), levels=[-np.inf, 1 - np.finfo(float).eps], colors=colors[i - 1], alpha=0.5)
        ax.contour(X, Y, abs(U), levels=[1], colors=colors[i - 1])
        ax.plot([None], [None], color=colors[i - 1], label=f'k={i}')

    ax.legend(frameon=False)


def plot_contraction(stats, fig=None, ax=None, iter=None, plot_increase=False, cbar=True, **kwargs):
    lambdas = get_sorted(stats, type='lambdas')[0][1]
    real = np.unique(lambdas.real)
    imag = np.unique(lambdas.imag)

    u = get_sorted(stats, type='u', sortby='iter')
    t = get_sorted(stats, type='u', sortby='time')[0][0]
    u_exact = np.exp(lambdas * t)

    kwargs['cmap'] = kwargs.get('cmap', 'seismic' if plot_increase else 'jet')

    # decide which iterations to look at
    iter = [0, 1] if iter is None else iter
    assert len(iter) > 1, 'Need to compute the contraction factor across multiple iterations!'

    # get solution for the specified iterations
    us = [me[1] for me in u if me[0] in iter]
    if 0 in iter:  # ic's are not stored in stats, so we have to add them manually
        us = np.append([np.ones_like(lambdas)], us, axis=0)

    # get error for each iteration
    e = abs(us - u_exact)
    e[e == 0] = np.finfo(float).eps

    # get contraction rates for each iteration
    rho = e[1:, :] / e[:-1, :]
    rho_avg = np.mean(rho, axis=0)
    rho_log = np.log(np.reshape(rho_avg, (len(imag), len(real))))

    # get spaceally averaged contraction factor
    # rho_avg_space = np.mean(rho, axis=1)
    # e_tot = np.sum(e, axis=1)
    # rho_tot = e_tot[1:] / e_tot[:-1]

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    # get a grid for plotting
    X, Y = np.meshgrid(real, imag)
    if plot_increase:
        ax.contour(X, Y, rho_log, levels=[0.0])
        lim = max(np.abs([rho_log.min(), rho_log.max()]))
        kwargs['vmin'] = kwargs.get('vmin', -lim)
        kwargs['vmax'] = kwargs.get('vmax', lim)
        cs = ax.contourf(X, Y, rho_log, **kwargs)
    else:
        cs = ax.contourf(X, Y, np.where(rho_log <= 0, rho_log, None), levels=500, **kwargs)

    # decorate
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')

    # fix pdf plotting
    ax.set_rasterized(True)

    if cbar:
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes('right', 0.2, pad=0.1)
        cb = fig.colorbar(cs, cbar_ax)
        cb.set_label(r'$\rho$')
        cbar_ax.set_rasterized(True)
    return cs


def plot_increment(stats, fig=None, ax=None, iter=None, cbar=True, **kwargs):
    lambdas = get_sorted(stats, type='lambdas')[0][1]
    u = get_sorted(stats, type='u', sortby='iter')

    kwargs['cmap'] = kwargs.get('cmap', 'jet')

    # decide which iterations to look at
    iter = [0, 1] if iter is None else iter
    assert len(iter) > 1, 'Need to compute the increment accross multiple iterations!'

    # get solution for the specified iterations
    u_iter = [me[1] for me in u if me[0] in iter]
    if 0 in iter:  # ic's are not stored in stats, so we have to add them manually
        u_iter = np.append(np.ones_like(lambdas), u_iter)
    us = np.reshape(u_iter, (len(iter), len(lambdas)))

    # get contraction rates for each iteration
    rho = abs(us[1:, :] / us[:-1, :])
    rho_avg = np.mean(rho, axis=0)
    rho_log = np.log(np.reshape(rho_avg, (len(np.unique(lambdas.real)), len(np.unique(lambdas.imag)))))

    if ax is None:
        fig, ax = plt.subplots(1, 1)

    # get a grid for plotting
    X, Y = np.meshgrid(np.unique(lambdas.real), np.unique(lambdas.imag))
    cs = ax.contourf(X, Y, rho_log, levels=500, **kwargs)

    # outline the region where the increment is 0
    ax.contour(X, Y, rho_log, levels=[-15], colors=['red'])

    # decorate
    ax.axhline(0, color='black')
    ax.axvline(0, color='black')

    # fix pdf plotting
    ax.set_rasterized(True)

    if cbar:
        divider = make_axes_locatable(ax)
        cbar_ax = divider.append_axes('right', 0.2, pad=0.1)
        cb = fig.colorbar(cs, cbar_ax)
        cb.set_label('increment')
        cbar_ax.set_rasterized(True)


def compare_contraction():
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={'width_ratios': [0.8, 1]})
    precons = ['TRAP', 'IE']
    norm = Normalize(vmin=-7, vmax=0)
    cbar = True
    for i in range(len(precons)):
        custom_description = {'sweeper_params': {'QI': precons[i]}}
        stats, controller, Tend = run_dahlquist(custom_description=custom_description, res=(400, 400))
        plot_contraction(stats, fig=fig, ax=axs[i], cbar=cbar, norm=norm, cmap='jet')
        cbar = False
        axs[i].set_title(precons[i])
    fig.tight_layout()


if __name__ == '__main__':
    custom_description = None
    stats, controller, Tend = run_dahlquist(custom_description=custom_description)
    plot_stability(stats, iter=[1, 2, 3])
    plot_contraction(stats, iter=[0, 4])
    plt.show()
