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

from pySDC.implementations.hooks.log_solution import LogSolutionAfterIteration
from pySDC.implementations.hooks.log_step_size import LogStepSize


class LogLambdas(hooks):
    """
    Store the lambda values at the beginning of the run
    """

    def pre_run(self, step, level_number):
        super().pre_run(step, level_number)
        L = step.levels[level_number]
        self.add_to_stats(process=0, time=0, level=0, iter=0, sweep=0, type='lambdas', value=L.prob.lambdas)


hooks = [LogLambdas, LogSolutionAfterIteration, LogStepSize]


def run_dahlquist(
    custom_description=None,
    num_procs=1,
    Tend=1.0,
    hook_class=hooks,
    fault_stuff=None,
    custom_controller_params=None,
    custom_problem_params=None,
    **kwargs,
):
    """
    Run a Dahlquist problem with default parameters.

    Args:
        custom_description (dict): Overwrite presets
        num_procs (int): Number of steps for MSSDC
        Tend (float): Time to integrate to
        hook_class (pySDC.Hook): A hook to store data
        fault_stuff (dict): A dictionary with information on how to add faults
        custom_controller_params (dict): Overwrite presets
        custom_problem_params (dict): Overwrite presets

    Returns:
        dict: The stats object
        controller: The controller
        Tend: The time that was supposed to be integrated to
    """

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
    """
    Plot the domain of stability by checking if the solution grows.

    Args:
        stats (pySDC.stats): The stats object of the run
        ax: Somewhere to plot
        iter (list): Check the stability for different numbers of iterations
        colors (list): Colors for the different iterations
        crosshair (bool): Whether to highlight the axes
        fill (bool): Fill the contours or not

    Returns:
        bool: If the method is A-stable or not
    """
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

    # check if the method is A-stable
    unstable = abs(U) > 1.0
    Astable = not any(X[unstable] < 0)

    ax.legend(frameon=False)

    return Astable


def plot_contraction(stats, fig=None, ax=None, iter=None, plot_increase=False, cbar=True, **kwargs):
    """
    Plot the contraction of the error.

    Args:
        stats (pySDC.stats): The stats object of the run
        fig: Figure of the plot, needed for a colorbar
        ax: Somewhere to plot
        iter (list): Plot the contraction for different numbers of iterations
        plot_increase (bool): Whether to also include increasing errors.
        cbar (bool): Plot a color bar or not

    Returns:
        The plot
    """
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
    """
    Plot the increment between iterations.

    Args:
        stats (pySDC.stats): The stats object of the run
        fig: Figure of the plot, needed for a colorbar
        ax: Somewhere to plot
        iter (list): Plot the contraction for different numbers of iterations
        cbar (bool): Plot a color bar or not

    Returns:
        None
    """
    lambdas = get_sorted(stats, type='lambdas')[0][1]
    u = get_sorted(stats, type='u', sortby='iter')

    kwargs['cmap'] = kwargs.get('cmap', 'jet')

    # decide which iterations to look at
    iter = [0, 1] if iter is None else iter
    assert len(iter) > 1, 'Need to compute the increment across multiple iterations!'

    # get solution for the specified iterations
    u_iter = [me[1] for me in u if me[0] in iter]
    if 0 in iter:  # ics are not stored in stats, so we have to add them manually
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
    """
    Make a plot comparing contraction factors between trapezoidal rule and implicit Euler.
    """
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
