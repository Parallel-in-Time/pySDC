import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np
from pathlib import Path

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.convergence_controller_classes.estimate_embedded_error import EstimateEmbeddedErrorNonMPI
from pySDC.implementations.convergence_controller_classes.estimate_extrapolation_error import (
    EstimateExtrapolationErrorNonMPI,
)
from pySDC.core.Hooks import hooks

import pySDC.helpers.plot_helper as plt_helper
from pySDC.projects.Resilience.piline import run_piline


class do_nothing(hooks):
    pass


class log_errors(hooks):
    """
    A hook that only logs errors, but includes a local error that is not estimated during runtime.
    What that means is problem specific. If an analytical solution is available, the local error is exact,
    otherwise it is estimated using a reference solution generated with scipy.
    """

    def post_step(self, step, level_number):

        super(log_errors, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='e_loc',
            value=abs(L.prob.u_exact(t=L.time + L.dt, u_init=L.u[0], t_init=L.time) - L.u[-1]),
        )


def setup_mpl(font_size=8):
    """
    Setup matplotlib to fit in with TeX scipt.

    Args:
        fontsize (int): Font size

    Returns:
        None
    """
    plt_helper.setup_mpl(reset=True)
    # Set up plotting parameters
    style_options = {
        "axes.labelsize": 12,  # LaTeX default is 10pt font.
        "legend.fontsize": 13,  # Make the legend/label fonts a little smaller
        "axes.xmargin": 0.03,
        "axes.ymargin": 0.03,
    }
    mpl.rcParams.update(style_options)


def get_results_from_stats(stats, var, val, hook_class=log_errors):
    """
    Extract results from the stats are used to compute the order.

    Args:
        stats (dict): The stats object from a pySDC run
        var (str): The variable to compute the order against
        val (float): The value of var corresponding to this run
        hook_class (pySDC.Hook): A hook such that we know what information is available

    Returns:
        dict: The information needed for the order plot
    """
    results = {
        'e_embedded': 0.0,
        'e_extrapolated': 0.0,
        'e': 0.0,
        var: val,
    }

    if hook_class == log_errors:
        e_extrapolated = np.array(get_sorted(stats, type='error_extrapolation_estimate'))[:, 1]
        e_embedded = np.array(get_sorted(stats, type='error_embedded_estimate'))[:, 1]
        e_loc = np.array(get_sorted(stats, type='e_loc'))[:, 1]

        if len(e_extrapolated[e_extrapolated != [None]]) > 0:
            results['e_extrapolated'] = e_extrapolated[e_extrapolated != [None]][-1]

        if len(e_loc[e_loc != [None]]) > 0:
            results['e'] = max([e_loc[e_loc != [None]][-1], np.finfo(float).eps])

        if len(e_embedded[e_embedded != [None]]) > 0:
            results['e_embedded'] = e_embedded[e_embedded != [None]][-1]

    return results


def multiple_runs(
    k=5,
    serial=True,
    Tend_fixed=None,
    custom_description=None,
    prob=run_piline,
    dt_list=None,
    hook_class=log_errors,
    custom_controller_params=None,
    var='dt',
    avoid_restarts=False,
):
    """
    A simple test program to compute the order of accuracy.

    Args:
        k (int): Number of SDC sweeps
        serial (bool): Whether to do regular SDC or Multi-step SDC with 5 processes
        Tend_fixed (float): The time you want to solve the equation to. If left at `None`, the local error will be
                            computed since a fixed number of steps will be performed.
        custom_description (dict): Custom parameters to pass to the problem
        prob (function): A function that can accept suitable arguments and run a problem (see the Resilience project)
        dt_list (list): A list of values to check the order with
        hook_class (pySDC.Hook): A hook for recording relevant information
        custom_controller_params (dict): Custom parameters to pass to the problem
        var (str): The variable to check the order against
        avoid_restarts (bool): Mode of running adaptivity if applicable

    Returns:
        dict: The errors for different values of var
    """

    # assemble list of dt
    if dt_list is not None:
        pass
    elif Tend_fixed:
        dt_list = 0.1 * 10.0 ** -(np.arange(5) / 2)
    else:
        dt_list = 0.01 * 10.0 ** -(np.arange(20) / 10.0)

    num_procs = 1 if serial else 5

    # perform rest of the tests
    for i in range(0, len(dt_list)):
        desc = {
            'step_params': {'maxiter': k},
            'convergence_controllers': {
                EstimateEmbeddedErrorNonMPI: {},
                EstimateExtrapolationErrorNonMPI: {'no_storage': not serial},
            },
        }

        # setup the variable we check the order against
        if var == 'dt':
            desc['level_params'] = {'dt': dt_list[i]}
        elif var == 'e_tol':
            from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

            desc['convergence_controllers'][Adaptivity] = {'e_tol': dt_list[i], 'avoid_restarts': avoid_restarts}

        if custom_description is not None:
            desc = {**desc, **custom_description}
        Tend = Tend_fixed if Tend_fixed else 30 * dt_list[i]
        stats, controller, _ = prob(
            custom_description=desc,
            num_procs=num_procs,
            Tend=Tend,
            hook_class=hook_class,
            custom_controller_params=custom_controller_params,
        )

        level = controller.MS[-1].levels[-1]
        e_glob = abs(level.prob.u_exact(t=level.time + level.dt) - level.u[-1])
        e_loc = abs(level.prob.u_exact(t=level.time + level.dt, u_init=level.u[0], t_init=level.time) - level.u[-1])

        res_ = get_results_from_stats(stats, var, dt_list[i], hook_class)
        res_['e_glob'] = e_glob
        res_['e_loc'] = e_loc

        if i == 0:
            res = res_.copy()
            for key in res.keys():
                res[key] = [res[key]]
        else:
            for key in res_.keys():
                res[key].append(res_[key])
    return res


def plot_order(res, ax, k):
    """
    Plot the order using results from `multiple_runs`.

    Args:
        res (dict): The results from `multiple_runs`
        ax: Somewhere to plot
        k (int): Number of iterations

    Returns:
        None
    """
    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][k - 2]

    key = 'e_loc'
    order = get_accuracy_order(res, key=key, thresh=1e-11)
    label = f'k={k}, p={np.mean(order):.2f}'
    ax.loglog(res['dt'], res[key], color=color, ls='-', label=label)
    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel(r'$\epsilon$')
    ax.legend(frameon=False, loc='lower right')


def plot(res, ax, k, var='dt'):
    """
    Plot the order of various errors using the results from `multiple_runs`.

    Args:
        results (dict): the dictionary containing the errors
        ax: Somewhere to plot
        k (int): Number of SDC sweeps
        var (str): The variable to compute the order against

    Returns:
        None
    """
    keys = ['e_embedded', 'e_extrapolated', 'e']
    ls = ['-', ':', '-.']
    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][k - 2]

    for i in range(len(keys)):
        if all([me == 0.0 for me in res[keys[i]]]):
            continue
        order = get_accuracy_order(res, key=keys[i], var=var)
        if keys[i] == 'e_embedded':
            label = rf'$k={{{np.mean(order):.2f}}}$'
            expect_order = k if var == 'dt' else 1.0
            assert np.isclose(
                np.mean(order), expect_order, atol=4e-1
            ), f'Expected embedded error estimate to have order {expect_order} \
but got {np.mean(order):.2f}'

        elif keys[i] == 'e_extrapolated':
            label = None
            expect_order = k + 1 if var == 'dt' else 1 + 1 / k
            assert np.isclose(
                np.mean(order), expect_order, rtol=3e-1
            ), f'Expected extrapolation error estimate to have order \
{expect_order} but got {np.mean(order):.2f}'
        else:
            label = None
        ax.loglog(res[var], res[keys[i]], color=color, ls=ls[i], label=label)

    if var == 'dt':
        ax.set_xlabel(r'$\Delta t$')
    elif var == 'e_tol':
        ax.set_xlabel(r'$\epsilon_\mathrm{TOL}$')
    else:
        ax.set_xlabel(var)
    ax.set_ylabel(r'$\epsilon$')
    ax.legend(frameon=False, loc='lower right')


def get_accuracy_order(results, key='e_embedded', thresh=1e-14, var='dt'):
    """
    Routine to compute the order of accuracy in time

    Args:
        results (dict): the dictionary containing the errors
        key (str): The key in the dictionary corresponding to a specific error
        thresh (float): A threshold below which values are not entered into the order computation
        var (str): The variable to compute the order against

    Returns:
        the list of orders
    """

    # retrieve the list of dt from results
    assert var in results, f'ERROR: expecting the list of {var} in the results dictionary'
    dt_list = sorted(results[var], reverse=True)

    order = []
    # loop over two consecutive errors/dt pairs
    for i in range(1, len(dt_list)):
        # compute order as log(prev_error/this_error)/log(this_dt/old_dt) <-- depends on the sorting of the list!
        try:
            tmp = np.log(results[key][i] / results[key][i - 1]) / np.log(dt_list[i] / dt_list[i - 1])
            if results[key][i] > thresh and results[key][i - 1] > thresh:
                order.append(tmp)
        except TypeError:
            print('Type Warning', results[key])

    return order


def plot_orders(
    ax,
    ks,
    serial,
    Tend_fixed=None,
    custom_description=None,
    prob=run_piline,
    dt_list=None,
    custom_controller_params=None,
):
    """
    Plot only the local error.

    Args:
        ax: Somewhere to plot
        ks (list): List of sweeps
        serial (bool): Whether to do regular SDC or Multi-step SDC with 5 processes
        Tend_fixed (float): The time you want to solve the equation to. If left at `None`, the local error will be
        custom_description (dict): Custom parameters to pass to the problem
        prob (function): A function that can accept suitable arguments and run a problem (see the Resilience project)
        dt_list (list): A list of values to check the order with
        custom_controller_params (dict): Custom parameters to pass to the problem

    Returns:
        None
    """
    for i in range(len(ks)):
        k = ks[i]
        res = multiple_runs(
            k=k,
            serial=serial,
            Tend_fixed=Tend_fixed,
            custom_description=custom_description,
            prob=prob,
            dt_list=dt_list,
            hook_class=do_nothing,
            custom_controller_params=custom_controller_params,
        )
        plot_order(res, ax, k)


def plot_all_errors(
    ax,
    ks,
    serial,
    Tend_fixed=None,
    custom_description=None,
    prob=run_piline,
    dt_list=None,
    custom_controller_params=None,
    var='dt',
    avoid_restarts=False,
):
    """
    Make tests for plotting the error and plot a bunch of error estimates

    Args:
        ax: Somewhere to plot
        ks (list): List of sweeps
        serial (bool): Whether to do regular SDC or Multi-step SDC with 5 processes
        Tend_fixed (float): The time you want to solve the equation to. If left at `None`, the local error will be
        custom_description (dict): Custom parameters to pass to the problem
        prob (function): A function that can accept suitable arguments and run a problem (see the Resilience project)
        dt_list (list): A list of values to check the order with
        custom_controller_params (dict): Custom parameters to pass to the problem
        var (str): The variable to compute the order against
        avoid_restarts (bool): Mode of running adaptivity if applicable

    Returns:
        None
    """
    for i in range(len(ks)):
        k = ks[i]
        res = multiple_runs(
            k=k,
            serial=serial,
            Tend_fixed=Tend_fixed,
            custom_description=custom_description,
            prob=prob,
            dt_list=dt_list,
            custom_controller_params=custom_controller_params,
            var=var,
            avoid_restarts=avoid_restarts,
        )

        # visualize results
        plot(res, ax, k, var=var)

    ax.plot([None, None], color='black', label=r'$\epsilon_\mathrm{embedded}$', ls='-')
    ax.plot([None, None], color='black', label=r'$\epsilon_\mathrm{extrapolated}$', ls=':')
    ax.plot([None, None], color='black', label=r'$e$', ls='-.')
    ax.legend(frameon=False, loc='lower right')


def check_order_with_adaptivity():
    """
    Test the order when running adaptivity.
    Since we replace the step size with the tolerance, we check the order against this.

    Irrespective of the number of sweeps we do, the embedded error estimate should scale linearly with the tolerance,
    since it is supposed to match it as closely as possible.

    The error estimate for the error of the last sweep, however will depend on the number of sweeps we do. The order
    we expect is 1 + 1/k.
    """
    setup_mpl()
    ks = [4, 3, 2]
    for serial in [True, False]:
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
        plot_all_errors(ax, ks, serial, Tend_fixed=5e-1, var='e_tol', dt_list=[1e-5, 1e-6, 1e-7], avoid_restarts=True)
        if serial:
            fig.savefig('data/error_estimate_order_adaptivity.png', dpi=300, bbox_inches='tight')
        else:
            fig.savefig('data/error_estimate_order_adaptivity_parallel.png', dpi=300, bbox_inches='tight')
        plt.close(fig)


def check_order_against_step_size():
    """
    Check the order versus the step size for different numbers of sweeps.
    """
    setup_mpl()
    ks = [4, 3, 2]
    for serial in [True, False]:
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))

        plot_all_errors(ax, ks, serial, Tend_fixed=1.0)

        if serial:
            fig.savefig('data/error_estimate_order.png', dpi=300, bbox_inches='tight')
        else:
            fig.savefig('data/error_estimate_order_parallel.png', dpi=300, bbox_inches='tight')
        plt.close(fig)


def main():
    """Run various tests"""
    check_order_with_adaptivity()
    check_order_against_step_size()


if __name__ == "__main__":
    main()
