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
    def post_step(self, step, level_number):

        super(log_errors, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='e_embedded',
            value=L.status.error_embedded_estimate,
        )
        self.add_to_stats(
            process=step.status.slot,
            time=L.time + L.dt,
            level=L.level_index,
            iter=0,
            sweep=L.status.sweep,
            type='e_extrapolated',
            value=L.status.error_extrapolation_estimate,
        )
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
    results = {
        'e_embedded': 0.,
        'e_extrapolated': 0.,
        'e': 0.,
        var: val,
    }

    if hook_class == log_errors:
        e_extrapolated = np.array(get_sorted(stats, type='e_extrapolated'))[:, 1]
        e_embedded = np.array(get_sorted(stats, type='e_embedded'))[:, 1]
        e_loc = np.array(get_sorted(stats, type='e_loc'))[:, 1]

        if len(e_extrapolated[e_extrapolated != [None]]) > 0:
            results['e_extrapolated'] = e_extrapolated[e_extrapolated != [None]][-1]

        if len(e_loc[e_loc != [None]]) > 0:
            results['e'] = max([e_loc[e_loc != [None]][-1], np.finfo(float).eps])

        if len(e_embedded[e_embedded != [None]]) > 0:
            results['e_embedded'] = e_embedded[e_embedded != [None]][-1]

    return results


def multiple_runs(k=5, serial=True, Tend_fixed=None, custom_description=None, prob=run_piline, dt_list=None,
                  hook_class=log_errors, custom_controller_params=None):
    """
    A simple test program to compute the order of accuracy in time
    """

    # assemble list of dt
    if dt_list is not None:
        pass
    elif Tend_fixed:
        dt_list = 0.1 * 10.**-(np.arange(5) / 2)
    else:
        dt_list = 0.01 * 10.0 ** -(np.arange(20) / 10.0)

    num_procs = 1 if serial else 5

    # perform rest of the tests
    for i in range(0, len(dt_list)):
        desc = {
            'level_params': {'dt': dt_list[i]},
            'step_params': {'maxiter': k},
            'convergence_controllers': {
                EstimateEmbeddedErrorNonMPI: {},
                EstimateExtrapolationErrorNonMPI: {'no_storage': not serial},
            },
        }
        if custom_description is not None:
            desc = {**desc, **custom_description}
        Tend = Tend_fixed if Tend_fixed else 30 * dt_list[i]
        stats, controller, _ = prob(custom_description=desc, num_procs=num_procs, Tend=Tend,
                                    hook_class=hook_class, custom_controller_params=custom_controller_params)

        level = controller.MS[-1].levels[-1]
        e_glob = abs(level.prob.u_exact(t=level.time + level.dt) - level.u[-1])
        e_loc = abs(level.prob.u_exact(t=level.time + level.dt, u_init=level.u[0], t_init=level.time) - level.u[-1])

        res_ = get_results_from_stats(stats, 'dt', dt_list[i], hook_class)
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
    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][k - 2]

    key = 'e_loc'
    order = get_accuracy_order(res, key=key, thresh=1e-11)
    label = f'k={k}, p={np.mean(order):.2f}'
    ax.loglog(res['dt'], res[key], color=color, ls='-', label=label)
    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel(r'$\epsilon$')
    ax.legend(frameon=False, loc='lower right')


def plot(res, ax, k):
    keys = ['e_embedded', 'e_extrapolated', 'e']
    ls = ['-', ':', '-.']
    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][k - 2]

    for i in range(len(keys)):
        if all([me == 0. for me in res[keys[i]]]):
            continue
        order = get_accuracy_order(res, key=keys[i])
        if keys[i] == 'e_embedded':
            label = rf'$k={{{np.mean(order):.2f}}}$'
            assert np.isclose(np.mean(order), k, atol=4e-1), f'Expected embedded error estimate to have order {k} \
but got {np.mean(order):.2f}'

        elif keys[i] == 'e_extrapolated':
            label = None
            assert np.isclose(np.mean(order), k + 1, rtol=3e-1), f'Expected extrapolation error estimate to have order \
{k+1} but got {np.mean(order):.2f}'
        else:
            label = None
        ax.loglog(res['dt'], res[keys[i]], color=color, ls=ls[i], label=label)

    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel(r'$\epsilon$')
    ax.legend(frameon=False, loc='lower right')


def get_accuracy_order(results, key='e_embedded', thresh=1e-14):
    """
    Routine to compute the order of accuracy in time

    Args:
        results (dict): the dictionary containing the errors
        key (str): The key in the dictionary correspdoning to a specific error

    Returns:
        the list of orders
    """

    # retrieve the list of dt from results
    assert 'dt' in results, 'ERROR: expecting the list of dt in the results dictionary'
    dt_list = sorted(results['dt'], reverse=True)

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


def plot_orders(ax, ks, serial, Tend_fixed=None, custom_description=None, prob=run_piline, dt_list=None,
                custom_controller_params=None):
    for i in range(len(ks)):
        k = ks[i]
        res = multiple_runs(k=k, serial=serial, Tend_fixed=Tend_fixed, custom_description=custom_description,
                            prob=prob, dt_list=dt_list, hook_class=do_nothing,
                            custom_controller_params=custom_controller_params)
        plot_order(res, ax, k)


def plot_all_errors(ax, ks, serial, Tend_fixed=None, custom_description=None, prob=run_piline, dt_list=None,
                    custom_controller_params=None):
    for i in range(len(ks)):
        k = ks[i]
        res = multiple_runs(k=k, serial=serial, Tend_fixed=Tend_fixed, custom_description=custom_description,
                            prob=prob, dt_list=dt_list, custom_controller_params=custom_controller_params)

        # visualize results
        plot(res, ax, k)

    ax.plot([None, None], color='black', label=r'$\epsilon_\mathrm{embedded}$', ls='-')
    ax.plot([None, None], color='black', label=r'$\epsilon_\mathrm{extrapolated}$', ls=':')
    ax.plot([None, None], color='black', label=r'$e$', ls='-.')
    ax.legend(frameon=False, loc='lower right')


def main():
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


if __name__ == "__main__":
    main()
