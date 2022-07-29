import matplotlib as mpl
import matplotlib.pylab as plt
import numpy as np

from pySDC.helpers.stats_helper import get_sorted
from pySDC.implementations.convergence_controller_classes.estimate_embedded_error import EstimateEmbeddedErrorNonMPI
from pySDC.implementations.convergence_controller_classes.estimate_extrapolation_error import (
    EstimateExtrapolationErrorNonMPI,
)
from pySDC.core.Hooks import hooks

import pySDC.helpers.plot_helper as plt_helper
from pySDC.projects.Resilience.piline import run_piline


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
            type='e_glob',
            value=abs(L.prob.u_exact(t=L.time + L.dt) - L.u[-1]),
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


def get_results_from_stats(stats, var, val):
    e_extrapolated = np.array(get_sorted(stats, type='e_extrapolated'))[:, 1]

    e_glob = np.array(get_sorted(stats, type='e_glob'))[:, 1]

    results = {
        'e_embedded': get_sorted(stats, type='e_embedded')[-1][1],
        'e_extrapolated': e_extrapolated[e_extrapolated != [None]][-1],
        'e': max([abs(e_glob[-1] - e_glob[-2]), np.finfo(float).eps]),
        var: val,
    }

    return results


def multiple_runs(ax, k=5, serial=True):
    """
    A simple test program to compute the order of accuracy in time
    """

    # assemble list of dt
    dt_list = 0.01 * 10.0 ** -(np.arange(20) / 10.0)

    num_procs = 1 if serial else 30

    # perform first test
    desc = {
        'level_params': {'dt': dt_list[0]},
        'step_params': {'maxiter': k},
        'convergence_controllers': {
            EstimateEmbeddedErrorNonMPI: {},
            EstimateExtrapolationErrorNonMPI: {'no_storage': not serial},
        },
    }
    res = get_results_from_stats(run_piline(desc, num_procs, 30 * dt_list[0], log_errors), 'dt', dt_list[0])
    for key in res.keys():
        res[key] = [res[key]]

    # perform rest of the tests
    for i in range(1, len(dt_list)):
        desc = {
            'level_params': {'dt': dt_list[i]},
            'step_params': {'maxiter': k},
            'convergence_controllers': {
                EstimateEmbeddedErrorNonMPI: {},
                EstimateExtrapolationErrorNonMPI: {'no_storage': not serial},
            },
        }
        res_ = get_results_from_stats(run_piline(desc, num_procs, 30 * dt_list[i], log_errors), 'dt', dt_list[i])
        for key in res_.keys():
            res[key].append(res_[key])

    # visualize results
    plot(res, ax, k)


def plot(res, ax, k):
    keys = ['e_embedded', 'e_extrapolated', 'e']
    ls = ['-', ':', '-.']
    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][k - 2]

    for i in range(len(keys)):
        order = get_accuracy_order(res, key=keys[i], order=k)
        if keys[i] == 'e_embedded':
            label = rf'$k={{{np.mean(order):.2f}}}$'
            assert np.isclose(
                np.mean(order), k, atol=3e-1
            ), f'Expected embedded error estimate to have order {k} \
but got {np.mean(order):.2f}'

        elif keys[i] == 'e_extrapolated':
            label = None
            assert np.isclose(
                np.mean(order), k + 1, rtol=3e-1
            ), f'Expected extrapolation error estimate to have order \
{k+1} but got {np.mean(order):.2f}'
        ax.loglog(res['dt'], res[keys[i]], color=color, ls=ls[i], label=label)

    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel(r'$\epsilon$')
    ax.legend(frameon=False, loc='lower right')


def get_accuracy_order(results, key='e_embedded', order=5):
    """
    Routine to compute the order of accuracy in time

    Args:
        results: the dictionary containing the errors

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
            if results[key][i] > 1e-14 and results[key][i - 1] > 1e-14:
                order.append(tmp)
        except TypeError:
            print('Type Warning', results[key])

    return order


def plot_all_errors(ax, ks, serial):
    for i in range(len(ks)):
        k = ks[i]
        multiple_runs(k=k, ax=ax, serial=serial)
    ax.plot([None, None], color='black', label=r'$\epsilon_\mathrm{embedded}$', ls='-')
    ax.plot([None, None], color='black', label=r'$\epsilon_\mathrm{extrapolated}$', ls=':')
    ax.plot([None, None], color='black', label=r'$e$', ls='-.')
    ax.legend(frameon=False, loc='lower right')


def main():
    setup_mpl()
    ks = [4, 3, 2]
    for serial in [True, False]:
        fig, ax = plt.subplots(1, 1, figsize=(3.5, 3))
        plot_all_errors(ax, ks, serial)
        if serial:
            fig.savefig('data/error_estimate_order.png', dpi=300, bbox_inches='tight')
        else:
            fig.savefig('data/error_estimate_order_parallel.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    main()
