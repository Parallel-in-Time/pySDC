import matplotlib

matplotlib.use('Agg')

import matplotlib.pylab as plt
import numpy as np

from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.core.Hooks import hooks
from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.problem_classes.Piline import piline
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI


class log_data(hooks):

    def post_step(self, step, level_number):

        super(log_data, self).post_step(step, level_number)

        # some abbreviations
        L = step.levels[level_number]

        L.sweep.compute_end_point()

        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='u', value=L.uend[0])
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='dt', value=L.dt)
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='e_embedded', value=L.status.e_embedded)
        self.add_to_stats(process=step.status.slot, time=L.time, level=L.level_index, iter=0,
                          sweep=L.status.sweep, type='e_extrapolated', value=L.status.e_extrapolated)


def single_run(var='dt', val=1e-1, k=5):
    """
    A simple test program to do PFASST runs for the heat equation
    """

    # initialize level parameters
    level_params = dict()
    level_params[var] = val

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = 'IE'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['QE'] = 'PIC'

    problem_params = {
        'Vs': 100.,
        'Rs': 1.,
        'C1': 1.,
        'Rpi': 0.2,
        'C2': 1.,
        'Lpi': 1.,
        'Rl': 5.,
    }

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = k

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = log_data
    controller_params['use_extrapolation_estimate'] = True
    controller_params['use_embedded_estimate'] = True

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = piline  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params

    # set time parameters
    t0 = 0.0
    if var == 'dt':
        Tend = 20 * val
    else:
        Tend = 2e1

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    results = {
        'e_embedded': sort_stats(filter_stats(stats, type='e_embedded'), sortby='time')[-1][1],
        'e_extrapolated': sort_stats(filter_stats(stats, type='e_extrapolated'), sortby='time')[-1][1],
        var: val,
    }

    return results


def mulitple_runs(ax, k=5):
    """
    A simple test program to compute the order of accuracy in time
    """

    # assemble list of dt
    dt_list = 0.05 * 10.**-(np.arange(20) / 10.)

    # perform first test
    res = single_run(var='dt', val=dt_list[0], k=k)
    for key in res.keys():
        res[key] = [res[key]]

    # perform rest of the tests
    for i in range(1, len(dt_list)):
        res_ = single_run(var='dt', val=dt_list[i], k=k)
        for key in res_.keys():
            res[key].append(res_[key])

    # visualize results
    plot(res, ax, k)


def plot(res, ax, k):
    keys = ['e_embedded', 'e_extrapolated']
    ls = ['-', '--']
    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][k - 3]

    for i in range(len(keys)):
        order = get_accuracy_order(res, key=keys[i], order=k)
        if i == 0:
            label = rf'$k={{{np.mean(order):.2f}}}$'
        else:
            label = None
        ax.loglog(res['dt'], res[keys[i]], color=color, ls=ls[i], label=label)

    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel(r'$\epsilon$')
    ax.legend(frameon=False)


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
        tmp = np.log(results[key][i] / results[key][i - 1]) / np.log(dt_list[i] / dt_list[i - 1])
        order.append(tmp)

    return order


def main():
    ks = [5, 4, 3, 2]
    fig, ax = plt.subplots(1, 1)
    for i in range(len(ks)):
        k = ks[i]
        mulitple_runs(k=k, ax=ax)
    ax.plot([None, None], color='black', label=r'$\epsilon_\mathrm{embedded}$', ls='-')
    ax.plot([None, None], color='black', label=r'$\epsilon_\mathrm{extrapolated}$', ls='--')
    ax.legend(frameon=False)
    fig.savefig('error_order.pdf', bbox_inches='tight')


if __name__ == "__main__":
    main()
