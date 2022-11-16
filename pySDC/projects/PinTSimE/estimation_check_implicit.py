import numpy as np
import dill
from pathlib import Path

from pySDC.helpers.stats_helper import get_sorted
from pySDC.core.Collocation import CollBase as Collocation
from pySDC.implementations.problem_classes.Battery import battery_implicit
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.PinTSimE.piline_model import setup_mpl
from pySDC.projects.PinTSimE.battery_model import log_data, proof_assertions_description
import pySDC.helpers.plot_helper as plt_helper

from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator


def run(dt, use_switch_estimator=True, V_ref=1.0):
    """
    A simple test program to do SDC/PFASST runs for the battery drain model
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-8
    level_params['dt'] = dt

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'LOBATTO'
    sweeper_params['num_nodes'] = 5
    sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['initial_guess'] = 'zero'

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_maxiter'] = 200
    problem_params['newton_tol'] = 1e-08
    problem_params['Vs'] = 5.0
    problem_params['Rs'] = 0.5
    problem_params['C'] = 1.0
    problem_params['R'] = 1.0
    problem_params['L'] = 1.0
    problem_params['alpha'] = 1.2
    problem_params['V_ref'] = V_ref
    problem_params['set_switch'] = np.array([False], dtype=bool)
    problem_params['t_switch'] = np.zeros(1)

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 10

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = log_data

    # convergence controllers
    switch_estimator_params = {}
    convergence_controllers = {SwitchEstimator: switch_estimator_params}

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = battery_implicit  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = generic_implicit  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params

    if use_switch_estimator:
        description['convergence_controllers'] = convergence_controllers

    proof_assertions_description(description, problem_params)

    # set time parameters
    t0 = 0.0
    Tend = 2.0

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    Path("data").mkdir(parents=True, exist_ok=True)
    fname = 'data/battery_implicit.dat'
    f = open(fname, 'wb')
    dill.dump(stats, f)
    f.close()

    # filter statistics by number of iterations
    iter_counts = get_sorted(stats, type='niter', sortby='time')

    # compute and print statistics
    f = open('data/battery_implicit_out.txt', 'w')
    niters = np.array([item[1] for item in iter_counts])

    assert np.mean(niters) <= 11, "Mean number of iterations is too high, got %s" % np.mean(niters)
    f.close()

    return description, stats


def check(cwd='./'):
    """
    Routine to check the differences between using a switch estimator or not
    """

    V_ref = 1.0
    dt_list = [4e-1, 4e-2, 4e-3]
    use_switch_estimator = [True, False]
    restarts = []
    for dt_item in dt_list:
        for item in use_switch_estimator:
            description, stats = run(dt=dt_item, use_switch_estimator=item, V_ref=V_ref)

            fname = 'data/battery_implicit_dt{}_USE{}.dat'.format(dt_item, item)
            f = open(fname, 'wb')
            dill.dump(stats, f)
            f.close()

            if item:
                restarts_sorted = np.array(get_sorted(stats, type='restart', recomputed=False))[:, 1]
                restarts.append(np.sum(restarts_sorted))
                print("Restarts for dt: ", dt_item, " -- ", np.sum(restarts_sorted))

    assert len(dt_list) > 1, 'ERROR: dt_list have to be contained more than one element due to the subplots'

    differences_around_switch(dt_list, restarts, V_ref)

    differences_over_time(dt_list, V_ref)

    iterations_over_time(dt_list, description['step_params']['maxiter'])


def differences_around_switch(dt_list, restarts, V_ref, cwd='./'):
    """
    Routine to plot the differences before, at, and after the switch. Produces the diffs_estimation_generic_implicit.png file
    """

    diffs_true = []
    diffs_false_before = []
    diffs_false_after = []

    for dt_item in dt_list:
        f1 = open(cwd + 'data/battery_implicit_dt{}_USETrue.dat'.format(dt_item), 'rb')
        stats_true = dill.load(f1)
        f1.close()

        f2 = open(cwd + 'data/battery_implicit_dt{}_USEFalse.dat'.format(dt_item), 'rb')
        stats_false = dill.load(f2)
        f2.close()

        val_switch = get_sorted(stats_true, type='switch1', sortby='time')
        t_switch = [v[0] for v in val_switch]
        t_switch = t_switch[0]  # battery has only one single switch

        vC_true = get_sorted(stats_true, type='voltage C', recomputed=False, sortby='time')
        vC_false = get_sorted(stats_false, type='voltage C', sortby='time')

        diff_true, diff_false = [v[1] - V_ref for v in vC_true], [v[1] - V_ref for v in vC_false]
        times_true, times_false = [v[0] for v in vC_true], [v[0] for v in vC_false]

        for m in range(len(times_true)):
            if np.round(times_true[m], 15) == np.round(t_switch, 15):
                diffs_true.append(diff_true[m])

        for m in range(1, len(times_false)):
            if times_false[m - 1] < t_switch < times_false[m]:
                diffs_false_before.append(diff_false[m - 1])
                diffs_false_after.append(diff_false[m])

    setup_mpl()
    fig_around, ax_around = plt_helper.plt.subplots(1, 1, figsize=(3, 3))
    ax_around.set_title("Difference $v_{C}-V_{ref}$")
    pos1 = ax_around.plot(dt_list, diffs_false_before, 'rs-', label='SE=False - before switch')
    pos2 = ax_around.plot(dt_list, diffs_false_after, 'bd--', label='SE=False - after switch')
    pos3 = ax_around.plot(dt_list, diffs_true, 'kd--', label='SE=True')
    # ax.legend(frameon=False, fontsize=8, loc='center right')
    ax_around.set_xticks(dt_list)
    ax_around.set_xticklabels(dt_list)
    ax_around.set_xscale('log', base=10)
    ax_around.set_yscale('symlog', linthresh=1e-8)
    ax_around.set_ylim(-1, 1)
    ax_around.set_xlabel(r'$\Delta t$', fontsize=6)

    restart_ax = ax_around.twinx()
    restarts_plt = restart_ax.plot(dt_list, restarts, 'cs--', label='Restarts')
    restart_ax.set_label('Restarts')

    lines = pos1 + pos2 + pos3 + restarts_plt
    labels = [l.get_label() for l in lines]
    ax_around.legend(lines, labels, frameon=False, fontsize=8, loc='center right')
    fig_around.savefig('data/diffs_estimation_generic_implicit.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig_around)


def differences_over_time(dt_list, V_ref, cwd='./'):
    """
    Routine to plot the differences in time using the switch estimator or not. Produces the difference_estimation_generic_implicit.png file
    """

    setup_mpl()
    fig_diffs, ax_diffs = plt_helper.plt.subplots(
        1, len(dt_list), figsize=(2 * len(dt_list), 2), sharex='col', sharey='row'
    )
    count_ax = 0
    for dt_item in dt_list:
        f1 = open(cwd + 'data/battery_implicit_dt{}_USETrue.dat'.format(dt_item), 'rb')
        stats_true = dill.load(f1)
        f1.close()

        f2 = open(cwd + 'data/battery_implicit_dt{}_USEFalse.dat'.format(dt_item), 'rb')
        stats_false = dill.load(f2)
        f2.close()

        val_switch = get_sorted(stats_true, type='switch1', sortby='time')
        t_switch = [v[0] for v in val_switch]
        t_switch = t_switch[0]  # battery has only one single switch

        vC_true = get_sorted(stats_true, type='voltage C', recomputed=False, sortby='time')
        vC_false = get_sorted(stats_false, type='voltage C', sortby='time')

        diff_true, diff_false = [v[1] - V_ref for v in vC_true], [v[1] - V_ref for v in vC_false]
        times_true, times_false = [v[0] for v in vC_true], [v[0] for v in vC_false]

        ax_diffs[count_ax].set_title('dt={}'.format(dt_item))
        ax_diffs[count_ax].plot(times_true, diff_true, label='SE=True', color='#ff7f0e')
        ax_diffs[count_ax].plot(times_false, diff_false, label='SE=False', color='#1f77b4')
        ax_diffs[count_ax].axvline(x=t_switch, linestyle='--', color='k', label='Switch')
        ax_diffs[count_ax].legend(frameon=False, fontsize=6, loc='lower left')
        ax_diffs[count_ax].set_yscale('symlog', linthresh=1e-5)
        ax_diffs[count_ax].set_xlabel('Time', fontsize=6)
        if count_ax == 0:
            ax_diffs[count_ax].set_ylabel('Difference $v_{C}-V_{ref}$', fontsize=6)

        if count_ax == 0 or count_ax == 1:
            ax_diffs[count_ax].legend(frameon=False, fontsize=6, loc='center right')

        else:
            ax_diffs[count_ax].legend(frameon=False, fontsize=6, loc='upper right')

        count_ax += 1

    fig_diffs.savefig('data/difference_estimation_generic_implicit.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig_diffs)


# def error_over_time(dt_list, cwd='./'):


def iterations_over_time(dt_list, maxiter, cwd='./'):
    """
    Routine  to plot the number of iterations over time using switch estimator or not. Produces the iters_generic_implicit.png file
    """

    iters_time_true = []
    iters_time_false = []
    times_true = []
    times_false = []
    t_switches = []

    for dt_item in dt_list:
        f1 = open(cwd + 'data/battery_implicit_dt{}_USETrue.dat'.format(dt_item), 'rb')
        stats_true = dill.load(f1)
        f1.close()

        f2 = open(cwd + 'data/battery_implicit_dt{}_USEFalse.dat'.format(dt_item), 'rb')
        stats_false = dill.load(f2)
        f2.close()

        iter_counts_true_val = get_sorted(stats_true, type='niter', recomputed=False, sortby='time')
        iter_counts_false_val = get_sorted(stats_false, type='niter', recomputed=False, sortby='time')

        iters_time_true.append([v[1] for v in iter_counts_true_val])
        iters_time_false.append([v[1] for v in iter_counts_false_val])

        times_true.append([v[1] for v in iter_counts_true_val])
        times_false.append([v[1] for v in iter_counts_false_val])

        val_switch = get_sorted(stats_true, type='switch1', sortby='time')
        t_switch = [v[0] for v in val_switch]
        t_switches.append(t_switch[0])

    setup_mpl()
    fig_iter_all, ax_iter_all = plt_helper.plt.subplots(
        nrows=2, ncols=len(dt_list), figsize=(2 * len(dt_list) - 1, 3), sharex='col', sharey='row'
    )
    for row in range(2):
        for col in range(len(dt_list)):
            if row == 0:
                # SE = False
                ax_iter_all[row, col].plot(times_false[col], iters_time_false[col], label='SE=False')
                ax_iter_all[row, col].set_title('dt={}'.format(dt_list[col]))
                ax_iter_all[row, col].set_ylim(1, maxiter)

            else:
                # SE = True
                ax_iter_all[row, col].plot(times_true[col], iters_time_true[col], label='SE=True')
                ax_iter_all[row, col].axvline(x=t_switches[col], linestyle='--', color='r', label='Switch')
                ax_iter_all[row, col].set_xlabel('Time', fontsize=6)
                ax_iter_all[row, col].set_ylim(1, maxiter)

            if col == 0:
                ax_iter_all[row, col].set_ylabel('Number iterations', fontsize=6)

            ax_iter_all[row, col].legend(frameon=False, fontsize=6, loc='upper right')
    plt_helper.plt.tight_layout()
    fig_iter_all.savefig('data/iters_generic_implicit.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig_iter_all)


if __name__ == "__main__":
    check()
