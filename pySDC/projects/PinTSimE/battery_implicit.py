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
from pySDC.core.Hooks import hooks

from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator


def main(dt=1e-2, use_switch_estimator=True):
    """
    A simple test program to do SDC/PFASST runs for the battery drain model
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-10
    level_params['dt'] = dt

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'LOBATTO'
    sweeper_params['num_nodes'] = 5
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
    problem_params['alpha'] = 5.0
    problem_params['V_ref'] = 1.0
    problem_params['set_switch'] = np.array([False], dtype=bool)
    problem_params['t_switch'] = np.zeros(1)

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

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
    min_iter = 20
    max_iter = 0

    f = open('data/battery_implicit_out.txt', 'w')
    niters = np.array([item[1] for item in iter_counts])
    out = '   Mean number of iterations: %4.2f' % np.mean(niters)
    f.write(out + '\n')
    print(out)
    for item in iter_counts:
        out = 'Number of iterations for time %4.2f: %1i' % item
        f.write(out + '\n')
        # print(out)
        min_iter = min(min_iter, item[1])
        max_iter = max(max_iter, item[1])

    assert np.mean(niters) <= 21.0, "Mean number of iterations is too high, got %s" % np.mean(niters)
    f.close()

    plot_voltages(description, use_switch_estimator)

    return description, stats


def plot_voltages(description, use_switch_estimator, cwd='./'):
    """
    Routine to plot voltages of implicit battery model
    """

    f = open(cwd + 'data/battery_implicit.dat', 'rb')
    stats = dill.load(f)
    f.close()

    # convert filtered statistics to list of iterations count, sorted by process
    cL = get_sorted(stats, type='current L', sortby='time')
    vC = get_sorted(stats, type='voltage C', sortby='time')

    times = [v[0] for v in cL]

    setup_mpl()
    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
    ax.plot(times, [v[1] for v in cL], label=r'$i_L$')
    ax.plot(times, [v[1] for v in vC], label=r'$v_C$')

    if use_switch_estimator:
        val_switch = get_sorted(stats, type='switch1', sortby='time')
        t_switch = [v[0] for v in val_switch]
        ax.axvline(x=t_switch[0], linestyle='--', color='k', label='Switch')

    ax.legend(frameon=False, fontsize=12, loc='upper right')

    ax.set_xlabel('Time')
    ax.set_ylabel('Energy')

    fig.savefig('data/battery_implicit_model_solution.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


def run_check(cwd='./'):
    """
    Function to compare runs of battery model using explicit Euler as sweeper
    It will compared the order between using the SE and not using the SE
    """

    V_ref = 1.0
    dt_list = [4e-1, 4e-2, 4e-3]
    use_switch_estimator = [True, False]
    restarts_all = []
    for dt_item in dt_list:
        for use_SE_item in use_switch_estimator:
            description, stats = main(dt_item, use_SE_item)

            fname = 'data/battery_implicit_dt{}_USE{}.dat'.format(dt_item, use_SE_item)
            f = open(fname, 'wb')
            dill.dump(stats, f)
            f.close()

            if use_SE_item:
                restarts = np.array(get_sorted(stats, type='restart', recomputed=False))[:, 1]
                restarts_all.append(np.sum(restarts))
                print("Restarts for dt: ", dt_item, " -- ", np.sum(restarts))

    niters_true_all = []
    niters_false_all = []
    val_switch_all = []
    diff_true_all = []
    diff_false_all_before = []
    diff_false_all_after = []
    for dt_item in dt_list:
        f1 = open(cwd + 'data/battery_implicit_dt{}_USETrue.dat'.format(dt_item), 'rb')
        stats_true = dill.load(f1)
        f1.close()

        f2 = open(cwd + 'data/battery_implicit_dt{}_USEFalse.dat'.format(dt_item), 'rb')
        stats_false = dill.load(f2)
        f2.close()

        iter_counts_true = get_sorted(stats_true, type='niter', sortby='time')
        niters_true_all.append(np.mean(np.array([item[1] for item in iter_counts_true])))

        iter_counts_false = get_sorted(stats_false, type='niter', sortby='time')
        niters_false_all.append(np.mean(np.array([item[1] for item in iter_counts_false])))

        val_switch = get_sorted(stats_true, type='switch1', sortby='time')
        t_switch = [v[0] for v in val_switch]
        vC_switch = [v[1] for v in val_switch]
        t_switch = t_switch[0]  # battery has only one single switch
        vC_switch = vC_switch[0]
        val_switch_all.append([t_switch, vC_switch])

        vC_true = get_sorted(stats_true, type='voltage C', recomputed=False, sortby='time')
        vC_false = get_sorted(stats_false, type='voltage C', sortby='time')

        diff_true = [v[1] - V_ref for v in vC_true]
        diff_false = [v[1] - V_ref for v in vC_false]

        times_true = [v[0] for v in vC_true]
        times_false = [v[0] for v in vC_false]
        for m in range(len(times_true)):
            if np.round(times_true[m], 15) == np.round(t_switch, 15):
                diff_true_all.append(diff_true[m])

        for m in range(1, len(times_false)):
            if times_false[m - 1] < t_switch < times_false[m]:
                diff_false_all_before.append(diff_false[m - 1])
                diff_false_all_after.append(diff_false[m])

    setup_mpl()
    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
    ax.set_title("Average number of iterations for implicit Euler")
    ax.plot(dt_list, niters_true_all, 'd--', label='SE=True')
    ax.plot(dt_list, niters_false_all, 'd--', label='SE=False')
    ax.legend(frameon=False, fontsize=10, loc='lower left')
    ax.set_xscale('log', base=10)
    ax.set_xlabel(r'$\Delta t$')
    ax.set_ylabel('Average number of iterations')

    fig.savefig('data/dt_niters_{}.png'.format(description['sweeper_class']), dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)

    setup_mpl()
    fig1, ax1 = plt_helper.plt.subplots(1, 1, figsize=(3, 3))
    ax1.set_title("Difference $v_{C}-V_{ref}$")
    pos1 = ax1.plot(dt_list, diff_false_all_before, 'rs-', label='SE=False - before switch')
    pos2 = ax1.plot(dt_list, diff_false_all_after, 'bd--', label='SE=False - after switch')
    pos3 = ax1.plot(dt_list, diff_true_all, 'kd--', label='SE=True')
    ax1.set_xticks(dt_list)
    ax1.set_xticklabels(dt_list)
    ax1.set_xscale('log', base=10)
    ax1.set_yscale('symlog', linthresh=1e-8)
    ax1.set_ylim(-1, 1)
    ax1.set_xlabel(r'$\Delta t$')

    restart_ax = ax1.twinx()
    restarts = restart_ax.plot(dt_list, restarts_all, 'cs--', label='Restarts')
    restart_ax.set_label('Restarts')

    lines = pos1 + pos2 + pos3 + restarts
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, frameon=False, fontsize=8, loc='center right')
    fig1.savefig('data/diffs_estimation_{}.png'.format(description['sweeper_class']), dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig1)


if __name__ == "__main__":
    run_check()
