import numpy as np
import dill

from pySDC.helpers.stats_helper import get_sorted
from pySDC.core import CollBase as Collocation
from pySDC.implementations.problem_classes.Battery import battery
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.PinTSimE.piline_model import setup_mpl
from pySDC.projects.PinTSimE.battery_model import log_data, proof_assertions_description
import pySDC.helpers.plot_helper as plt_helper

from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator


def run(dt, use_switch_estimator=True, V_ref=1):
    """
    A simple test program to do SDC/PFASST runs for the battery drain model
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-13
    level_params['dt'] = dt

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = Collocation
    sweeper_params['node_type'] = 'LEGENDRE'
    sweeper_params['quad_type'] = 'LOBATTO'
    sweeper_params['num_nodes'] = 5
    sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    sweeper_params['initial_guess'] = 'zero'

    # initialize problem parameters
    problem_params = dict()
    problem_params['Vs'] = 5.0
    problem_params['Rs'] = 0.5
    problem_params['C'] = 1
    problem_params['R'] = 1
    problem_params['L'] = 1
    problem_params['alpha'] = 5
    problem_params['V_ref'] = V_ref
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
    description['problem_class'] = battery  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper
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
    fname = 'data/battery.dat'
    f = open(fname, 'wb')
    dill.dump(stats, f)
    f.close()

    # filter statistics by number of iterations
    iter_counts = get_sorted(stats, type='niter', sortby='time')

    # compute and print statistics
    f = open('data/battery_out.txt', 'w')
    niters = np.array([item[1] for item in iter_counts])

    # depends on which step sizes are used
    assert np.mean(niters) <= 8, "Mean number of iterations is too high, got %s" % np.mean(niters)
    f.close()

    return stats


def check(cwd='./'):
    """
    Routine to check the differences between using a switch estimator or not
    """

    V_ref = 1.0
    dt_list = [1E-1, 1E-2, 1E-3, 1E-4]
    use_switch_estimator = [True, False]
    restarts_all = []
    for dt_item in dt_list:
        for item in use_switch_estimator:
            stats = run(dt=dt_item, use_switch_estimator=item, V_ref=V_ref)

            fname = 'data/battery_dt{}_USE{}.dat'.format(dt_item, item)
            f = open(fname, 'wb')
            dill.dump(stats, f)
            f.close()

            if item:
                restarts = np.array(get_sorted(stats, type='restart', recomputed=False))[:, 1]
                restarts_all.append(np.sum(restarts))

    val_switch_all = []
    diff_true_all = []
    diff_false_all_before = []
    diff_false_all_after = []
    for dt_item in dt_list:
        f1 = open(cwd + 'data/battery_dt{}_USETrue.dat'.format(dt_item), 'rb')
        stats_true = dill.load(f1)
        f1.close()

        f2 = open(cwd + 'data/battery_dt{}_USEFalse.dat'.format(dt_item), 'rb')
        stats_false = dill.load(f2)
        f2.close()

        val_switch = get_sorted(stats_true, type='switch1', sortby='time')
        t_switch = [v[0] for v in val_switch]
        vC_switch = [v[1] for v in val_switch]

        if len(t_switch) > 1:
            t_switch = t_switch[-1]  # battery has only one single switch
            vC_switch = vC_switch[-1]

        else:
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
        fig1, ax1 = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
        ax1.set_title('Simulation of drain battery model using SE')
        ax1.plot(times_true, [v[1] for v in vC_true], label='voltage C')
        ax1.legend(frameon=False, fontsize=10, loc='upper right')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Energy')
        fig1.savefig('data/simulation_dt{}.png'.format(dt_item), dpi=300, bbox_inches='tight')

        setup_mpl()
        fig, ax = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
        ax.set_title('Time evolution of $v_{C}-V_{ref}$')
        ax.plot(times_true, diff_true, label='SE=True', color='#ff7f0e')
        ax.plot(times_false, diff_false, label='SE=False', color='#1f77b4')
        ax.axvline(x=t_switch, linestyle='--', color='k', label='Switch')
        ax.legend(frameon=False, fontsize=10, loc='lower left')
        ax.set_yscale('symlog', linthresh=1e-5)
        ax.set_xlabel('Time')

        fig.savefig('data/difference_estimation_dt{}.png'.format(dt_item), dpi=300, bbox_inches='tight')

    setup_mpl()
    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(3, 3))
    ax.set_title("Difference $v_{C}-V_{ref}$")
    pos1 = ax.plot(dt_list, diff_false_all_before, 'rs-', label='SE=False - before switch')
    pos2 = ax.plot(dt_list, diff_false_all_after, 'bd-', label='SE=False - after switch')
    pos3 = ax.plot(dt_list, diff_true_all, 'kd-', label='SE=True')
    ax.set_xticks(dt_list)
    ax.set_xticklabels(dt_list)
    ax.set_xscale('log', base=10)
    ax.set_yscale('symlog', linthresh=1e-8)
    ax.set_ylim(-1, 1)
    ax.set_xlabel("$\Delta t$")

    restart_ax = ax.twinx()
    restarts = restart_ax.plot(dt_list, restarts_all, 'cs--', label='Restarts')
    restart_ax.set_ylabel("Restarts")

    lines = pos1 + pos2 + pos3 + restarts
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, frameon=False, fontsize=8, loc='center right')

    fig.savefig('data/diffs_estimation.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    check()
