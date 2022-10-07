import numpy as np
import dill

from pySDC.helpers.stats_helper import get_sorted
from pySDC.core import CollBase as Collocation
from pySDC.implementations.problem_classes.Battery_2Condensators import battery_2condensators
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.projects.PinTSimE.piline_model import setup_mpl
from pySDC.projects.PinTSimE.battery_2condensators_model import log_data, proof_assertions_description
import pySDC.helpers.plot_helper as plt_helper

from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator
from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity


def run(dt, use_switch_estimator=True, use_adaptivity=False):

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-13
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
    problem_params['C1'] = 1
    problem_params['C2'] = 1
    problem_params['R'] = 1
    problem_params['L'] = 1
    problem_params['alpha'] = 5
    problem_params['V_ref'] = np.array([1, 1])  # [V_ref1, V_ref2]
    problem_params['set_switch'] = np.array([False, False], dtype=bool)
    problem_params['t_switch'] = np.zeros(np.shape(problem_params['V_ref'])[0])

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 20

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = log_data

    # convergence controllers
    convergence_controllers = dict()
    if use_switch_estimator:
        switch_estimator_params = {}
        # convergence_controllers = {SwitchEstimator: switch_estimator_params}
        convergence_controllers[SwitchEstimator] = switch_estimator_params

    if use_adaptivity:
        adaptivity_params = {'e_tol': 1e-7}
        # convergence_controllers = {Adaptivity: adaptivity_params}
        convergence_controllers[Adaptivity] = adaptivity_params
        controller_params['mssdc_jac'] = False

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = battery_2condensators  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params
    description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class

    if use_switch_estimator or use_adaptivity:
        description['convergence_controllers'] = convergence_controllers

    proof_assertions_description(description, problem_params)

    # set time parameters
    t0 = 0.0
    Tend = 3.5

    # instantiate controller
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # fname = 'data/battery_2condensators.dat'
    fname = 'battery_2condensators.dat'
    f = open(fname, 'wb')
    dill.dump(stats, f)
    f.close()

    # filter statistics by number of iterations
    iter_counts = get_sorted(stats, type='niter', sortby='time')

    # compute and print statistics
    min_iter = 20
    max_iter = 0

    f = open('battery_2condensators_out.txt', 'w')
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

    assert np.mean(niters) <= 10, "Mean number of iterations is too high, got %s" % np.mean(niters)
    f.close()

    return stats, description


def check(cwd='./'):
    """
        Routine to check the differences between using a switch estimator or not
    """

    dt_list = [1e-1, 1e-2, 1e-3]
    use_switch_estimator = [True, False]
    restarts_all = []
    for dt_item in dt_list:
        for item in use_switch_estimator:
            stats, description = run(dt=dt_item, use_switch_estimator=item)

            fname = 'battery_2condensators_dt{}_USE{}.dat'.format(dt_item, item)
            f = open(fname, 'wb')
            dill.dump(stats, f)
            f.close()

            if item:
                restarts = np.array(get_sorted(stats, type='restart', recomputed=False))[:, 1]
                restarts_all.append(np.sum(restarts))
                print("Restarts for dt: ", dt_item, " -- ", np.sum(restarts))

    V_ref = description['problem_params']['V_ref']

    val_switch_all = []
    diff_true_all1 = []
    diff_false_all_before1 = []
    diff_false_all_after1 = []
    diff_true_all2 = []
    diff_false_all_before2 = []
    diff_false_all_after2 = []
    for dt_item in dt_list:
        f1 = open(cwd + 'battery_2condensators_dt{}_USETrue.dat'.format(dt_item), 'rb')
        stats_true = dill.load(f1)
        f1.close()

        f2 = open(cwd + 'battery_2condensators_dt{}_USEFalse.dat'.format(dt_item), 'rb')
        stats_false = dill.load(f2)
        f2.close()

        val_switch1 = get_sorted(stats_true, type='switch1', sortby='time')
        val_switch2 = get_sorted(stats_true, type='switch2', sortby='time')
        t_switch1 = [v[0] for v in val_switch1]
        t_switch2 = [v[0] for v in val_switch2]

        if len(t_switch1) > 1:
            t_switch1 = t_switch1[-1]

        else:
            t_switch1 = t_switch1[0]

        if len(t_switch2) > 1:
            t_switch2 = t_switch2[-1]

        else:
            t_switch2 = t_switch2[0]

        val_switch_all.append([t_switch1, t_switch2])

        vC1_true = get_sorted(stats_true, type='voltage C1', recomputed=False, sortby='time')
        vC2_true = get_sorted(stats_true, type='voltage C2', recomputed=False, sortby='time')
        vC1_false = get_sorted(stats_false, type='voltage C1', sortby='time')
        vC2_false = get_sorted(stats_false, type='voltage C2', sortby='time')

        diff_true1 = [v[1] - V_ref[0] for v in vC1_true]
        diff_true2 = [v[1] - V_ref[1] for v in vC2_true]
        diff_false1 = [v[1] - V_ref[0] for v in vC1_false]
        diff_false2 = [v[1] - V_ref[1] for v in vC2_false]

        times_true1 = [v[0] for v in vC1_true]
        times_true2 = [v[0] for v in vC2_true]
        times_false1 = [v[0] for v in vC1_false]
        times_false2 = [v[0] for v in vC2_false]

        for m in range(len(times_true1)):
            if np.round(times_true1[m], 15) == np.round(t_switch1, 15):
                diff_true_all1.append(diff_true1[m])

        for m in range(len(times_true2)):
            if np.round(times_true2[m], 15) == np.round(t_switch2, 15):
                diff_true_all2.append(diff_true2[m])

        for m in range(1, len(times_false1)):
            if times_false1[m - 1] < t_switch1 < times_false1[m]:
                diff_false_all_before1.append(diff_false1[m - 1])
                diff_false_all_after1.append(diff_false1[m])

        for m in range(1, len(times_false2)):
            if times_false2[m - 1] < t_switch2 < times_false2[m]:
                diff_false_all_before2.append(diff_false2[m - 1])
                diff_false_all_after2.append(diff_false2[m])

        setup_mpl()
        fig1, ax1 = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
        ax1.set_title('Time evolution of $v_{C_{1}}-V_{ref1}$')
        ax1.plot(times_true1, diff_true1, label='SE=True', color='#ff7f0e')
        ax1.plot(times_false1, diff_false1, label='SE=False', color='#1f77b4')
        ax1.axvline(x=t_switch1, linestyle='--', color='k', label='Switch1')
        ax1.legend(frameon=False, fontsize=10, loc='lower left')
        # ax1.set_yticks(np.arange(-3e-2, 3e-2))
        # ax1.set_xlim(t_switch-5e-1, 2.5)
        ax1.set_yscale('symlog', linthresh=1e-5)
        ax1.set_xlabel('Time')

        fig1.savefig('difference_estimation_vC1_dt{}.png'.format(dt_item), dpi=300, bbox_inches='tight')

        setup_mpl()
        fig2, ax2 = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
        ax2.set_title('Time evolution of $v_{C_{2}}-V_{ref2}$')
        ax2.plot(times_true2, diff_true2, label='SE=True', color='#ff7f0e')
        ax2.plot(times_false2, diff_false2, label='SE=False', color='#1f77b4')
        ax2.axvline(x=t_switch2, linestyle='--', color='k', label='Switch2')
        ax2.legend(frameon=False, fontsize=10, loc='lower left')
        # ax2.set_yticks(np.arange(-3e-2, 3e-2))
        # ax2.set_xlim(t_switch-5e-1, 2.5)
        ax2.set_yscale('symlog', linthresh=1e-5)
        ax2.set_xlabel('Time')

        fig2.savefig('difference_estimation_vC2_dt{}.png'.format(dt_item), dpi=300, bbox_inches='tight')

    setup_mpl()
    fig1, ax1 = plt_helper.plt.subplots(1, 1, figsize=(3, 3))
    ax1.set_title("Difference $v_{C_{1}}-V_{ref1}$")
    pos1 = ax1.plot(dt_list, diff_false_all_before1, 'rs-', label='SE=False - before switch1')
    pos2 = ax1.plot(dt_list, diff_false_all_after1, 'bd-', label='SE=False - after switch1')
    pos3 = ax1.plot(dt_list, diff_true_all1, 'kd-', label='SE=True')
    ax1.set_xticks(dt_list)
    ax1.set_xticklabels(dt_list)
    ax1.set_xscale('log', base=10)
    ax1.set_yscale('symlog', linthresh=1e-10)
    ax1.set_ylim(-2, 2)
    ax1.set_xlabel("$\Delta t$")

    restart_ax = ax1.twinx()
    restarts = restart_ax.plot(dt_list, restarts_all, 'cs--', label='Restarts')
    restart_ax.set_ylabel("Restarts")

    lines = pos1 + pos2 + pos3 + restarts
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, frameon=False, fontsize=8, loc='center right')

    fig1.savefig('diffs_estimation_vC1.png', dpi=300, bbox_inches='tight')

    setup_mpl()
    fig2, ax2 = plt_helper.plt.subplots(1, 1, figsize=(3, 3))
    ax2.set_title("Difference $v_{C_{2}}-V_{ref2}$")
    pos1 = ax2.plot(dt_list, diff_false_all_before2, 'rs-', label='SE=False - before switch2')
    pos2 = ax2.plot(dt_list, diff_false_all_after2, 'bd-', label='SE=False - after switch2')
    pos3 = ax2.plot(dt_list, diff_true_all2, 'kd-', label='SE=True')
    ax2.set_xticks(dt_list)
    ax2.set_xticklabels(dt_list)
    ax2.set_xscale('log', base=10)
    ax2.set_yscale('symlog', linthresh=1e-10)
    ax2.set_ylim(-2, 2)
    ax2.set_xlabel("$\Delta t$")

    restart_ax = ax2.twinx()
    restarts = restart_ax.plot(dt_list, restarts_all, 'cs--', label='Restarts')
    restart_ax.set_ylabel("Restarts")

    lines = pos1 + pos2 + pos3 + restarts
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, frameon=False, fontsize=8, loc='center right')

    fig2.savefig('diffs_estimation_vC2.png', dpi=300, bbox_inches='tight')


if __name__ == "__main__":
    check()
