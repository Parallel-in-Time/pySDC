import numpy as np
import dill
from pathlib import Path

from pySDC.helpers.stats_helper import get_sorted
from pySDC.core.Collocation import CollBase as Collocation
from pySDC.implementations.problem_classes.Battery import battery, battery_implicit
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.PinTSimE.piline_model import setup_mpl
from pySDC.projects.PinTSimE.battery_model import log_data, proof_assertions_description
import pySDC.helpers.plot_helper as plt_helper

from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator
from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
from pySDC.implementations.convergence_controller_classes.estimate_embedded_error import EstimateEmbeddedErrorNonMPI


def run(dt, problem, sweeper, use_switch_estimator, use_adaptivity, V_ref):
    """
    A simple test program to do SDC/PFASST runs for the battery drain model
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = -1
    level_params['dt'] = dt

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['quad_type'] = 'LOBATTO'
    sweeper_params['num_nodes'] = 5
    # sweeper_params['QI'] = 'LU'  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
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
    step_params['maxiter'] = 4

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 20
    controller_params['hook_class'] = log_data
    controller_params['mssdc_jac'] = False

    # convergence controllers
    convergence_controllers = dict()
    if use_switch_estimator:
        switch_estimator_params = dict()
        convergence_controllers.update({SwitchEstimator: switch_estimator_params})

    if use_adaptivity:
        adaptivity_params = dict()
        adaptivity_params['e_tol'] = 1e-7
        convergence_controllers.update({Adaptivity: adaptivity_params})

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = problem  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = sweeper  # pass sweeper
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params

    if use_switch_estimator or use_adaptivity:
        description['convergence_controllers'] = convergence_controllers

    proof_assertions_description(description, problem_params)

    # set time parameters
    t0 = 0.0
    Tend = 0.3

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
    iter_counts = get_sorted(stats, type='niter', recomputed=False, sortby='time')

    # compute and print statistics
    f = open('data/battery_out.txt', 'w')
    niters = np.array([item[1] for item in iter_counts])

    assert np.mean(niters) <= 11, "Mean number of iterations is too high, got %s" % np.mean(niters)
    f.close()

    return description, stats


def check(cwd='./'):
    """
    Routine to check the differences between using a switch estimator or not
    """

    V_ref = 1.0
    dt_list = [1e-2, 1e-3]
    use_switch_estimator = [True, False]
    use_adaptivity = [True, False]
    restarts_true = []
    restarts_false_adapt = []
    restarts_true_adapt = []

    problem_classes = [battery, battery_implicit]
    sweeper_classes = [imex_1st_order, generic_implicit]

    for problem, sweeper in zip(problem_classes, sweeper_classes):
        for dt_item in dt_list:
            for use_SE in use_switch_estimator:
                for use_A in use_adaptivity:
                    description, stats = run(
                        dt=dt_item,
                        problem=problem,
                        sweeper=sweeper,
                        use_switch_estimator=use_SE,
                        use_adaptivity=use_A,
                        V_ref=V_ref,
                    )

                    fname = 'data/battery_dt{}_USE{}_USA{}_{}.dat'.format(dt_item, use_SE, use_A, sweeper.__name__)
                    f = open(fname, 'wb')
                    dill.dump(stats, f)
                    f.close()

                    if use_SE or use_A:
                        restarts_sorted = np.array(get_sorted(stats, type='restart', recomputed=None))[:, 1]
                        print('Restarts for dt={}: {}'.format(dt_item, np.sum(restarts_sorted)))
                        if use_SE and not use_A:
                            restarts_true.append(np.sum(restarts_sorted))

                        elif not use_SE and use_A:
                            restarts_false_adapt.append(np.sum(restarts_sorted))

                        elif use_SE and use_A:
                            restarts_true_adapt.append(np.sum(restarts_sorted))

        accuracy_check(dt_list, problem.__name__, sweeper.__name__, V_ref)

        differences_around_switch(
            dt_list,
            problem.__name__,
            restarts_true,
            restarts_false_adapt,
            restarts_true_adapt,
            sweeper.__name__,
            V_ref,
        )

        differences_over_time(dt_list, problem.__name__, sweeper.__name__, V_ref)

        iterations_over_time(dt_list, description['step_params']['maxiter'], problem.__name__, sweeper.__name__)

        restarts_true = []
        restarts_false_adapt = []
        restarts_true_adapt = []


def accuracy_check(dt_list, problem, sweeper, V_ref, cwd='./'):
    """
    Routine to check accuracy for different step sizes in case of using adaptivity
    """

    if len(dt_list) > 1:
        setup_mpl()
        fig_acc, ax_acc = plt_helper.plt.subplots(
            1, len(dt_list), figsize=(3 * len(dt_list), 3), sharex='col', sharey='row'
        )

    else:
        setup_mpl()
        fig_acc, ax_acc = plt_helper.plt.subplots(
            1, 1, figsize=(3, 3), sharex='col', sharey='row'
        )

    count_ax = 0
    for dt_item in dt_list:
        f3 = open(cwd + 'data/battery_dt{}_USETrue_USATrue_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_TT = dill.load(f3)
        f3.close()

        f4 = open(cwd + 'data/battery_dt{}_USEFalse_USATrue_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_FT = dill.load(f4)
        f4.close()

        val_switch_TT = get_sorted(stats_TT, type='switch1', sortby='time')
        t_switch_adapt = [v[1] for v in val_switch_TT]
        t_switch_adapt = t_switch_adapt[-1]

        dt_TT_val = get_sorted(stats_TT, type='dt', recomputed=False)
        dt_FT_val = get_sorted(stats_FT, type='dt', recomputed=False)

        e_emb_TT_val = get_sorted(stats_TT, type='e_embedded', recomputed=False)
        e_emb_FT_val = get_sorted(stats_FT, type='e_embedded', recomputed=False)

        times_TT = [v[0] for v in e_emb_TT_val]
        times_FT = [v[0] for v in e_emb_FT_val]

        e_emb_TT = [v[1] for v in e_emb_TT_val]
        e_emb_FT = [v[1] for v in e_emb_FT_val]

        if len(dt_list) > 1:
            ax_acc[count_ax].set_title(r'$\Delta t$={}'.format(dt_item))
            dt1 = ax_acc[count_ax].plot([v[0] for v in dt_TT_val], [v[1] for v in dt_TT_val], 'ko-', label=r'SE+A - $\Delta t$')
            dt2 = ax_acc[count_ax].plot([v[0] for v in dt_FT_val], [v[1] for v in dt_FT_val], 'g-', label=r'A - $\Delta t$')
            ax_acc[count_ax].axvline(x=t_switch_adapt, linestyle='--', linewidth=0.5, color='r', label='Switch')
            ax_acc[count_ax].set_xlabel('Time', fontsize=6)
            if count_ax == 0:
                ax_acc[count_ax].set_ylabel(r'$\Delta t_{adapted}$', fontsize=6)

            e_ax = ax_acc[count_ax].twinx()
            e_plt1 = e_ax.plot(times_TT, e_emb_TT, 'k--', label=r'SE+A - $\epsilon_{emb}$')
            e_plt2 = e_ax.plot(times_FT, e_emb_FT, 'g--', label=r'A - $\epsilon_{emb}$')
            e_ax.set_yscale('log', base=10)
            e_ax.set_ylim(1e-16, 1e-7)
            e_ax.tick_params(labelsize=6)

            lines = dt1 + e_plt1 + dt2 + e_plt2
            labels = [l.get_label() for l in lines]

            ax_acc[count_ax].legend(lines, labels, frameon=False, fontsize=6, loc='upper left')

        else:
            ax_acc.set_title(r'$\Delta t$={}'.format(dt_item))
            dt1 = ax_acc.plot([v[0] for v in dt_TT_val], [v[1] for v in dt_TT_val], 'ko-', label=r'SE+A - $\Delta t$')
            dt2 = ax_acc.plot([v[0] for v in dt_FT_val], [v[1] for v in dt_FT_val], 'go-', label=r'A - $\Delta t$')
            ax_acc.axvline(x=t_switch_adapt, linestyle='--', linewidth=0.5, color='r', label='Switch')
            ax_acc.set_xlabel('Time', fontsize=6)
            ax_acc.set_ylabel(r'$Delta t_{adapted}$', fontsize=6)

            e_ax = ax_acc.twinx()
            e_plt1 = e_ax.plot(times_TT, e_emb_TT, 'k--', label=r'SE+A - $\epsilon_{emb}$')
            e_plt2 = e_ax.plot(times_FT, e_emb_FT, 'g--', label=r'A - $\epsilon_{emb}$')
            e_ax.set_yscale('log', base=10)
            e_ax.tick_params(labelsize=6)

            lines = dt1 + e_plt1 + dt2 + e_plt2
            labels = [l.get_label() for l in lines]

            ax_acc.legend(lines, labels, frameon=False, fontsize=6, loc='upper left')

        count_ax += 1

    fig_acc.savefig('data/embedded_error_adaptivity_{}.png'.format(sweeper), dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig_acc)


def differences_around_switch(dt_list, problem, restarts_true, restarts_false_adapt, restarts_true_adapt, sweeper, V_ref, cwd='./'):
    """
    Routine to plot the differences before, at, and after the switch. Produces the diffs_estimation_<sweeper_class>.png file
    """

    diffs_true_at = []
    diffs_false_before = []
    diffs_false_after = []

    diffs_true_at_adapt = []
    diffs_true_before_adapt = []
    diffs_true_after_adapt = []

    diffs_false_before_adapt = []
    diffs_false_after_adapt = []
    for dt_item in dt_list:
        f1 = open(cwd + 'data/battery_dt{}_USETrue_USAFalse_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_TF = dill.load(f1)
        f1.close()

        f2 = open(cwd + 'data/battery_dt{}_USEFalse_USAFalse_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_FF = dill.load(f2)
        f2.close()

        f3 = open(cwd + 'data/battery_dt{}_USETrue_USATrue_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_TT = dill.load(f3)
        f3.close()

        f4 = open(cwd + 'data/battery_dt{}_USEFalse_USATrue_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_FT = dill.load(f4)
        f4.close()

        val_switch_TF = get_sorted(stats_TF, type='switch1', sortby='time')
        t_switch = [v[1] for v in val_switch_TF]
        t_switch = t_switch[-1]  # battery has only one single switch

        val_switch_TT = get_sorted(stats_TT, type='switch1', sortby='time')
        t_switch_adapt = [v[1] for v in val_switch_TT]
        t_switch_adapt = t_switch_adapt[-1]

        vC_TF = get_sorted(stats_TF, type='voltage C', recomputed=False, sortby='time')
        vC_FT = get_sorted(stats_FT, type='voltage C', recomputed=False, sortby='time')
        vC_TT = get_sorted(stats_TT, type='voltage C', recomputed=False, sortby='time')
        vC_FF = get_sorted(stats_FF, type='voltage C', sortby='time')

        diff_TF, diff_FF = [v[1] - V_ref for v in vC_TF], [v[1] - V_ref for v in vC_FF]
        times_TF, times_FF = [v[0] for v in vC_TF], [v[0] for v in vC_FF]

        diff_FT, diff_TT = [v[1] - V_ref for v in vC_FT], [v[1] - V_ref for v in vC_TT]
        times_FT, times_TT = [v[0] for v in vC_FT], [v[0] for v in vC_TT]

        for m in range(len(times_TF)):
            if np.round(times_TF[m], 15) == np.round(t_switch, 15):
                diffs_true_at.append(diff_TF[m])

        for m in range(1, len(times_FF)):
            if times_FF[m - 1] <= t_switch <= times_FF[m]:
                diffs_false_before.append(diff_FF[m - 1])
                diffs_false_after.append(diff_FF[m])

        for m in range(len(times_TT)):
            if np.round(times_TT[m], 13) == np.round(t_switch_adapt, 13):
                diffs_true_at_adapt.append(diff_TT[m])
                diffs_true_before_adapt.append(diff_TT[m - 1])
                diffs_true_after_adapt.append(diff_TT[m + 1])

        for m in range(len(times_FT)):
            if times_FT[m - 1] <= t_switch <= times_FT[m]:
                diffs_false_before_adapt.append(diff_FT[m - 1])
                diffs_false_after_adapt.append(diff_FT[m])

    setup_mpl()
    fig_around, ax_around = plt_helper.plt.subplots(1, 3, figsize=(9, 3), sharex='col', sharey='row')
    ax_around[0].set_title("Using SE")
    pos11 = ax_around[0].plot(dt_list, diffs_false_before, 'rs-', label='before switch')
    pos12 = ax_around[0].plot(dt_list, diffs_false_after, 'bd--', label='after switch')
    pos13 = ax_around[0].plot(dt_list, diffs_true_at, 'ko--', label='at switch')
    ax_around[0].set_xticks(dt_list)
    ax_around[0].set_xticklabels(dt_list)
    ax_around[0].set_xscale('log', base=10)
    ax_around[0].set_yscale('symlog', linthresh=1e-8)
    ax_around[0].set_ylim(-1, 1)
    ax_around[0].set_xlabel(r'$\Delta t$', fontsize=6)
    ax_around[0].set_ylabel(r'$v_{C}-V_{ref}$', fontsize=6)

    restart_ax0 = ax_around[0].twinx()
    restarts_plt0 = restart_ax0.plot(dt_list, restarts_true, 'cs--', label='Restarts')
    restart_ax0.tick_params(labelsize=6)

    lines = pos11 + pos12 + pos13 + restarts_plt0
    labels = [l.get_label() for l in lines]
    ax_around[0].legend(lines, labels, frameon=False, fontsize=6, loc='lower right')

    ax_around[1].set_title("Using Adaptivity")
    pos21 = ax_around[1].plot(dt_list, diffs_false_before_adapt, 'rs-', label='before switch')
    pos22 = ax_around[1].plot(dt_list, diffs_false_after_adapt, 'bd--', label='after switch')
    ax_around[1].set_xticks(dt_list)
    ax_around[1].set_xticklabels(dt_list)
    ax_around[1].set_xscale('log', base=10)
    ax_around[1].set_yscale('symlog', linthresh=1e-8)
    ax_around[1].set_ylim(-1, 1)
    ax_around[1].set_xlabel(r'$\Delta t$', fontsize=6)

    restart_ax1 = ax_around[1].twinx()
    restarts_plt1 = restart_ax1.plot(dt_list, restarts_false_adapt, 'cs--', label='Restarts')
    restart_ax1.tick_params(labelsize=6)

    lines = pos21 + pos22 + restarts_plt1
    labels = [l.get_label() for l in lines]
    ax_around[1].legend(lines, labels, frameon=False, fontsize=6, loc='lower right')

    ax_around[2].set_title("Using SE + Adaptivity")
    pos31 = ax_around[2].plot(dt_list, diffs_true_before_adapt, 'rs-', label='before switch')
    pos32 = ax_around[2].plot(dt_list, diffs_true_after_adapt, 'bd--', label='after switch')
    pos33 = ax_around[2].plot(dt_list, diffs_true_at_adapt, 'ko--', label='at switch')
    ax_around[2].set_xticks(dt_list)
    ax_around[2].set_xticklabels(dt_list)
    ax_around[2].set_xscale('log', base=10)
    ax_around[2].set_yscale('symlog', linthresh=1e-8)
    ax_around[2].set_ylim(-1, 1)
    ax_around[2].set_xlabel(r'$\Delta t$', fontsize=6)

    restart_ax2 = ax_around[2].twinx()
    restarts_plt2 = restart_ax2.plot(dt_list, restarts_true_adapt, 'cs--', label='Restarts')
    restart_ax2.tick_params(labelsize=6)

    lines = pos31 + pos32 + pos33 + restarts_plt2
    labels = [l.get_label() for l in lines]
    ax_around[2].legend(frameon=False, fontsize=6, loc='lower right')

    fig_around.savefig('data/diffs_estimation_{}.png'.format(sweeper), dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig_around)


def differences_over_time(dt_list, problem, sweeper, V_ref, cwd='./'):
    """
    Routine to plot the differences in time using the switch estimator or not. Produces the difference_estimation_<sweeper_class>.png file
    """

    if len(dt_list) > 1:
        setup_mpl()
        fig_diffs, ax_diffs = plt_helper.plt.subplots(
            2, len(dt_list), figsize=(3 * len(dt_list), 4), sharex='col', sharey='row'
        )

    else:
        setup_mpl()
        fig_diffs, ax_diffs = plt_helper.plt.subplots(2, 1, figsize=(3, 3))

    count_ax = 0
    for dt_item in dt_list:
        f1 = open(cwd + 'data/battery_dt{}_USETrue_USAFalse_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_TF = dill.load(f1)
        f1.close()

        f2 = open(cwd + 'data/battery_dt{}_USEFalse_USAFalse_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_FF = dill.load(f2)
        f2.close()

        f3 = open(cwd + 'data/battery_dt{}_USETrue_USATrue_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_TT = dill.load(f3)
        f3.close()

        f4 = open(cwd + 'data/battery_dt{}_USEFalse_USATrue_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_FT = dill.load(f4)
        f4.close()

        val_switch_TF = get_sorted(stats_TF, type='switch1', sortby='time')
        t_switch_TF = [v[1] for v in val_switch_TF]
        t_switch_TF = t_switch_TF[-1]  # battery has only one single switch

        val_switch_TT = get_sorted(stats_TT, type='switch1', sortby='time')
        t_switch_adapt = [v[1] for v in val_switch_TT]
        t_switch_adapt = t_switch_adapt[-1]

        dt_FT = np.array(get_sorted(stats_FT, type='dt', recomputed=False, sortby='time'))
        dt_TT = np.array(get_sorted(stats_TT, type='dt', recomputed=False, sortby='time'))

        restart_FT = np.array(get_sorted(stats_FT, type='restart', recomputed=None, sortby='time'))
        restart_TT = np.array(get_sorted(stats_TT, type='restart', recomputed=None, sortby='time'))

        vC_TF = get_sorted(stats_TF, type='voltage C', recomputed=False, sortby='time')
        vC_FT = get_sorted(stats_FT, type='voltage C', recomputed=False, sortby='time')
        vC_TT = get_sorted(stats_TT, type='voltage C', recomputed=False, sortby='time')
        vC_FF = get_sorted(stats_FF, type='voltage C', sortby='time')

        diff_TF, diff_FF = [v[1] - V_ref for v in vC_TF], [v[1] - V_ref for v in vC_FF]
        times_TF, times_FF = [v[0] for v in vC_TF], [v[0] for v in vC_FF]

        diff_FT, diff_TT = [v[1] - V_ref for v in vC_FT], [v[1] - V_ref for v in vC_TT]
        times_FT, times_TT = [v[0] for v in vC_FT], [v[0] for v in vC_TT]

        if len(dt_list) > 1:
            ax_diffs[0, count_ax].set_title(r'$\Delta t$={}'.format(dt_item))
            ax_diffs[0, count_ax].plot(times_TF, diff_TF, label='SE=True, A=False', color='#ff7f0e')
            ax_diffs[0, count_ax].plot(times_FF, diff_FF, label='SE=False, A=False', color='#1f77b4')
            ax_diffs[0, count_ax].plot(times_FT, diff_FT, label='SE=False, A=True', color='red', linestyle='--')
            ax_diffs[0, count_ax].plot(times_TT, diff_TT, label='SE=True, A=True', color='limegreen', linestyle='-.')
            ax_diffs[0, count_ax].axvline(x=t_switch_TF, linestyle='--', linewidth=0.5, color='k', label='Switch')
            ax_diffs[0, count_ax].legend(frameon=False, fontsize=6, loc='lower left')
            ax_diffs[0, count_ax].set_yscale('symlog', linthresh=1e-5)
            if count_ax == 0:
                ax_diffs[0, count_ax].set_ylabel('Difference $v_{C}-V_{ref}$', fontsize=6)

            if count_ax == 0 or count_ax == 1:
                ax_diffs[0, count_ax].legend(frameon=False, fontsize=6, loc='upper right')

            else:
                ax_diffs[0, count_ax].legend(frameon=False, fontsize=6, loc='upper right')

            ax_diffs[1, count_ax].plot(dt_FT[:, 0], dt_FT[:, 1], label=r'$\Delta t$ - SE=F, A=T', color='red', linestyle='--')
            ax_diffs[1, count_ax].plot([None], [None], label='Restart - SE=F, A=T', color='grey', linestyle='-.')

            for i in range(len(restart_FT)):
                if restart_FT[i, 1] > 0:
                    ax_diffs[1, count_ax].axvline(restart_FT[i, 0], color='grey', linestyle='-.')

            ax_diffs[1, count_ax].plot(dt_TT[:, 0], dt_TT[:, 1], label=r'$ \Delta t$ - SE=T, A=T', color='limegreen', linestyle='-.')
            ax_diffs[1, count_ax].plot([None], [None], label='Restart - SE=T, A=T', color='black', linestyle='-.')

            for i in range(len(restart_TT)):
                if restart_TT[i, 1] > 0:
                    ax_diffs[1, count_ax].axvline(restart_TT[i, 0], color='black', linestyle='-.')

            ax_diffs[1, count_ax].set_xlabel('Time', fontsize=6)
            if count_ax == 0:
                ax_diffs[1, count_ax].set_ylabel(r'$\Delta t_{adapted}$', fontsize=6)

            ax_diffs[1, count_ax].legend(frameon=True, fontsize=6, loc='upper left')

        else:
            ax_diffs[0].set_title(r'$\Delta t$={}'.format(dt_item))
            ax_diffs[0].plot(times_TF, diff_TF, label='SE=True', color='#ff7f0e')
            ax_diffs[0].plot(times_FF, diff_FF, label='SE=False', color='#1f77b4')
            ax_diffs[0].plot(times_FT, diff_FT, label='SE=False, A=True', color='red', linestyle='--')
            ax_diffs[0].plot(times_TT, diff_TT, label='SE=True, A=True', color='limegreen', linestyle='-.')
            ax_diffs[0].axvline(x=t_switch_TF, linestyle='--', linewidth=0.5, color='k', label='Switch')
            ax_diffs[0].legend(frameon=False, fontsize=6, loc='lower left')
            ax_diffs[0].set_yscale('symlog', linthresh=1e-5)
            ax_diffs[0].set_ylabel('Difference $v_{C}-V_{ref}$', fontsize=6)
            ax_diffs[0].legend(frameon=False, fontsize=6, loc='center right')

            ax_diffs[1].plot(dt_FT[:, 0], dt_FT[:, 1], label='SE=False, A=True', color='red', linestyle='--')
            ax_diffs[1].plot(dt_TT[:, 0], dt_TT[:, 1], label='SE=True, A=True', color='limegreen', linestyle='-.')
            ax_diffs[1].set_xlabel('Time', fontsize=6)
            ax_diffs[1].set_ylabel(r'$\Delta t_{adapted}$', fontsize=6)

            ax_diffs[1].legend(frameon=False, fontsize=6, loc='upper right')

        count_ax += 1

    plt_helper.plt.tight_layout()
    fig_diffs.savefig('data/difference_estimation_{}.png'.format(sweeper), dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig_diffs)

def iterations_over_time(dt_list, maxiter, problem, sweeper, cwd='./'):
    """
    Routine  to plot the number of iterations over time using switch estimator or not. Produces the iters_<sweeper_class>.png file
    """

    iters_time_TF = []
    iters_time_FF = []
    iters_time_TT = []
    iters_time_FT = []
    times_TF = []
    times_FF = []
    times_TT = []
    times_FT = []
    t_switches_TF = []
    t_switches_adapt = []

    for dt_item in dt_list:
        f1 = open(cwd + 'data/battery_dt{}_USETrue_USAFalse_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_TF = dill.load(f1)
        f1.close()

        f2 = open(cwd + 'data/battery_dt{}_USEFalse_USAFalse_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_FF = dill.load(f2)
        f2.close()

        f3 = open(cwd + 'data/battery_dt{}_USETrue_USATrue_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_TT = dill.load(f3)
        f3.close()

        f4 = open(cwd + 'data/battery_dt{}_USEFalse_USATrue_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_FT = dill.load(f4)
        f4.close()

        iter_counts_TF_val = get_sorted(stats_TF, type='niter', recomputed=False, sortby='time')
        iter_counts_TT_val = get_sorted(stats_TT, type='niter', recomputed=False, sortby='time')
        iter_counts_FT_val = get_sorted(stats_FT, type='niter', recomputed=False, sortby='time')
        iter_counts_FF_val = get_sorted(stats_FF, type='niter', recomputed=False, sortby='time')

        iters_time_TF.append([v[1] for v in iter_counts_TF_val])
        iters_time_TT.append([v[1] for v in iter_counts_TT_val])
        iters_time_FT.append([v[1] for v in iter_counts_FT_val])
        iters_time_FF.append([v[1] for v in iter_counts_FF_val])

        times_TF.append([v[0] for v in iter_counts_TF_val])
        times_TT.append([v[0] for v in iter_counts_TT_val])
        times_FT.append([v[0] for v in iter_counts_FT_val])
        times_FF.append([v[0] for v in iter_counts_FF_val])

        val_switch_TF = get_sorted(stats_TF, type='switch1', sortby='time')
        t_switch_TF = [v[1] for v in val_switch_TF]
        t_switches_TF.append(t_switch_TF[-1])

        val_switch_TT = get_sorted(stats_TT, type='switch1', sortby='time')
        t_switch_adapt = [v[1] for v in val_switch_TT]
        t_switches_adapt.append(t_switch_adapt[-1])

    if len(dt_list) > 1:
        setup_mpl()
        fig_iter_all, ax_iter_all = plt_helper.plt.subplots(
            nrows=1, ncols=len(dt_list), figsize=(2 * len(dt_list) - 1, 3), sharex='col', sharey='row'
        )
        for col in range(len(dt_list)):
            ax_iter_all[col].plot(times_FF[col], iters_time_FF[col], label='SE=F, A=F')
            ax_iter_all[col].plot(times_TF[col], iters_time_TF[col], label='SE=T, A=F')
            ax_iter_all[col].plot(times_TT[col], iters_time_TT[col], '--', label='SE=T, A=T')
            ax_iter_all[col].plot(times_FT[col], iters_time_FT[col], '--', label='SE=F, A=T')
            ax_iter_all[col].axvline(x=t_switches_TF[col], linestyle='--', linewidth=0.5, color='k', label='Switch')
            if t_switches_adapt[col] != t_switches_TF[col]:
                ax_iter_all[col].axvline(x=t_switches_adapt[col], linestyle='--', linewidth=0.5, color='k', label='Switch')
            ax_iter_all[col].set_title('dt={}'.format(dt_list[col]))
            ax_iter_all[col].set_ylim(0, maxiter + 2)
            ax_iter_all[col].set_xlabel('Time', fontsize=6)

            if col == 0:
                ax_iter_all[col].set_ylabel('Number iterations', fontsize=6)

            ax_iter_all[col].legend(frameon=False, fontsize=6, loc='upper right')
    else:
        setup_mpl()
        fig_iter_all, ax_iter_all = plt_helper.plt.subplots(
            nrows=1, ncols=1, figsize=(3, 3)
        )

        ax_iter_all.plot(times_FF[0], iters_time_FF[0], label='SE=False')
        ax_iter_all.plot(times_TF[0], iters_time_TF[0], label='SE=True')
        ax_iter_all.plot(times_TT[0], iters_time_TT[0], '--', label='SE=T, A=T')
        ax_iter_all.plot(times_FT[0], iters_time_FT[0], '--', label='SE=F, A=T')
        ax_iter_all.axvline(x=t_switches_TF[0], linestyle='--', linewidth=0.5, color='k', label='Switch')
        if t_switches_adapt[0] != t_switches_TF[0]:
            ax_iter_all.axvline(x=t_switches_adapt[0], linestyle='--', linewidth=0.5, color='k', label='Switch')
        ax_iter_all.set_title('dt={}'.format(dt_list[0]))
        ax_iter_all.set_ylim(0, maxiter + 2)
        ax_iter_all.set_xlabel('Time', fontsize=6)

        ax_iter_all.set_ylabel('Number iterations', fontsize=6)
        ax_iter_all.legend(frameon=False, fontsize=6, loc='upper right')

    plt_helper.plt.tight_layout()
    fig_iter_all.savefig('data/iters_{}.png'.format(sweeper), dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig_iter_all)


if __name__ == "__main__":
    check()
