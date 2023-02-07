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
from pySDC.projects.PinTSimE.battery_model import (
    controller_run,
    check_solution,
    generate_description,
    get_recomputed,
    log_data,
    proof_assertions_description,
)
import pySDC.helpers.plot_helper as plt_helper
from pySDC.core.Hooks import hooks

from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator
from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity


def run(cwd='./'):
    """
    Routine to check the differences between using a switch estimator or not

    Args:
        cwd (str): current working directory
    """

    dt_list = [4e-2, 4e-3]
    t0 = 0.0
    Tend = 0.3

    problem_classes = [battery, battery_implicit]
    sweeper_classes = [imex_1st_order, generic_implicit]

    ncapacitors = 1
    alpha = 1.2
    V_ref = np.array([1.0])
    C = np.array([1.0])

    max_restarts = 1
    use_switch_estimator = [True, False]
    use_adaptivity = [True, False]
    restarts_SE = []
    restarts_adapt = []
    restarts_SE_adapt = []

    for problem, sweeper in zip(problem_classes, sweeper_classes):
        for dt_item in dt_list:
            for use_SE in use_switch_estimator:
                for use_A in use_adaptivity:
                    description, controller_params = generate_description(
                        dt_item,
                        problem,
                        sweeper,
                        log_data,
                        use_A,
                        use_SE,
                        ncapacitors,
                        alpha,
                        V_ref,
                        C,
                        max_restarts,
                    )

                    # Assertions
                    proof_assertions_description(description, use_A, use_SE)

                    stats = controller_run(description, controller_params, use_A, use_SE, t0, Tend)

                    if use_A or use_SE:
                        check_solution(stats, dt_item, problem.__name__, use_A, use_SE)

                    if use_SE:
                        assert (
                            len(get_recomputed(stats, type='switch', sortby='time')) >= 1
                        ), 'No switches found for dt={}!'.format(dt_item)

                    fname = 'data/battery_dt{}_USE{}_USA{}_{}.dat'.format(dt_item, use_SE, use_A, sweeper.__name__)
                    f = open(fname, 'wb')
                    dill.dump(stats, f)
                    f.close()

                    if use_SE or use_A:
                        restarts_sorted = np.array(get_sorted(stats, type='restart', recomputed=None))[:, 1]
                        if use_SE and not use_A:
                            restarts_SE.append(np.sum(restarts_sorted))

                        elif not use_SE and use_A:
                            restarts_adapt.append(np.sum(restarts_sorted))

                        elif use_SE and use_A:
                            restarts_SE_adapt.append(np.sum(restarts_sorted))

        accuracy_check(dt_list, problem.__name__, sweeper.__name__, V_ref)

        differences_around_switch(
            dt_list,
            problem.__name__,
            restarts_SE,
            restarts_adapt,
            restarts_SE_adapt,
            sweeper.__name__,
            V_ref,
        )

        differences_over_time(dt_list, problem.__name__, sweeper.__name__, V_ref)

        iterations_over_time(dt_list, description['step_params']['maxiter'], problem.__name__, sweeper.__name__)

        restarts_SE = []
        restarts_adapt = []
        restarts_SE_adapt = []


def accuracy_check(dt_list, problem, sweeper, V_ref, cwd='./'):
    """
    Routine to check accuracy for different step sizes in case of using adaptivity

    Args:
        dt_list (list): list of considered (initial) step sizes
        problem (pySDC.core.Problem.ptype): Problem class used to consider (the class name)
        sweeper (pySDC.core.Sweeper.sweeper): Sweeper used to solve (the class name)
        V_ref (np.float): reference value for the switch
        cwd (str): current working directory
    """

    if len(dt_list) > 1:
        setup_mpl()
        fig_acc, ax_acc = plt_helper.plt.subplots(
            1, len(dt_list), figsize=(3 * len(dt_list), 3), sharex='col', sharey='row'
        )

    else:
        setup_mpl()
        fig_acc, ax_acc = plt_helper.plt.subplots(1, 1, figsize=(3, 3), sharex='col', sharey='row')

    count_ax = 0
    for dt_item in dt_list:
        f3 = open(cwd + 'data/battery_dt{}_USETrue_USATrue_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_SE_adapt = dill.load(f3)
        f3.close()

        f4 = open(cwd + 'data/battery_dt{}_USEFalse_USATrue_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_adapt = dill.load(f4)
        f4.close()

        switches_SE_adapt = get_recomputed(stats_SE_adapt, type='switch', sortby='time')
        t_switch_SE_adapt = [v[1] for v in switches_SE_adapt]
        t_switch_SE_adapt = t_switch_SE_adapt[-1]

        dt_SE_adapt_val = get_sorted(stats_SE_adapt, type='dt', recomputed=False)
        dt_adapt_val = get_sorted(stats_adapt, type='dt', recomputed=False)

        e_emb_SE_adapt_val = get_sorted(stats_SE_adapt, type='e_embedded', recomputed=False)
        e_emb_adapt_val = get_sorted(stats_adapt, type='e_embedded', recomputed=False)

        times_SE_adapt = [v[0] for v in e_emb_SE_adapt_val]
        times_adapt = [v[0] for v in e_emb_adapt_val]

        e_emb_SE_adapt = [v[1] for v in e_emb_SE_adapt_val]
        e_emb_adapt = [v[1] for v in e_emb_adapt_val]

        if len(dt_list) > 1:
            ax_acc[count_ax].set_title(r'$\Delta t_\mathrm{initial}$=%s' % dt_item)
            dt1 = ax_acc[count_ax].plot(
                [v[0] for v in dt_SE_adapt_val],
                [v[1] for v in dt_SE_adapt_val],
                'ko-',
                label=r'SE+A - $\Delta t_\mathrm{adapt}$',
            )
            dt2 = ax_acc[count_ax].plot(
                [v[0] for v in dt_adapt_val], [v[1] for v in dt_adapt_val], 'g-', label=r'A - $\Delta t_\mathrm{adapt}$'
            )
            ax_acc[count_ax].axvline(x=t_switch_SE_adapt, linestyle='--', linewidth=0.5, color='r', label='Switch')
            ax_acc[count_ax].tick_params(axis='both', which='major', labelsize=6)
            ax_acc[count_ax].set_xlabel('Time', fontsize=6)
            if count_ax == 0:
                ax_acc[count_ax].set_ylabel(r'$\Delta t_\mathrm{adapt}$', fontsize=6)

            e_ax = ax_acc[count_ax].twinx()
            e_plt1 = e_ax.plot(times_SE_adapt, e_emb_SE_adapt, 'k--', label=r'SE+A - $\epsilon_{emb}$')
            e_plt2 = e_ax.plot(times_adapt, e_emb_adapt, 'g--', label=r'A - $\epsilon_{emb}$')
            e_ax.set_yscale('log', base=10)
            e_ax.set_ylim(1e-16, 1e-7)
            e_ax.tick_params(labelsize=6)

            lines = dt1 + e_plt1 + dt2 + e_plt2
            labels = [l.get_label() for l in lines]

            ax_acc[count_ax].legend(lines, labels, frameon=False, fontsize=6, loc='upper right')

        else:
            ax_acc.set_title(r'$\Delta t_\mathrm{initial}$=%s' % dt_item)
            dt1 = ax_acc.plot(
                [v[0] for v in dt_SE_adapt_val],
                [v[1] for v in dt_SE_adapt_val],
                'ko-',
                label=r'SE+A - $\Delta t_\mathrm{adapt}$',
            )
            dt2 = ax_acc.plot(
                [v[0] for v in dt_adapt_val],
                [v[1] for v in dt_adapt_val],
                'go-',
                label=r'A - $\Delta t_\mathrm{adapt}$',
            )
            ax_acc.axvline(x=t_switch_SE_adapt, linestyle='--', linewidth=0.5, color='r', label='Switch')
            ax_acc.tick_params(axis='both', which='major', labelsize=6)
            ax_acc.set_xlabel('Time', fontsize=6)
            ax_acc.set_ylabel(r'$Delta t_\mathrm{adapt}$', fontsize=6)

            e_ax = ax_acc.twinx()
            e_plt1 = e_ax.plot(times_SE_adapt, e_emb_SE_adapt, 'k--', label=r'SE+A - $\epsilon_{emb}$')
            e_plt2 = e_ax.plot(times_adapt, e_emb_adapt, 'g--', label=r'A - $\epsilon_{emb}$')
            e_ax.set_yscale('log', base=10)
            e_ax.tick_params(labelsize=6)

            lines = dt1 + e_plt1 + dt2 + e_plt2
            labels = [l.get_label() for l in lines]

            ax_acc.legend(lines, labels, frameon=False, fontsize=6, loc='upper right')

        count_ax += 1

    fig_acc.savefig('data/embedded_error_adaptivity_{}.png'.format(sweeper), dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig_acc)


def differences_around_switch(
    dt_list, problem, restarts_SE, restarts_adapt, restarts_SE_adapt, sweeper, V_ref, cwd='./'
):
    """
    Routine to plot the differences before, at, and after the switch. Produces the diffs_estimation_<sweeper_class>.png file

    Args:
        dt_list (list): list of considered (initial) step sizes
        problem (pySDC.core.Problem.ptype): Problem class used to consider (the class name)
        restarts_SE (list): Restarts for the solve only using the switch estimator
        restarts_adapt (list): Restarts for the solve of only using adaptivity
        restarts_SE_adapt (list): Restarts for the solve of using both, switch estimator and adaptivity
        sweeper (pySDC.core.Sweeper.sweeper): Sweeper used to solve (the class name)
        V_ref (np.float): reference value for the switch
        cwd (str): current working directory
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
        stats_SE = dill.load(f1)
        f1.close()

        f2 = open(cwd + 'data/battery_dt{}_USEFalse_USAFalse_{}.dat'.format(dt_item, sweeper), 'rb')
        stats = dill.load(f2)
        f2.close()

        f3 = open(cwd + 'data/battery_dt{}_USETrue_USATrue_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_SE_adapt = dill.load(f3)
        f3.close()

        f4 = open(cwd + 'data/battery_dt{}_USEFalse_USATrue_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_adapt = dill.load(f4)
        f4.close()

        switches_SE = get_recomputed(stats_SE, type='switch', sortby='time')
        t_switch = [v[1] for v in switches_SE]
        t_switch = t_switch[-1]  # battery has only one single switch

        switches_SE_adapt = get_recomputed(stats_SE_adapt, type='switch', sortby='time')
        t_switch_SE_adapt = [v[1] for v in switches_SE_adapt]
        t_switch_SE_adapt = t_switch_SE_adapt[-1]

        vC_SE = [me[1][1] for me in get_sorted(stats_SE, type='u', recomputed=False)]
        vC_adapt = [me[1][1] for me in get_sorted(stats_adapt, type='u', recomputed=False)]
        vC_SE_adapt = [me[1][1] for me in get_sorted(stats_SE_adapt, type='u', recomputed=False)]
        vC = [me[1][1] for me in get_sorted(stats, type='u', recomputed=False)]

        diff_SE, diff = vC_SE - V_ref[0], vC - V_ref[0]
        times_SE = [me[0] for me in get_sorted(stats_SE, type='u', recomputed=False)]
        times = [me[0] for me in get_sorted(stats, type='u', recomputed=False)]

        diff_adapt, diff_SE_adapt = vC_adapt - V_ref[0], vC_SE_adapt - V_ref[0]
        times_adapt = [me[0] for me in get_sorted(stats_adapt, type='u', recomputed=False)]
        times_SE_adapt = [me[0] for me in get_sorted(stats_SE_adapt, type='u', recomputed=False)]

        diffs_true_at.append(
            [diff_SE[m] for m in range(len(times_SE)) if np.isclose(times_SE[m], t_switch, atol=1e-15)][0]
        )

        diffs_false_before.append(
            [diff[m - 1] for m in range(1, len(times)) if times[m - 1] <= t_switch <= times[m]][0]
        )
        diffs_false_after.append([diff[m] for m in range(1, len(times)) if times[m - 1] <= t_switch <= times[m]][0])

        for m in range(len(times_SE_adapt)):
            if np.isclose(times_SE_adapt[m], t_switch_SE_adapt, atol=1e-10):
                diffs_true_at_adapt.append(diff_SE_adapt[m])
                diffs_true_before_adapt.append(diff_SE_adapt[m - 1])
                diffs_true_after_adapt.append(diff_SE_adapt[m + 1])

        diffs_false_before_adapt.append(
            [diff_adapt[m - 1] for m in range(len(times_adapt)) if times_adapt[m - 1] <= t_switch <= times_adapt[m]][0]
        )
        diffs_false_after_adapt.append(
            [diff_adapt[m] for m in range(len(times_adapt)) if times_adapt[m - 1] <= t_switch <= times_adapt[m]][0]
        )

    setup_mpl()
    fig_around, ax_around = plt_helper.plt.subplots(1, 3, figsize=(9, 3), sharex='col', sharey='row')
    ax_around[0].set_title("Using SE")
    pos11 = ax_around[0].plot(dt_list, diffs_false_before, 'rs-', label='before switch')
    pos12 = ax_around[0].plot(dt_list, diffs_false_after, 'bd--', label='after switch')
    pos13 = ax_around[0].plot(dt_list, diffs_true_at, 'ko--', label='at switch')
    ax_around[0].set_xticks(dt_list)
    ax_around[0].set_xticklabels(dt_list)
    ax_around[0].tick_params(axis='both', which='major', labelsize=6)
    ax_around[0].set_xscale('log', base=10)
    ax_around[0].set_yscale('symlog', linthresh=1e-8)
    ax_around[0].set_ylim(-1, 1)
    ax_around[0].set_xlabel(r'$\Delta t_\mathrm{initial}$', fontsize=6)
    ax_around[0].set_ylabel(r'$v_{C}-V_{ref}$', fontsize=6)

    restart_ax0 = ax_around[0].twinx()
    restarts_plt0 = restart_ax0.plot(dt_list, restarts_SE, 'cs--', label='Restarts')
    restart_ax0.tick_params(labelsize=6)

    lines = pos11 + pos12 + pos13 + restarts_plt0
    labels = [l.get_label() for l in lines]
    ax_around[0].legend(lines, labels, frameon=False, fontsize=6, loc='lower right')

    ax_around[1].set_title("Using Adaptivity")
    pos21 = ax_around[1].plot(dt_list, diffs_false_before_adapt, 'rs-', label='before switch')
    pos22 = ax_around[1].plot(dt_list, diffs_false_after_adapt, 'bd--', label='after switch')
    ax_around[1].set_xticks(dt_list)
    ax_around[1].set_xticklabels(dt_list)
    ax_around[1].tick_params(axis='both', which='major', labelsize=6)
    ax_around[1].set_xscale('log', base=10)
    ax_around[1].set_yscale('symlog', linthresh=1e-8)
    ax_around[1].set_ylim(-1, 1)
    ax_around[1].set_xlabel(r'$\Delta t_\mathrm{initial}$', fontsize=6)

    restart_ax1 = ax_around[1].twinx()
    restarts_plt1 = restart_ax1.plot(dt_list, restarts_adapt, 'cs--', label='Restarts')
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
    ax_around[2].tick_params(axis='both', which='major', labelsize=6)
    ax_around[2].set_xscale('log', base=10)
    ax_around[2].set_yscale('symlog', linthresh=1e-8)
    ax_around[2].set_ylim(-1, 1)
    ax_around[2].set_xlabel(r'$\Delta t_\mathrm{initial}$', fontsize=6)

    restart_ax2 = ax_around[2].twinx()
    restarts_plt2 = restart_ax2.plot(dt_list, restarts_SE_adapt, 'cs--', label='Restarts')
    restart_ax2.tick_params(labelsize=6)

    lines = pos31 + pos32 + pos33 + restarts_plt2
    labels = [l.get_label() for l in lines]
    ax_around[2].legend(frameon=False, fontsize=6, loc='lower right')

    fig_around.savefig('data/diffs_around_switch_{}.png'.format(sweeper), dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig_around)


def differences_over_time(dt_list, problem, sweeper, V_ref, cwd='./'):
    """
    Routine to plot the differences in time using the switch estimator or not. Produces the difference_estimation_<sweeper_class>.png file

    Args:
        dt_list (list): list of considered (initial) step sizes
        problem (pySDC.core.Problem.ptype): Problem class used to consider (the class name)
        sweeper (pySDC.core.Sweeper.sweeper): Sweeper used to solve (the class name)
        V_ref (np.float): reference value for the switch
        cwd (str): current working directory
    """

    if len(dt_list) > 1:
        setup_mpl()
        fig_diffs, ax_diffs = plt_helper.plt.subplots(
            2, len(dt_list), figsize=(4 * len(dt_list), 6), sharex='col', sharey='row'
        )

    else:
        setup_mpl()
        fig_diffs, ax_diffs = plt_helper.plt.subplots(2, 1, figsize=(4, 6))

    count_ax = 0
    for dt_item in dt_list:
        f1 = open(cwd + 'data/battery_dt{}_USETrue_USAFalse_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_SE = dill.load(f1)
        f1.close()

        f2 = open(cwd + 'data/battery_dt{}_USEFalse_USAFalse_{}.dat'.format(dt_item, sweeper), 'rb')
        stats = dill.load(f2)
        f2.close()

        f3 = open(cwd + 'data/battery_dt{}_USETrue_USATrue_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_SE_adapt = dill.load(f3)
        f3.close()

        f4 = open(cwd + 'data/battery_dt{}_USEFalse_USATrue_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_adapt = dill.load(f4)
        f4.close()

        switches_SE = get_recomputed(stats_SE, type='switch', sortby='time')
        t_switch_SE = [v[1] for v in switches_SE]
        t_switch_SE = t_switch_SE[-1]  # battery has only one single switch

        switches_SE_adapt = get_recomputed(stats_SE_adapt, type='switch', sortby='time')
        t_switch_SE_adapt = [v[1] for v in switches_SE_adapt]
        t_switch_SE_adapt = t_switch_SE_adapt[-1]

        dt_adapt = np.array(get_sorted(stats_adapt, type='dt', recomputed=False))
        dt_SE_adapt = np.array(get_sorted(stats_SE_adapt, type='dt', recomputed=False))

        restart_adapt = np.array(get_sorted(stats_adapt, type='restart', recomputed=None))
        restart_SE_adapt = np.array(get_sorted(stats_SE_adapt, type='restart', recomputed=None))

        vC_SE = [me[1][1] for me in get_sorted(stats_SE, type='u', recomputed=False)]
        vC_adapt = [me[1][1] for me in get_sorted(stats_adapt, type='u', recomputed=False)]
        vC_SE_adapt = [me[1][1] for me in get_sorted(stats_SE_adapt, type='u', recomputed=False)]
        vC = [me[1][1] for me in get_sorted(stats, type='u', recomputed=False)]

        diff_SE, diff = vC_SE - V_ref[0], vC - V_ref[0]
        times_SE = [me[0] for me in get_sorted(stats_SE, type='u', recomputed=False)]
        times = [me[0] for me in get_sorted(stats, type='u', recomputed=False)]

        diff_adapt, diff_SE_adapt = vC_adapt - V_ref[0], vC_SE_adapt - V_ref[0]
        times_adapt = [me[0] for me in get_sorted(stats_adapt, type='u', recomputed=False)]
        times_SE_adapt = [me[0] for me in get_sorted(stats_SE_adapt, type='u', recomputed=False)]

        if len(dt_list) > 1:
            ax_diffs[0, count_ax].set_title(r'$\Delta t$=%s' % dt_item)
            ax_diffs[0, count_ax].plot(times_SE, diff_SE, label='SE=True, A=False', color='#ff7f0e')
            ax_diffs[0, count_ax].plot(times, diff, label='SE=False, A=False', color='#1f77b4')
            ax_diffs[0, count_ax].plot(times_adapt, diff_adapt, label='SE=False, A=True', color='red', linestyle='--')
            ax_diffs[0, count_ax].plot(
                times_SE_adapt, diff_SE_adapt, label='SE=True, A=True', color='limegreen', linestyle='-.'
            )
            ax_diffs[0, count_ax].axvline(x=t_switch_SE, linestyle='--', linewidth=0.5, color='k', label='Switch')
            ax_diffs[0, count_ax].legend(frameon=False, fontsize=6, loc='lower left')
            ax_diffs[0, count_ax].set_yscale('symlog', linthresh=1e-5)
            ax_diffs[0, count_ax].tick_params(axis='both', which='major', labelsize=6)
            if count_ax == 0:
                ax_diffs[0, count_ax].set_ylabel('Difference $v_{C}-V_{ref}$', fontsize=6)

            if count_ax == 0 or count_ax == 1:
                ax_diffs[0, count_ax].legend(frameon=False, fontsize=6, loc='upper right')

            else:
                ax_diffs[0, count_ax].legend(frameon=False, fontsize=6, loc='upper right')

            ax_diffs[1, count_ax].plot(
                dt_adapt[:, 0], dt_adapt[:, 1], label=r'$\Delta t$ - SE=F, A=T', color='red', linestyle='--'
            )
            ax_diffs[1, count_ax].plot([None], [None], label='Restart - SE=F, A=T', color='grey', linestyle='-.')

            for i in range(len(restart_adapt)):
                if restart_adapt[i, 1] > 0:
                    ax_diffs[1, count_ax].axvline(restart_adapt[i, 0], color='grey', linestyle='-.')

            ax_diffs[1, count_ax].plot(
                dt_SE_adapt[:, 0],
                dt_SE_adapt[:, 1],
                label=r'$ \Delta t$ - SE=T, A=T',
                color='limegreen',
                linestyle='-.',
            )
            ax_diffs[1, count_ax].plot([None], [None], label='Restart - SE=T, A=T', color='black', linestyle='-.')

            for i in range(len(restart_SE_adapt)):
                if restart_SE_adapt[i, 1] > 0:
                    ax_diffs[1, count_ax].axvline(restart_SE_adapt[i, 0], color='black', linestyle='-.')

            ax_diffs[1, count_ax].set_xlabel('Time', fontsize=6)
            ax_diffs[1, count_ax].tick_params(axis='both', which='major', labelsize=6)
            if count_ax == 0:
                ax_diffs[1, count_ax].set_ylabel(r'$\Delta t_\mathrm{adapted}$', fontsize=6)

            ax_diffs[1, count_ax].set_yscale('log', base=10)
            ax_diffs[1, count_ax].legend(frameon=True, fontsize=6, loc='lower left')

        else:
            ax_diffs[0].set_title(r'$\Delta t$=%s' % dt_item)
            ax_diffs[0].plot(times_SE, diff_SE, label='SE=True', color='#ff7f0e')
            ax_diffs[0].plot(times, diff, label='SE=False', color='#1f77b4')
            ax_diffs[0].plot(times_adapt, diff_adapt, label='SE=False, A=True', color='red', linestyle='--')
            ax_diffs[0].plot(times_SE_adapt, diff_SE_adapt, label='SE=True, A=True', color='limegreen', linestyle='-.')
            ax_diffs[0].axvline(x=t_switch_SE, linestyle='--', linewidth=0.5, color='k', label='Switch')
            ax_diffs[0].tick_params(axis='both', which='major', labelsize=6)
            ax_diffs[0].set_yscale('symlog', linthresh=1e-5)
            ax_diffs[0].set_ylabel('Difference $v_{C}-V_{ref}$', fontsize=6)
            ax_diffs[0].legend(frameon=False, fontsize=6, loc='center right')

            ax_diffs[1].plot(dt_adapt[:, 0], dt_adapt[:, 1], label='SE=False, A=True', color='red', linestyle='--')
            ax_diffs[1].plot(
                dt_SE_adapt[:, 0], dt_SE_adapt[:, 1], label='SE=True, A=True', color='limegreen', linestyle='-.'
            )
            ax_diffs[1].tick_params(axis='both', which='major', labelsize=6)
            ax_diffs[1].set_xlabel('Time', fontsize=6)
            ax_diffs[1].set_ylabel(r'$\Delta t_\mathrm{adapted}$', fontsize=6)
            ax_diffs[1].set_yscale('log', base=10)

            ax_diffs[1].legend(frameon=False, fontsize=6, loc='upper right')

        count_ax += 1

    plt_helper.plt.tight_layout()
    fig_diffs.savefig('data/diffs_over_time_{}.png'.format(sweeper), dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig_diffs)


def iterations_over_time(dt_list, maxiter, problem, sweeper, cwd='./'):
    """
    Routine  to plot the number of iterations over time using switch estimator or not. Produces the iters_<sweeper_class>.png file

    Args:
        dt_list (list): list of considered (initial) step sizes
        maxiter (np.int): maximum number of iterations
        problem (pySDC.core.Problem.ptype): Problem class used to consider (the class name)
        sweeper (pySDC.core.Sweeper.sweeper): Sweeper used to solve (the class name)
        cwd (str): current working directory
    """

    iters_time_SE = []
    iters_time = []
    iters_time_SE_adapt = []
    iters_time_adapt = []
    times_SE = []
    times = []
    times_SE_adapt = []
    times_adapt = []
    t_switches_SE = []
    t_switches_SE_adapt = []

    for dt_item in dt_list:
        f1 = open(cwd + 'data/battery_dt{}_USETrue_USAFalse_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_SE = dill.load(f1)
        f1.close()

        f2 = open(cwd + 'data/battery_dt{}_USEFalse_USAFalse_{}.dat'.format(dt_item, sweeper), 'rb')
        stats = dill.load(f2)
        f2.close()

        f3 = open(cwd + 'data/battery_dt{}_USETrue_USATrue_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_SE_adapt = dill.load(f3)
        f3.close()

        f4 = open(cwd + 'data/battery_dt{}_USEFalse_USATrue_{}.dat'.format(dt_item, sweeper), 'rb')
        stats_adapt = dill.load(f4)
        f4.close()

        # consider iterations before restarts to see what happens
        iter_counts_SE_val = get_sorted(stats_SE, type='niter')
        iter_counts_SE_adapt_val = get_sorted(stats_SE_adapt, type='niter')
        iter_counts_adapt_val = get_sorted(stats_adapt, type='niter')
        iter_counts_val = get_sorted(stats, type='niter')

        iters_time_SE.append([v[1] for v in iter_counts_SE_val])
        iters_time_SE_adapt.append([v[1] for v in iter_counts_SE_adapt_val])
        iters_time_adapt.append([v[1] for v in iter_counts_adapt_val])
        iters_time.append([v[1] for v in iter_counts_val])

        times_SE.append([v[0] for v in iter_counts_SE_val])
        times_SE_adapt.append([v[0] for v in iter_counts_SE_adapt_val])
        times_adapt.append([v[0] for v in iter_counts_adapt_val])
        times.append([v[0] for v in iter_counts_val])

        switches_SE = get_recomputed(stats_SE, type='switch', sortby='time')
        t_switch_SE = [v[1] for v in switches_SE]
        t_switches_SE.append(t_switch_SE[-1])

        switches_SE_adapt = get_recomputed(stats_SE_adapt, type='switch', sortby='time')
        t_switch_SE_adapt = [v[1] for v in switches_SE_adapt]
        t_switches_SE_adapt.append(t_switch_SE_adapt[-1])

    if len(dt_list) > 1:
        setup_mpl()
        fig_iter_all, ax_iter_all = plt_helper.plt.subplots(
            nrows=1, ncols=len(dt_list), figsize=(2 * len(dt_list) - 1, 3), sharex='col', sharey='row'
        )
        for col in range(len(dt_list)):
            ax_iter_all[col].plot(times[col], iters_time[col], label='SE=F, A=F')
            ax_iter_all[col].plot(times_SE[col], iters_time_SE[col], label='SE=T, A=F')
            ax_iter_all[col].plot(times_SE_adapt[col], iters_time_SE_adapt[col], '--', label='SE=T, A=T')
            ax_iter_all[col].plot(times_adapt[col], iters_time_adapt[col], '--', label='SE=F, A=T')
            ax_iter_all[col].axvline(x=t_switches_SE[col], linestyle='--', linewidth=0.5, color='k', label='Switch')
            ax_iter_all[col].set_title(r'$\Delta t_\mathrm{initial}$=%s' % dt_list[col])
            ax_iter_all[col].set_ylim(0, maxiter + 2)
            ax_iter_all[col].set_xlabel('Time', fontsize=6)
            ax_iter_all[col].tick_params(axis='both', which='major', labelsize=6)

            if col == 0:
                ax_iter_all[col].set_ylabel('Number iterations', fontsize=6)

            ax_iter_all[col].legend(frameon=False, fontsize=6, loc='upper right')
    else:
        setup_mpl()
        fig_iter_all, ax_iter_all = plt_helper.plt.subplots(nrows=1, ncols=1, figsize=(3, 3))

        ax_iter_all.plot(times[0], iters_time[0], label='SE=False')
        ax_iter_all.plot(times_SE[0], iters_time_SE[0], label='SE=True')
        ax_iter_all.plot(times_SE_adapt[0], iters_time_SE_adapt[0], '--', label='SE=T, A=T')
        ax_iter_all.plot(times_adapt[0], iters_time_adapt[0], '--', label='SE=F, A=T')
        ax_iter_all.axvline(x=t_switches_SE[0], linestyle='--', linewidth=0.5, color='k', label='Switch')
        ax_iter_all.set_title(r'$\Delta t_\mathrm{initial}$=%s' % dt_list[0])
        ax_iter_all.set_ylim(0, maxiter + 2)
        ax_iter_all.set_xlabel('Time', fontsize=6)
        ax_iter_all.tick_params(axis='both', which='major', labelsize=6)

        ax_iter_all.set_ylabel('Number iterations', fontsize=6)
        ax_iter_all.legend(frameon=False, fontsize=6, loc='upper right')

    plt_helper.plt.tight_layout()
    fig_iter_all.savefig('data/iters_{}.png'.format(sweeper), dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig_iter_all)


if __name__ == "__main__":
    run()
