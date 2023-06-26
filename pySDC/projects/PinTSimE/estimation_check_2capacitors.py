import numpy as np
import dill
from pathlib import Path

from pySDC.helpers.stats_helper import get_sorted
from pySDC.core.Collocation import CollBase as Collocation
from pySDC.implementations.problem_classes.Battery import battery_n_capacitors
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.projects.PinTSimE.battery_model import controller_run, generate_description, get_recomputed, log_data
from pySDC.projects.PinTSimE.piline_model import setup_mpl
from pySDC.projects.PinTSimE.battery_2capacitors_model import (
    check_solution,
    proof_assertions_description,
    proof_assertions_time,
)
import pySDC.helpers.plot_helper as plt_helper

from pySDC.projects.PinTSimE.switch_estimator import SwitchEstimator


def run(cwd='./'):
    """
    Routine to check the differences between using a switch estimator or not

    Args:
        cwd (str): current working directory
    """

    dt_list = [4e-1, 4e-2, 4e-3]
    t0 = 0.0
    Tend = 3.5

    problem_classes = [battery_n_capacitors]
    sweeper_classes = [imex_1st_order]

    ncapacitors = 2
    alpha = 5.0
    V_ref = np.array([1.0, 1.0])
    C = np.array([1.0, 1.0])

    use_switch_estimator = [True, False]
    restarts_all = []
    restarts_dict = dict()
    for problem, sweeper in zip(problem_classes, sweeper_classes):
        for dt_item in dt_list:
            for use_SE in use_switch_estimator:
                description, controller_params = generate_description(
                    dt_item,
                    problem,
                    sweeper,
                    log_data,
                    False,
                    use_SE,
                    ncapacitors,
                    alpha,
                    V_ref,
                    C,
                )

                # Assertions
                proof_assertions_description(description, False, use_SE)

                proof_assertions_time(dt_item, Tend, V_ref, alpha)

                stats = controller_run(description, controller_params, False, use_SE, t0, Tend)

                if use_SE:
                    switches = get_recomputed(stats, type='switch', sortby='time')
                    assert len(switches) >= 2, f"Expected at least 2 switches for dt: {dt_item}, got {len(switches)}!"

                    check_solution(stats, dt_item, use_SE)

                fname = 'data/{}_dt{}_USE{}.dat'.format(problem.__name__, dt_item, use_SE)
                f = open(fname, 'wb')
                dill.dump(stats, f)
                f.close()

                if use_SE:
                    restarts_dict[dt_item] = np.array(get_sorted(stats, type='restart', recomputed=None))
                    restarts = restarts_dict[dt_item][:, 1]
                    restarts_all.append(np.sum(restarts))
                    print(f"Restarts for dt: {dt_item:.2e} -- {np.sum(restarts):.0f}")

    V_ref = description['problem_params']['V_ref']

    val_switch_all = []
    diff_true_all1 = []
    diff_false_all_before1 = []
    diff_false_all_after1 = []
    diff_true_all2 = []
    diff_false_all_before2 = []
    diff_false_all_after2 = []
    restarts_dt_switch1 = []
    restarts_dt_switch2 = []
    for dt_item in dt_list:
        f1 = open(cwd + 'data/{}_dt{}_USETrue.dat'.format(problem.__name__, dt_item), 'rb')
        stats_true = dill.load(f1)
        f1.close()

        f2 = open(cwd + 'data/{}_dt{}_USEFalse.dat'.format(problem.__name__, dt_item), 'rb')
        stats_false = dill.load(f2)
        f2.close()

        switches = get_recomputed(stats_true, type='switch', sortby='time')
        t_switch = [v[1] for v in switches]

        val_switch_all.append([t_switch[0], t_switch[1]])

        vC1_true = [me[1][1] for me in get_sorted(stats_true, type='u', recomputed=False)]
        vC2_true = [me[1][2] for me in get_sorted(stats_true, type='u', recomputed=False)]
        vC1_false = [me[1][1] for me in get_sorted(stats_false, type='u', recomputed=False)]
        vC2_false = [me[1][2] for me in get_sorted(stats_false, type='u', recomputed=False)]

        diff_true1 = vC1_true - V_ref[0]
        diff_true2 = vC2_true - V_ref[1]
        diff_false1 = vC1_false - V_ref[0]
        diff_false2 = vC2_false - V_ref[1]

        t_true = [me[0] for me in get_sorted(stats_true, type='u', recomputed=False)]
        t_false = [me[0] for me in get_sorted(stats_false, type='u', recomputed=False)]

        diff_true_all1.append(
            [diff_true1[m] for m in range(len(t_true)) if np.isclose(t_true[m], t_switch[0], atol=1e-15)]
        )
        diff_true_all2.append([diff_true2[np.argmin([abs(me - t_switch[1]) for me in t_true])]])

        diff_false_all_before1.append(
            [diff_false1[m - 1] for m in range(1, len(t_false)) if t_false[m - 1] < t_switch[0] < t_false[m]]
        )
        diff_false_all_after1.append(
            [diff_false1[m] for m in range(1, len(t_false)) if t_false[m - 1] < t_switch[0] < t_false[m]]
        )

        diff_false_all_before2.append(
            [diff_false2[m - 1] for m in range(1, len(t_false)) if t_false[m - 1] < t_switch[1] < t_false[m]]
        )
        diff_false_all_after2.append(
            [diff_false2[m] for m in range(1, len(t_false)) if t_false[m - 1] < t_switch[1] < t_false[m]]
        )

        restarts_dt = restarts_dict[dt_item]
        for i in range(len(restarts_dt[:, 0])):
            if np.isclose(restarts_dt[i, 0], t_switch[0], atol=1e-15):
                restarts_dt_switch1.append(np.sum(restarts_dt[0 : i - 1, 1]))
                break

        for i in range(len(restarts_dt[:, 0])):
            if np.isclose(restarts_dt[i, 0], t_switch[1], atol=1e-15):
                restarts_dt_switch2.append(np.sum(restarts_dt[i - 2 :, 1]))
                break

        setup_mpl()
        fig1, ax1 = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
        ax1.set_title('Time evolution of $v_{C_{1}}-V_{ref1}$')
        ax1.plot(t_true, diff_true1, label='SE=True', color='#ff7f0e')
        ax1.plot(t_false, diff_false1, label='SE=False', color='#1f77b4')
        ax1.axvline(x=t_switch[0], linestyle='--', color='k', label='Switch1')
        ax1.legend(frameon=False, fontsize=10, loc='lower left')
        ax1.set_yscale('symlog', linthresh=1e-5)
        ax1.set_xlabel('Time')

        fig1.savefig('data/difference_estimation_vC1_dt{}.png'.format(dt_item), dpi=300, bbox_inches='tight')
        plt_helper.plt.close(fig1)

        setup_mpl()
        fig2, ax2 = plt_helper.plt.subplots(1, 1, figsize=(4.5, 3))
        ax2.set_title('Time evolution of $v_{C_{2}}-V_{ref2}$')
        ax2.plot(t_true, diff_true2, label='SE=True', color='#ff7f0e')
        ax2.plot(t_false, diff_false2, label='SE=False', color='#1f77b4')
        ax2.axvline(x=t_switch[1], linestyle='--', color='k', label='Switch2')
        ax2.legend(frameon=False, fontsize=10, loc='lower left')
        ax2.set_yscale('symlog', linthresh=1e-5)
        ax2.set_xlabel('Time')

        fig2.savefig('data/difference_estimation_vC2_dt{}.png'.format(dt_item), dpi=300, bbox_inches='tight')
        plt_helper.plt.close(fig2)

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
    ax1.set_xlabel(r'$\Delta t$')

    restart_ax = ax1.twinx()
    restarts = restart_ax.plot(dt_list, restarts_dt_switch1, 'cs--', label='Restarts')
    restart_ax.set_ylabel('Restarts')

    lines = pos1 + pos2 + pos3 + restarts
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, frameon=False, fontsize=8, loc='center right')

    fig1.savefig('data/diffs_estimation_vC1.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig1)

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
    ax2.set_xlabel(r'$\Delta t$')

    restart_ax = ax2.twinx()
    restarts = restart_ax.plot(dt_list, restarts_dt_switch2, 'cs--', label='Restarts')
    restart_ax.set_ylabel('Restarts')

    lines = pos1 + pos2 + pos3 + restarts
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, frameon=False, fontsize=8, loc='center right')

    fig2.savefig('data/diffs_estimation_vC2.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig2)


if __name__ == "__main__":
    run()
