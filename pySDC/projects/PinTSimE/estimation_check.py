import numpy as np
from pathlib import Path

from pySDC.helpers.stats_helper import sort_stats, filter_stats, get_sorted
from pySDC.implementations.problem_classes.Battery import battery_n_capacitors, battery, battery_implicit
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit

from pySDC.projects.PinTSimE.battery_model import runSimulation, plotStylingStuff

import pySDC.helpers.plot_helper as plt_helper

from pySDC.projects.PinTSimE.battery_model import LogEventBattery
from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.implementations.hooks.log_step_size import LogStepSize
from pySDC.implementations.hooks.log_embedded_error_estimate import LogEmbeddedErrorEstimate


def run_estimation_check():
    r"""
    Generates plots to visualise results applying the Switch Estimator and Adaptivity to the battery models
    containing.

    Note
    ----
    Hardcoded solutions for battery models in `pySDC.projects.PinTSimE.hardcoded_solutions` are only computed for
    ``dt_list=[1e-2, 1e-3]`` and ``M_fix=4``. Hence changing ``dt_list`` and ``M_fix`` to different values could arise
    an ``AssertionError``.
    """

    Path("data").mkdir(parents=True, exist_ok=True)

    # --- defines parameters for sweeper ----
    M_fix = 4
    sweeper_params = {
        'num_nodes': M_fix,
        'quad_type': 'LOBATTO',
        'QI': 'IE',
    }

    # --- defines parameters for event detection and maximum number of iterations ----
    handling_params = {
        'restol': -1,
        'maxiter': 8,
        'max_restarts': 50,
        'recomputed': False,
        'tol_event': 1e-10,
        'alpha': 1.0,
        'exact_event_time_avail': None,
    }

    problem_classes = [battery, battery_implicit, battery_n_capacitors]
    prob_class_names = [cls.__name__ for cls in problem_classes]
    sweeper_classes = [imex_1st_order, generic_implicit, imex_1st_order]

    # --- defines parameters for battery models ----
    params_battery_1capacitor = {
        'ncapacitors': 1,
        'C': np.array([1.0]),
        'alpha': 1.2,
        'V_ref': np.array([1.0]),
    }

    params_battery_2capacitors = {
        'ncapacitors': 2,
        'C': np.array([1.0, 1.0]),
        'alpha': 1.2,
        'V_ref': np.array([1.0, 1.0]),
    }

    # --- parameters for each problem class are stored in this dictionary ----
    all_params = {
        'battery': {
            'sweeper_params': sweeper_params,
            'handling_params': handling_params,
            'problem_params': params_battery_1capacitor,
        },
        'battery_implicit': {
            'sweeper_params': sweeper_params,
            'handling_params': handling_params,
            'problem_params': params_battery_1capacitor,
        },
        'battery_n_capacitors': {
            'sweeper_params': sweeper_params,
            'handling_params': handling_params,
            'problem_params': params_battery_2capacitors,
        },
    }

    # ---- simulation domain for each problem class ----
    interval = {
        'battery': (0.0, 0.3),
        'battery_implicit': (0.0, 0.3),
        'battery_n_capacitors': (0.0, 0.5),
    }

    hook_class = [LogSolution, LogEventBattery, LogEmbeddedErrorEstimate, LogStepSize]

    use_detection = [True, False]
    use_adaptivity = [True, False]

    for problem, sweeper, prob_cls_name in zip(problem_classes, sweeper_classes, prob_class_names):
        u_num = runSimulation(
            problem=problem,
            sweeper=sweeper,
            all_params=all_params[prob_cls_name],
            use_adaptivity=use_adaptivity,
            use_detection=use_detection,
            hook_class=hook_class,
            interval=interval[prob_cls_name],
            dt_list=[1e-2, 1e-3],
            nnodes=[M_fix],
        )

        plotAccuracyCheck(u_num, prob_cls_name, M_fix)

        plotStateFunctionAroundEvent(u_num, prob_cls_name, M_fix)

        plotStateFunctionOverTime(u_num, prob_cls_name, M_fix)


def plotAccuracyCheck(u_num, prob_cls_name, M_fix):  # pragma: no cover
    r"""
    Routine to check accuracy for different step sizes in case of using adaptivity.

    Parameters
    ----------
    u_num : dict
        Contains the all the data. Dictionary has the structure ``u_num[dt][M][use_SE][use_A]``,
        where for each step size ``dt``, for each number of collocation node ``M``, for each
        combination of event detection ``use_SE`` and adaptivity ``use_A`` appropriate stuff is stored.
        For more details, see ``pySDC.projects.PinTSimE.battery_model.getDataDict``.
    prob_cls_name : str
        Name of the problem class.
    M_fix : int
        Fixed number of collocation nodes the plot is generated for.
    """

    colors = plotStylingStuff()
    dt_list = u_num.keys()

    use_A = True
    for dt in dt_list:
        fig, ax = plt_helper.plt.subplots(1, 1, figsize=(7.5, 5), sharex='col', sharey='row')
        e_ax = ax.twinx()
        for use_SE in u_num[dt][M_fix].keys():
            dt_val = u_num[dt][M_fix][use_SE][use_A]['dt']
            e_em_val = u_num[dt][M_fix][use_SE][use_A]['e_em']
            if use_SE:
                t_switches = u_num[dt][M_fix][use_SE][use_A]['t_switches']

                for i in range(len(t_switches)):
                    ax.axvline(x=t_switches[i], linestyle='--', color='tomato', label='Event {}'.format(i + 1))

            ax.plot(dt_val[:, 0], dt_val[:, 1], color=colors[use_SE][use_A], label=r'SE={}, A={}'.format(use_SE, use_A))

            e_ax.plot(e_em_val[:, 0], e_em_val[:, 1], linestyle='dashdot', color=colors[use_SE][use_A])

        ax.plot(0, 0, color='black', linestyle='solid', label=r'$\Delta t_\mathrm{adapt}$')
        ax.plot(0, 0, color='black', linestyle='dashdot', label=r'$e_{em}$')

        e_ax.set_yscale('log', base=10)
        e_ax.set_ylabel(r'Embedded error estimate $e_{em}$', fontsize=16)
        e_ax.set_ylim(1e-16, 1e-7)
        e_ax.tick_params(labelsize=16)
        e_ax.minorticks_off()

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_ylim(1e-9, 1e0)
        ax.set_yscale('log', base=10)
        ax.set_xlabel(r'Time $t$', fontsize=16)
        ax.set_ylabel(r'Adapted step sizes $\Delta t_\mathrm{adapt}$', fontsize=16)
        ax.grid(visible=True)
        ax.minorticks_off()
        ax.legend(frameon=True, fontsize=12, loc='center left')

        fig.savefig(
            'data/detection_and_adaptivity_{}_dt={}_M={}.png'.format(prob_cls_name, dt, M_fix),
            dpi=300,
            bbox_inches='tight',
        )
        plt_helper.plt.close(fig)


def plotStateFunctionAroundEvent(u_num, prob_cls_name, M_fix):  # pragma: no cover
    r"""
    Routine that plots the state function at time before the event, exactly at the event, and after the event. Note
    that this routine does make sense only for a state function that remains constant after the event.

    Parameters
    ----------
    u_num : dict
        Contains the all the data. Dictionary has the structure ``u_num[dt][M][use_SE][use_A]``,
        where for each step size ``dt``, for each number of collocation node ``M``, for each
        combination of event detection ``use_SE`` and adaptivity ``use_A`` appropriate stuff is stored.
        For more details, see ``pySDC.projects.PinTSimE.battery_model.getDataDict``.
    prob_cls_name : str
        Name of the problem class.
    M_fix : int
        Fixed number of collocation nodes the plot is generated for.
    """

    title_cases = {
        0: 'Using detection',
        1: 'Using adaptivity',
        2: 'Using adaptivity and detection',
    }

    dt_list = list(u_num.keys())
    use_detection = u_num[list(dt_list)[0]][M_fix].keys()
    use_adaptivity = u_num[list(dt_list)[0]][M_fix][list(use_detection)[0]].keys()
    h0 = u_num[list(dt_list)[0]][M_fix][list(use_detection)[0]][list(use_adaptivity)[0]]['state_function']
    n = h0[0].shape[0]

    for i in range(n):
        fig, ax = plt_helper.plt.subplots(1, 3, figsize=(12, 4), sharex='col', sharey='row', squeeze=False)
        dt_list = list(u_num.keys())
        for use_SE in use_detection:
            for use_A in use_adaptivity:
                # ---- decide whether state function (w/o handling) has two entries or one; choose correct one with reshaping ----
                h_val_no_handling = [u_num[dt][M_fix][False][False]['state_function'] for dt in dt_list]
                h_no_handling = [item[:] if n == 1 else item[:, i] for item in h_val_no_handling]
                h_no_handling = [item.reshape((item.shape[0],)) for item in h_no_handling]

                t_no_handling = [u_num[dt][M_fix][False][False]['t'] for dt in dt_list]

                if not use_A and not use_SE:
                    continue
                else:
                    ind = 0 if (not use_A and use_SE) else (1 if (use_A and not use_SE) else 2)
                    ax[0, ind].set_title(r'{} for $n={}$'.format(title_cases[ind], i + 1))

                    # ---- same is done here for state function of other cases ----
                    h_val = [u_num[dt][M_fix][use_SE][use_A]['state_function'] for dt in dt_list]
                    h = [item[:] if n == 1 else item[:, i] for item in h_val]
                    h = [item.reshape((item.shape[0],)) for item in h]

                    t = [u_num[dt][M_fix][use_SE][use_A]['t'] for dt in dt_list]

                    if use_SE:
                        t_switches = [u_num[dt][M_fix][use_SE][use_A]['t_switches'] for dt in dt_list]
                        t_switch = [t_event[i] for t_event in t_switches]

                        ax[0, ind].plot(
                            dt_list,
                            [
                                h_item[m]
                                for (t_item, h_item, t_switch_item) in zip(t, h, t_switch)
                                for m in range(len(t_item))
                                if abs(t_item[m] - t_switch_item) <= 1e-14
                            ],
                            color='limegreen',
                            marker='s',
                            linestyle='solid',
                            alpha=0.4,
                            label='At event',
                        )

                        ax[0, ind].plot(
                            dt_list,
                            [
                                h_item[m - 1]
                                for (t_item, h_item, t_switch_item) in zip(t_no_handling, h_no_handling, t_switch)
                                for m in range(1, len(t_item))
                                if t_item[m - 1] < t_switch_item < t_item[m]
                            ],
                            color='firebrick',
                            marker='o',
                            linestyle='solid',
                            alpha=0.4,
                            label='Before event',
                        )

                        ax[0, ind].plot(
                            dt_list,
                            [
                                h_item[m]
                                for (t_item, h_item, t_switch_item) in zip(t_no_handling, h_no_handling, t_switch)
                                for m in range(1, len(t_item))
                                if t_item[m - 1] < t_switch_item < t_item[m]
                            ],
                            color='deepskyblue',
                            marker='*',
                            linestyle='solid',
                            alpha=0.4,
                            label='After event',
                        )

                    else:
                        ax[0, ind].plot(
                            dt_list,
                            [
                                h_item[m - 1]
                                for (t_item, h_item, t_switch_item) in zip(t, h, t_switch)
                                for m in range(1, len(t_item))
                                if t_item[m - 1] < t_switch_item < t_item[m]
                            ],
                            color='firebrick',
                            marker='o',
                            linestyle='solid',
                            alpha=0.4,
                            label='Before event',
                        )

                        ax[0, ind].plot(
                            dt_list,
                            [
                                h_item[m]
                                for (t_item, h_item, t_switch_item) in zip(t, h, t_switch)
                                for m in range(1, len(t_item))
                                if t_item[m - 1] < t_switch_item < t_item[m]
                            ],
                            color='deepskyblue',
                            marker='*',
                            linestyle='solid',
                            alpha=0.4,
                            label='After event',
                        )

                ax[0, ind].tick_params(axis='both', which='major', labelsize=16)
                ax[0, ind].set_xticks(dt_list)
                ax[0, ind].set_xticklabels(dt_list)
                ax[0, ind].set_ylim(1e-15, 1e1)
                ax[0, ind].set_yscale('log', base=10)
                ax[0, ind].set_xlabel(r'Step size $\Delta t$', fontsize=16)
                ax[0, 0].set_ylabel(r'Absolute values of h $|h(v_{C_n}(t))|$', fontsize=16)
                ax[0, ind].grid(visible=True)
                ax[0, ind].minorticks_off()
                ax[0, ind].legend(frameon=True, fontsize=12, loc='lower left')

        fig.savefig(
            'data/{}_comparison_event{}_M={}.png'.format(prob_cls_name, i + 1, M_fix), dpi=300, bbox_inches='tight'
        )
        plt_helper.plt.close(fig)


def plotStateFunctionOverTime(u_num, prob_cls_name, M_fix):  # pragma: no cover
    r"""
    Routine that plots the state function over time.

    Parameters
    ----------
    u_num : dict
        Contains the all the data. Dictionary has the structure ``u_num[dt][M][use_SE][use_A]``,
        where for each step size ``dt``, for each number of collocation node ``M``, for each
        combination of event detection ``use_SE`` and adaptivity ``use_A`` appropriate stuff is stored.
        For more details, see ``pySDC.projects.PinTSimE.battery_model.getDataDict``.
    prob_cls_name : str
        Indicates the name of the problem class to be considered.
    M_fix : int
        Fixed number of collocation nodes the plot is generated for.
    """

    dt_list = u_num.keys()
    use_detection = u_num[list(dt_list)[0]][M_fix].keys()
    use_adaptivity = u_num[list(dt_list)[0]][M_fix][list(use_detection)[0]].keys()
    h0 = u_num[list(dt_list)[0]][M_fix][list(use_detection)[0]][list(use_adaptivity)[0]]['state_function']
    n = h0[0].shape[0]
    for dt in dt_list:
        figsize = (7.5, 5) if n == 1 else (12, 5)
        fig, ax = plt_helper.plt.subplots(1, n, figsize=figsize, sharex='col', sharey='row', squeeze=False)

        for use_SE in use_detection:
            for use_A in use_adaptivity:
                t = u_num[dt][M_fix][use_SE][use_A]['t']
                h_val = u_num[dt][M_fix][use_SE][use_A]['state_function']

                linestyle = 'dashdot' if use_A else 'dotted'
                for i in range(n):
                    h = h_val[:] if n == 1 else h_val[:, i]
                    ax[0, i].set_title(r'$n={}$'.format(i + 1))
                    ax[0, i].plot(
                        t, h, linestyle=linestyle, label='Detection: {}, Adaptivity: {}'.format(use_SE, use_A)
                    )

                    ax[0, i].tick_params(axis='both', which='major', labelsize=16)
                    ax[0, i].set_ylim(1e-15, 1e0)
                    ax[0, i].set_yscale('log', base=10)
                    ax[0, i].set_xlabel(r'Time $t$', fontsize=16)
                    ax[0, 0].set_ylabel(r'Absolute values of h $|h(v_{C_n}(t))|$', fontsize=16)
                    ax[0, i].grid(visible=True)
                    ax[0, i].minorticks_off()
                    ax[0, i].legend(frameon=True, fontsize=12, loc='lower left')

        fig.savefig(
            'data/{}_state_function_over_time_dt={}_M={}.png'.format(prob_cls_name, dt, M_fix),
            dpi=300,
            bbox_inches='tight',
        )
        plt_helper.plt.close(fig)


if __name__ == "__main__":
    run_estimation_check()
