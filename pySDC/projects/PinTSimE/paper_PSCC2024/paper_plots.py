from pathlib import Path
import numpy as np
import dill

from pySDC.core.Errors import ParameterError

from pySDC.projects.DAE.sweepers.fully_implicit_DAE import fully_implicit_DAE
from pySDC.projects.DAE.problems.DiscontinuousTestDAE import DiscontinuousTestDAE
from pySDC.projects.DAE.problems.WSCC9BusSystem import WSCC9BusSystem

from pySDC.projects.PinTSimE.battery_model import generateDescription, getRecomputed
from pySDC.projects.PinTSimE.battery_model import controllerRun
from pySDC.helpers.stats_helper import get_sorted
import pySDC.helpers.plot_helper as plt_helper

from pySDC.projects.PinTSimE.paper_PSCC2024.log_event import LogEventDiscontinuousTestDAE, LogEventWSCC9
from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostStep
from pySDC.implementations.hooks.log_restarts import LogRestarts


def make_plots_for_test_DAE():  # pragma: no cover
    """
    Makes the plot for the discontinuous test DAE, i.e.,

        - error over time for fixed time step size for different number of collocation nodes in
          comparison with use of switch detection and not,
        - error norm for different step sizes and different number of collocation nodes in
          comparison with use of switch detection and not, additionally with number of restarts
          for each case,
        - absolute value of state function at end time for different number of collocation nodes
          and different step sizes in comparison with use of switch detection and not,
        - event error to exact event time for differen step sizes and different number of
          collocation nodes,
        - plots event time error of all founded events (not necessarily satisfying the tolerances)
          and the maximum value of the state function in this time step for different number of
          collocation nodes in comparison with use of switch detection and not.

    Thus, this function contains all the parameters used in the paper for this numerical example.
    """

    Path("data").mkdir(parents=True, exist_ok=True)

    problem_class = DiscontinuousTestDAE
    prob_class_name = DiscontinuousTestDAE.__name__

    sweeper = fully_implicit_DAE
    nnodes = [2, 3, 4, 5]
    quad_type = 'RADAU-RIGHT'
    QI = 'LU'
    maxiter = 45
    tol_hybr = 1e-6
    restol = 1e-13

    hook_class = [LogGlobalErrorPostStep, LogEventDiscontinuousTestDAE, LogRestarts]

    problem_params = dict()
    problem_params['newton_tol'] = tol_hybr

    use_detection = [False, True]
    max_restarts = 200
    epsilon_SE = 1e-10
    alpha = 0.95

    t0 = 3.0
    Tend = 5.4

    dt_list = [1 / (2**m) for m in range(2, 9)]
    dt_fix = 1 / (2**7)

    recomputed = False

    results_error_over_time = {}
    results_error_norm = {}
    results_state_function = {}
    results_event_error = {}
    results_event_error_restarts = {}

    for M in nnodes:
        results_error_over_time[M], results_error_norm[M] = {}, {}
        results_state_function[M], results_event_error[M] = {}, {}
        results_event_error_restarts[M] = {}

        for dt in dt_list:
            results_error_over_time[M][dt], results_error_norm[M][dt] = {}, {}
            results_state_function[M][dt], results_event_error[M][dt] = {}, {}
            results_event_error_restarts[M][dt] = {}

            for use_SE in use_detection:
                results_error_over_time[M][dt][use_SE], results_error_norm[M][dt][use_SE] = {}, {}
                results_state_function[M][dt][use_SE], results_event_error[M][dt][use_SE] = {}, {}
                results_event_error_restarts[M][dt][use_SE] = {}

                description, controller_params = generateDescription(
                    dt,
                    problem_class,
                    sweeper,
                    M,
                    quad_type,
                    QI,
                    hook_class,
                    False,
                    use_SE,
                    problem_params,
                    restol,
                    maxiter,
                    max_restarts,
                    epsilon_SE,
                    alpha,
                )

                stats, t_switch_exact = controllerRun(
                    description, controller_params, t0, Tend, exact_event_time_avail=True
                )

                err_val = get_sorted(stats, type='e_global_post_step', sortby='time', recomputed=recomputed)
                results_error_over_time[M][dt][use_SE] = err_val

                err_norm = max([item[1] for item in err_val])
                results_error_norm[M][dt][use_SE] = err_norm

                h_val = get_sorted(stats, type='state_function', sortby='time', recomputed=recomputed)
                h_abs = abs([item[1] for item in h_val][-1])
                results_state_function[M][dt][use_SE]['h_abs'] = h_abs

                if use_SE:
                    switches = getRecomputed(stats, type='switch', sortby='time')

                    t_switch = [item[1] for item in switches][-1]
                    results_event_error[M][dt][use_SE] = abs(t_switch_exact - t_switch)

                    restarts = get_sorted(stats, type='restart', sortby='time', recomputed=None)
                    sum_restarts = sum([item[1] for item in restarts])
                    results_state_function[M][dt][use_SE]['restarts'] = sum_restarts

                    switches_all = get_sorted(stats, type='switch_all', sortby='time', recomputed=None)
                    t_switches_all = [item[1] for item in switches_all]
                    event_error_all = [abs(t_switch_exact - t_switch) for t_switch in t_switches_all]
                    results_event_error_restarts[M][dt][use_SE]['event_error_all'] = event_error_all
                    h_val_all = get_sorted(stats, type='h_all', sortby='time', recomputed=None)
                    results_event_error_restarts[M][dt][use_SE]['h_max_event'] = [item[1] for item in h_val_all]

    plot_functions_over_time(
        results_error_over_time, prob_class_name, r'Global error $|y(t) - y_{ex}(t)|$', 'upper left', dt_fix
    )
    plot_error_norm(results_error_norm, prob_class_name)
    plot_state_function_detection(
        results_state_function, prob_class_name, r'Absolute value of $h$ $|h(y(T))|$', 'upper left'
    )
    plot_event_time_error(results_event_error, prob_class_name)
    plot_event_time_error_before_restarts(results_event_error_restarts, prob_class_name, dt_fix)


def make_plots_for_WSCC9_test_case(cwd='./'):  # pragma: no cover
    """
    Generates the plots for the WSCC 9-bus test case, i.e.,

        - the values of the state function over time for different number of collocation nodes in comparison
          with event detection and not,
        - the values of the state function at end time for different number of collocation nodes and
          different step sizes.

    Thus, this function contains all the parameters used for this numerical example.

    Parameters
    ----------
    cwd : str, optional
        Current working directory.
    """

    Path("data").mkdir(parents=True, exist_ok=True)

    problem_class = WSCC9BusSystem
    prob_class_name = WSCC9BusSystem.__name__

    sweeper = fully_implicit_DAE
    nnodes = [2, 3, 4, 5]
    quad_type = 'RADAU-RIGHT'
    QI = 'LU'
    maxiter = 50
    tol_hybr = 1e-10
    restol = 5e-13

    hook_class = [LogEventWSCC9, LogRestarts]

    problem_params = dict()
    problem_params['newton_tol'] = tol_hybr

    use_detection = [False, True]
    max_restarts = 400
    epsilon_SE = 1e-10
    alpha = 0.95

    t0 = 0.0
    Tend = 0.7

    dt_list = [1 / (2**m) for m in range(5, 11)]
    dt_fix = 1 / (2**8)

    recomputed = False

    results_state_function_over_time = {}
    results_state_function_detection = {}
    for M in nnodes:
        results_state_function_over_time[M], results_state_function_detection[M] = {}, {}

        for dt in dt_list:
            results_state_function_over_time[M][dt], results_state_function_detection[M][dt] = {}, {}

            for use_SE in use_detection:
                results_state_function_over_time[M][dt][use_SE], results_state_function_detection[M][dt][use_SE] = (
                    {},
                    {},
                )

                description, controller_params = generateDescription(
                    dt,
                    problem_class,
                    sweeper,
                    M,
                    quad_type,
                    QI,
                    hook_class,
                    False,
                    use_SE,
                    problem_params,
                    restol,
                    maxiter,
                    max_restarts,
                    epsilon_SE,
                    alpha,
                )

                # ---- either solution is computed or it is loaded from .dat file already created ----
                path = Path('data/{}_M={}_dt={}_useSE={}.dat'.format(prob_class_name, M, dt, use_SE))
                if path.is_file():
                    f = open(cwd + 'data/{}_M={}_dt={}_useSE={}.dat'.format(prob_class_name, M, dt, use_SE), 'rb')
                    stats = dill.load(f)
                    f.close()
                else:
                    stats, _ = controllerRun(description, controller_params, t0, Tend)

                    fname = 'data/{}_M={}_dt={}_useSE={}.dat'.format(prob_class_name, M, dt, use_SE)
                    f = open(fname, 'wb')
                    dill.dump(stats, f)
                    f.close()

                h_val = get_sorted(stats, type='state_function', sortby='time', recomputed=recomputed)
                results_state_function_over_time[M][dt][use_SE] = h_val

                h_abs_end = abs([me[1] for me in h_val][-1])
                results_state_function_detection[M][dt][use_SE]['h_abs'] = h_abs_end

                if use_SE:
                    restarts = get_sorted(stats, type='restart', sortby='time', recomputed=None)
                    sum_restarts = sum([me[1] for me in restarts])
                    results_state_function_detection[M][dt][use_SE]['restarts'] = sum_restarts

    plot_functions_over_time(
        results_state_function_over_time,
        prob_class_name,
        r'Absolute value of $h$ $|h(P_{SV,0}(t))|$',
        'lower left',
        dt_fix,
    )
    plot_state_function_detection(
        results_state_function_detection, prob_class_name, r'Absolute value of $h$ $|h(P_{SV,0}(T))|$', 'upper right'
    )


def plot_styling_stuff(prob_class):  # pragma: no cover
    """
    Implements all the stuff needed for making the plots more pretty.
    """

    colors = {
        2: 'limegreen',
        3: 'firebrick',
        4: 'deepskyblue',
        5: 'purple',
    }

    markers = {
        2: 's',
        3: 'o',
        4: '*',
        5: 'd',
    }

    if prob_class == 'DiscontinuousTestDAE':
        xytext = {
            2: (-15.0, 16.5),
            3: (-2.0, 55),
            4: (-13.0, -27),
            5: (-1.0, -40),
        }
    elif prob_class == 'WSCC9BusSystem':
        xytext = {
            2: (-13.0, 16),
            3: (-13.0, 30),
            4: (-13.0, -17),
            5: (-1.0, -38),
        }
    else:
        raise ParameterError(f"For {prob_class} no dictionary for position of data points is set up!")

    return colors, markers, xytext


def plot_functions_over_time(
    results_function_over_time, prob_class, y_label, loc_legend, dt_fix=None
):  # pragma: no cover
    """
    Plots the functions over time for different numbers of collocation nodes in comparison with detection
    and not.

    Parameters
    ----------
    results_function_over_time : dict
        Results of some function over time for different number of coll.nodes.
    prob_class : str
        Indicates of which problem class results are plotted (used to define the file name).
    y_label : str
        y-label used for the plot.
    loc_legend : str
        Location of the legend in the plot.
    dt_fix : bool, optional
        If it is set to a considered step size, only one plot will generated.
    """

    colors, _, _ = plot_styling_stuff(prob_class)
    x0 = 3.5 if prob_class == 'DiscontinuousTestDAE' else 0.5

    M_key = list(results_function_over_time.keys())[0]
    dt_list = [dt_fix] if dt_fix is not None else results_function_over_time[M_key].keys()
    for dt in dt_list:
        fig, ax = plt_helper.plt.subplots(1, 1, figsize=(7.5, 5))
        for M in results_function_over_time.keys():
            for use_SE in results_function_over_time[M][dt].keys():
                err_val = results_function_over_time[M][dt][use_SE]
                t, err = [item[0] for item in err_val], [abs(item[1]) for item in err_val]

                linestyle_detection = 'solid' if not use_SE else 'dashdot'
                (line,) = ax.plot(t, err, color=colors[M], linestyle=linestyle_detection)

                if not use_SE:
                    line.set_label(r'$M={}$'.format(M))

                if M == 5:  # dummy plot for more pretty legend
                    ax.plot(x0, 0, color='black', linestyle=linestyle_detection, label='Detection: {}'.format(use_SE))

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_ylim(1e-15, 1e1)
        ax.set_yscale('log', base=10)
        ax.set_xlabel(r'Time $t$', fontsize=16)
        ax.set_ylabel(y_label, fontsize=16)
        ax.grid(visible=True)
        ax.legend(frameon=True, fontsize=12, loc=loc_legend)
        ax.minorticks_off()

        if prob_class == 'DiscontinuousTestDAE':
            file_name = 'data/test_DAE_error_over_time_dt{}.png'.format(dt)
        elif prob_class == 'WSCC9BusSystem':
            file_name = 'data/wscc9_state_function_over_time_dt{}.png'.format(dt)
        else:
            raise ParameterError(f"For {prob_class} no file name is implemented!")

        fig.savefig(file_name, dpi=300, bbox_inches='tight')
        plt_helper.plt.close(fig)


def plot_error_norm(results_error_norm, prob_class):  # pragma: no cover
    """
    Plots the error norm for different step sizes and different number of collocation nodes in comparison
    with detection and not.

    Parameters
    ----------
    results_error_norm : dict
        Statistics containing the error norms and sum of restarts for all considered coll. nodes.
    prob_class : str
        Indicates of which problem class results are plotted (used to define the file name).
    """

    colors, markers, xytext = plot_styling_stuff(prob_class)

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(7.5, 5))
    for M in results_error_norm.keys():
        dt = list(results_error_norm[M].keys())
        for use_SE in results_error_norm[M][dt[0]].keys():
            err_norm_dt = [results_error_norm[M][k][use_SE] for k in dt]

            linestyle_detection = 'solid' if not use_SE else 'dashdot'
            (line,) = ax.loglog(
                dt,
                err_norm_dt,
                color=colors[M],
                linestyle=linestyle_detection,
                marker=markers[M],
            )

            if not use_SE:
                line.set_label(r'$M={}$'.format(M))

            if M == 5:  # dummy plot for more pretty legend
                ax.plot(0, 0, color='black', linestyle=linestyle_detection, label='Detection: {}'.format(use_SE))

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylim(1e-15, 1e3)
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)
    ax.set_xlabel(r'Step size $\Delta t$', fontsize=16)
    ax.set_ylabel(r'Error norm $||y(t) - \tilde{y}(t)||_\infty$', fontsize=16)
    ax.grid(visible=True)
    ax.minorticks_off()
    ax.legend(frameon=True, fontsize=12, loc='lower right')

    fig.savefig('data/test_DAE_error_norms.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


def plot_state_function_detection(results_state_function, prob_class, y_label, loc_legend):  # pragma: no cover
    """
    Plots the absolute value of the state function after the event which denotes how close it is to the zero.

    Parameters
    ----------
    results_state_function : dict
        Contains the absolute value of the state function for each number of coll. nodes, each step size and
        detection and not.
    prob_class : str
        Indicates of which problem class results are plotted (used to define the file name).
    y_label : str
        y-label used for the plot.
    loc_legend : str
        Location of the legend in the plot.
    """

    colors, markers, xytext = plot_styling_stuff(prob_class)

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(7.5, 5))
    for M in results_state_function.keys():
        dt = list(results_state_function[M].keys())
        for use_SE in results_state_function[M][dt[0]].keys():
            h_abs = [results_state_function[M][k][use_SE]['h_abs'] for k in dt]

            linestyle_detection = 'solid' if not use_SE else 'dashdot'
            (line,) = ax.loglog(
                dt,
                h_abs,
                color=colors[M],
                linestyle=linestyle_detection,
                marker=markers[M],
            )

            if not use_SE:
                line.set_label(r'$M={}$'.format(M))

            if use_SE:
                sum_restarts = [results_state_function[M][k][use_SE]['restarts'] for k in dt]
                for m in range(len(dt)):
                    ax.annotate(
                        sum_restarts[m],
                        (dt[m], h_abs[m]),
                        xytext=xytext[M],
                        textcoords="offset points",
                        color=colors[M],
                        fontsize=16,
                    )

            if M == 5:  # dummy plot for more pretty legend
                ax.plot(0, 0, color='black', linestyle=linestyle_detection, label='Detection: {}'.format(use_SE))

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylim(1e-17, 1e3)
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)
    ax.set_xlabel(r'Step size $\Delta t$', fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.grid(visible=True)
    ax.minorticks_off()
    ax.legend(frameon=True, fontsize=12, loc=loc_legend)

    if prob_class == 'DiscontinuousTestDAE':
        file_name = 'data/test_DAE_state_function.png'
    elif prob_class == 'WSCC9BusSystem':
        file_name = 'data/wscc9_state_function_detection.png'
    else:
        raise ParameterError(f"For {prob_class} no file name is set up!")

    fig.savefig(file_name, dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


def plot_event_time_error(results_event_error, prob_class):  # pragma: no cover
    """
    Plots the error between event time founded by detection and exact event time.

    Parameters
    ----------
    results_event_error : dict
        Contains event time error for each considered number of coll. nodes, step size and
        event detection and not.
    prob_class : str
        Indicates of which problem class results are plotted (used to define the file name).
    """

    colors, markers, _ = plot_styling_stuff(prob_class)

    fig, ax = plt_helper.plt.subplots(1, 1, figsize=(7.5, 5))
    for M in results_event_error.keys():
        dt = list(results_event_error[M].keys())
        for use_SE in [True]:
            event_error = [results_event_error[M][k][use_SE] for k in dt]

            linestyle_detection = 'solid' if not use_SE else 'dashdot'
            ax.loglog(
                dt,
                event_error,
                color=colors[M],
                linestyle=linestyle_detection,
                marker=markers[M],
                label=r'$M={}$'.format(M),
            )

    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_ylim(1e-15, 1e1)
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)
    ax.set_xlabel(r'Step size $\Delta t$', fontsize=16)
    ax.set_ylabel(r'Event time error $|t^*_{ex} - t^*_{SE}|$', fontsize=16)
    ax.grid(visible=True)
    ax.minorticks_off()
    ax.legend(frameon=True, fontsize=12, loc='lower right')

    fig.savefig('data/test_DAE_event_time_error.png', dpi=300, bbox_inches='tight')
    plt_helper.plt.close(fig)


def plot_event_time_error_before_restarts(results_event_error_restarts, prob_class, dt_fix=None):  # pragma: no cover
    """
    Plots all events founded by switch estimation, not necessarily satisfying the conditions.

    Parameters
    ----------
    results_event_error_restarts : dict
        Contains all events for each considered number of coll. nodes, step size and
        event detection and not.
    prob_class : str
        Indicates of which problem class results are plotted (used to define the file name).
    dt_fix : float, optional
        Step size considered.
    """

    colors, markers, _ = plot_styling_stuff(prob_class)

    M_key = list(results_event_error_restarts.keys())[0]
    dt_list = [dt_fix] if dt_fix is not None else results_event_error_restarts[M_key].keys()
    for dt in dt_list:
        fig, ax = plt_helper.plt.subplots(1, 1, figsize=(7.5, 5))
        h_ax = ax.twinx()
        for M in results_event_error_restarts.keys():
            for use_SE in results_event_error_restarts[M][dt].keys():
                if use_SE:
                    event_error_all = results_event_error_restarts[M][dt][use_SE]['event_error_all']

                    (line,) = ax.semilogy(
                        np.arange(1, len(event_error_all) + 1),
                        event_error_all,
                        color=colors[M],
                        linestyle='solid',
                        # marker=markers[M],
                    )

                    line.set_label(r'$M={}$'.format(M))

                    h_max_event = results_event_error_restarts[M][dt][use_SE]['h_max_event']
                    h_ax.semilogy(
                        np.arange(1, len(h_max_event) + 1),
                        h_max_event,
                        color=colors[M],
                        linestyle='dashdot',
                        marker=markers[M],
                        markersize=5,
                        alpha=0.4,
                    )

                    if M == 5:  # dummy plot for more pretty legend
                        ax.plot(
                            1, event_error_all[0], color='black', linestyle='solid', label=r'$|t^*_{ex} - t^*_{SE}|$'
                        )
                        ax.plot(
                            1,
                            1e2,
                            color='black',
                            linestyle='dashdot',
                            marker=markers[M],
                            markersize=5,
                            alpha=0.4,
                            label=r'$||h(t)||_\infty$',
                        )

        h_ax.tick_params(labelsize=16)
        h_ax.set_ylim(1e-11, 1e0)
        h_ax.set_yscale('log', base=10)
        h_ax.set_ylabel(r'Maximum value of h $||h(t)||_\infty$', fontsize=16)
        h_ax.minorticks_off()

        ax.tick_params(axis='both', which='major', labelsize=16)
        ax.set_ylim(1e-11, 1e-1)
        ax.set_yscale('log', base=10)
        ax.set_xlabel(r'Restarted steps $n_{restart}$', fontsize=16)
        ax.set_ylabel(r'Event time error $|t^*_{ex} - t^*_{SE}|$', fontsize=16)
        ax.grid(visible=True)
        ax.minorticks_off()
        ax.legend(frameon=True, fontsize=12, loc='upper right')

        fig.savefig('data/test_DAE_event_time_error_restarts_dt{}.png'.format(dt), dpi=300, bbox_inches='tight')
        plt_helper.plt.close(fig)


if __name__ == "__main__":
    make_plots_for_test_DAE()
    # make_plots_for_WSCC9_test_case()
