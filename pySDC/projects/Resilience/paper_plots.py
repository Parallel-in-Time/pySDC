# script to make pretty plots for papers or talks
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pySDC.projects.Resilience.fault_stats import (
    FaultStats,
    run_Lorenz,
    run_Schroedinger,
    run_vdp,
    run_quench,
    run_AC,
    RECOVERY_THRESH_ABS,
)
from pySDC.projects.Resilience.strategies import (
    BaseStrategy,
    AdaptivityStrategy,
    IterateStrategy,
    HotRodStrategy,
    DIRKStrategy,
    ERKStrategy,
    AdaptivityPolynomialError,
)
from pySDC.helpers.plot_helper import setup_mpl, figsize_by_journal
from pySDC.helpers.stats_helper import get_sorted


cm = 1 / 2.5
TEXTWIDTH = 11.9446244611 * cm
JOURNAL = 'Springer_Numerical_Algorithms'
BASE_PATH = 'data/paper'


def get_stats(problem, path='data/stats-jusuf', num_procs=1, strategy_type='SDC'):
    """
    Create a FaultStats object for a given problem to use for the plots.
    Note that the statistics need to be already generated somewhere else, this function will only load them.

    Args:
        problem (function): A problem to run
        path (str): Path to the associated stats for the problem

    Returns:
        FaultStats: Object to analyse resilience statistics from
    """
    if strategy_type == 'SDC':
        strategies = [BaseStrategy(), AdaptivityStrategy(), IterateStrategy()]
        if JOURNAL not in ['JSC_beamer']:
            strategies += [HotRodStrategy(), AdaptivityPolynomialError()]
    elif strategy_type == 'RK':
        strategies = [DIRKStrategy()]
        if problem.__name__ in ['run_Lorenz', 'run_vdp']:
            strategies += [ERKStrategy()]

    stats_analyser = FaultStats(
        prob=problem,
        strategies=strategies,
        faults=[False, True],
        reload=True,
        recovery_thresh=1.1,
        recovery_thresh_abs=RECOVERY_THRESH_ABS.get(problem, 0),
        mode='default',
        stats_path=path,
        num_procs=num_procs,
    )
    stats_analyser.get_recovered()
    return stats_analyser


def my_setup_mpl(**kwargs):
    setup_mpl(reset=True, font_size=8)
    mpl.rcParams.update({'lines.markersize': 6})


def savefig(fig, name, format='pdf', tight_layout=True):  # pragma: no cover
    """
    Save a figure to some predefined location.

    Args:
        fig (Matplotlib.Figure): The figure of the plot
        name (str): The name of the plot
        tight_layout (bool): Apply tight layout or leave as is
    Returns:
        None
    """
    if tight_layout:
        fig.tight_layout()
    path = f'{BASE_PATH}/{name}.{format}'
    fig.savefig(path, bbox_inches='tight', transparent=True, dpi=200)
    print(f'saved "{path}"')


def analyse_resilience(problem, path='data/stats', **kwargs):  # pragma: no cover
    """
    Generate some stats for resilience / load them if already available and make some plots.

    Args:
        problem (function): A problem to run
        path (str): Path to the associated stats for the problem

    Returns:
        None
    """

    stats_analyser = get_stats(problem, path)
    stats_analyser.get_recovered()

    strategy = IterateStrategy()
    not_fixed = stats_analyser.get_mask(strategy=strategy, key='recovered', val=False)
    not_overflow = stats_analyser.get_mask(strategy=strategy, key='bit', val=1, op='uneq', old_mask=not_fixed)
    stats_analyser.print_faults(not_overflow)

    compare_strategies(stats_analyser, **kwargs)
    plot_recovery_rate(stats_analyser, **kwargs)


def compare_strategies(stats_analyser, **kwargs):  # pragma: no cover
    """
    Make a plot showing local error and iteration number of time for all strategies

    Args:
        stats_analyser (FaultStats): Fault stats object, which contains some stats

    Returns:
        None
    """
    my_setup_mpl()
    fig, ax = plt.subplots(figsize=(TEXTWIDTH, 5 * cm))
    stats_analyser.compare_strategies(ax=ax)
    savefig(fig, 'compare_strategies', **kwargs)


def plot_recovery_rate(stats_analyser, **kwargs):  # pragma: no cover
    """
    Make a plot showing recovery rate for all faults and only for those that can be recovered.

    Args:
        stats_analyser (FaultStats): Fault stats object, which contains some stats

    Returns:
        None
    """
    my_setup_mpl()
    fig, axs = plt.subplots(1, 2, figsize=(TEXTWIDTH, 5 * cm), sharex=True, sharey=True)
    stats_analyser.plot_things_per_things(
        'recovered',
        'bit',
        False,
        op=stats_analyser.rec_rate,
        args={'ylabel': 'recovery rate'},
        plotting_args={'markevery': 5},
        ax=axs[0],
    )
    plot_recovery_rate_recoverable_only(stats_analyser, fig, axs[1], ylabel='')
    axs[0].get_legend().remove()
    axs[0].set_title('All faults')
    axs[1].set_title('Only recoverable faults')
    axs[0].set_ylim((-0.05, 1.05))
    savefig(fig, 'recovery_rate_compared', **kwargs)


def plot_recovery_rate_recoverable_only(stats_analyser, fig, ax, **kwargs):  # pragma: no cover
    """
    Plot the recovery rate considering only faults that can be recovered theoretically.

    Args:
        stats_analyser (FaultStats): Fault stats object, which contains some stats
        fig (matplotlib.pyplot.figure): Figure in which to plot
        ax (matplotlib.pyplot.axes): Somewhere to plot

    Returns:
        None
    """
    for i in range(len(stats_analyser.strategies)):
        fixable = stats_analyser.get_fixable_faults_only(strategy=stats_analyser.strategies[i])

        stats_analyser.plot_things_per_things(
            'recovered',
            'bit',
            False,
            op=stats_analyser.rec_rate,
            mask=fixable,
            args={**kwargs},
            ax=ax,
            fig=fig,
            strategies=[stats_analyser.strategies[i]],
            plotting_args={'markevery': 5},
        )


def compare_recovery_rate_problems(**kwargs):  # pragma: no cover
    """
    Compare the recovery rate for vdP, Lorenz and Schroedinger problems.
    Only faults that can be recovered are shown.

    Returns:
        None
    """
    stats = [
        get_stats(run_vdp, **kwargs),
        get_stats(run_quench, **kwargs),
        get_stats(run_Schroedinger, **kwargs),
        get_stats(run_AC, **kwargs),
    ]
    titles = ['Van der Pol', 'Quench', r'Schr\"odinger', 'Allen-Cahn']

    my_setup_mpl()
    fig, axs = plt.subplots(2, 2, figsize=figsize_by_journal(JOURNAL, 1, 0.8), sharey=True)
    [
        plot_recovery_rate_recoverable_only(stats[i], fig, axs.flatten()[i], ylabel='', title=titles[i])
        for i in range(len(stats))
    ]

    for ax in axs.flatten():
        ax.get_legend().remove()

    if kwargs.get('strategy_type', 'SDC') == 'SDC':
        axs[1, 1].legend(frameon=False)
    else:
        axs[0, 1].legend(frameon=False)
    axs[0, 0].set_ylim((-0.05, 1.05))
    axs[1, 0].set_ylabel('recovery rate')
    axs[0, 0].set_ylabel('recovery rate')

    name = ''
    for key, val in kwargs.items():
        name = f'{name}_{key}-{val}'

    savefig(fig, f'compare_equations{name}.pdf')


def plot_adaptivity_stuff():  # pragma: no cover
    """
    Plot the solution for a van der Pol problem as well as the local error and cost associated with the base scheme and
    adaptivity in k and dt in order to demonstrate that adaptivity is useful.

    Returns:
        None
    """
    from pySDC.implementations.convergence_controller_classes.estimate_embedded_error import EstimateEmbeddedError
    from pySDC.implementations.hooks.log_errors import LogLocalErrorPostStep
    from pySDC.implementations.hooks.log_work import LogWork
    from pySDC.projects.Resilience.hook import LogData

    stats_analyser = get_stats(run_vdp, 'data/stats')

    my_setup_mpl()
    scale = 0.5 if JOURNAL == 'JSC_beamer' else 1.0
    fig, axs = plt.subplots(3, 1, figsize=figsize_by_journal(JOURNAL, scale, 1), sharex=True, sharey=False)

    def plot_error(stats, ax, iter_ax, strategy, **kwargs):
        """
        Plot global error and cumulative sum of iterations

        Args:
            stats (dict): Stats from pySDC run
            ax (Matplotlib.pyplot.axes): Somewhere to plot the error
            iter_ax (Matplotlib.pyplot.axes): Somewhere to plot the iterations
            strategy (pySDC.projects.Resilience.fault_stats.Strategy): The resilience strategy

        Returns:
            None
        """
        markevery = 40
        e = get_sorted(stats, type='e_local_post_step', recomputed=False)
        ax.plot([me[0] for me in e], [me[1] for me in e], markevery=markevery, **strategy.style, **kwargs)
        k = get_sorted(stats, type='work_newton')
        iter_ax.plot(
            [me[0] for me in k], np.cumsum([me[1] for me in k]), **strategy.style, markevery=markevery, **kwargs
        )
        ax.set_yscale('log')
        ax.set_ylabel('local error')
        iter_ax.set_ylabel(r'Newton iterations')

    force_params = {}
    for strategy in [BaseStrategy, AdaptivityStrategy, IterateStrategy, AdaptivityPolynomialError]:
        if strategy == AdaptivityPolynomialError:
            from pySDC.implementations.convergence_controller_classes.adaptivity import (
                AdaptivityPolynomialError as adaptivity,
            )

            force_params = {'sweeper_params': {'num_nodes': 2}}
            force_params['convergence_controllers'] = {
                adaptivity: {
                    'e_tol': 7e-5,
                    'restol_rel': 1e-4,
                    'restol_min': 1e-10,
                    'restart_at_maxiter': True,
                    'factor_if_not_converged': 4.0,
                },
            }
        else:
            force_params = {}
        stats, _, _ = stats_analyser.single_run(
            strategy=strategy(useMPI=False),
            force_params=force_params,
            hook_class=[LogLocalErrorPostStep, LogData, LogWork],
        )
        plot_error(stats, axs[1], axs[2], strategy())

        if strategy == BaseStrategy:
            u = get_sorted(stats, type='u', recomputed=False)
            axs[0].plot([me[0] for me in u], [me[1][0] for me in u], color='black', label=r'$u$')

    axs[2].set_xlabel(r'$t$')
    axs[0].set_ylabel('solution')
    axs[2].legend(frameon=JOURNAL == 'JSC_beamer')
    axs[1].legend(frameon=True)
    savefig(fig, 'adaptivity')


def plot_fault_vdp(bit=0):  # pragma: no cover
    """
    Make a plot showing the impact of a fault on van der Pol without any resilience.
    The faults are inserted in the last iteration in the last node in u_t such that you can best see the impact.

    Args:
        bit (int): The bit that you want to flip

    Returns:
        None
    """
    from pySDC.projects.Resilience.fault_stats import (
        FaultStats,
        BaseStrategy,
    )
    from pySDC.projects.Resilience.hook import LogData

    stats_analyser = FaultStats(
        prob=run_vdp,
        strategies=[BaseStrategy()],
        faults=[False, True],
        reload=True,
        recovery_thresh=1.1,
        num_procs=1,
        mode='combination',
    )

    my_setup_mpl()
    fig, ax = plt.subplots(figsize=figsize_by_journal(JOURNAL, 0.8, 0.5))
    colors = ['blue', 'red', 'magenta']
    ls = ['--', '-']
    markers = ['*', '^']
    do_faults = [False, True]
    superscripts = ['*', '']
    subscripts = ['', 't', '']

    run = 779 + 12 * bit  # for faults in u_t
    #  run = 11 + 12 * bit  # for faults in u

    for i in range(len(do_faults)):
        stats, controller, Tend = stats_analyser.single_run(
            strategy=BaseStrategy(),
            run=run,
            faults=do_faults[i],
            hook_class=[LogData],
        )
        u = get_sorted(stats, type='u')
        faults = get_sorted(stats, type='bitflip')
        for j in [0, 1]:
            ax.plot(
                [me[0] for me in u],
                [me[1][j] for me in u],
                ls=ls[i],
                color=colors[j],
                label=rf'$u^{{{superscripts[i]}}}_{{{subscripts[j]}}}$',
                marker=markers[j],
                markevery=60,
            )
        for idx in range(len(faults)):
            ax.axvline(faults[idx][0], color='black', label='Fault', ls=':')
            print(
                f'Fault at t={faults[idx][0]:.2e}, iter={faults[idx][1][1]}, node={faults[idx][1][2]}, space={faults[idx][1][3]}, bit={faults[idx][1][4]}'
            )
            ax.set_title(f'Fault in bit {faults[idx][1][4]}')

    ax.legend(frameon=True, loc='lower left')
    ax.set_xlabel(r'$t$')
    savefig(fig, f'fault_bit_{bit}')


def plot_quench_solution():  # pragma: no cover
    """
    Plot the solution of Quench problem over time

    Returns:
        None
    """
    my_setup_mpl()
    if JOURNAL == 'JSC_beamer':
        fig, ax = plt.subplots(figsize=figsize_by_journal(JOURNAL, 0.5, 0.9))
    else:
        fig, ax = plt.subplots(figsize=figsize_by_journal(JOURNAL, 1.0, 0.45))

    strategy = BaseStrategy()

    custom_description = strategy.get_custom_description(run_quench, num_procs=1)

    stats, controller, _ = run_quench(custom_description=custom_description, Tend=strategy.get_Tend(run_quench))

    prob = controller.MS[0].levels[0].prob

    u = get_sorted(stats, type='u', recomputed=False)

    ax.plot([me[0] for me in u], [max(me[1]) for me in u], color='black', label='$T$')
    ax.axhline(prob.u_thresh, label='$T_\mathrm{thresh}$', ls='--', color='grey', zorder=-1)
    ax.axhline(prob.u_max, label='$T_\mathrm{max}$', ls=':', color='grey', zorder=-1)

    ax.set_xlabel(r'$t$')
    ax.legend(frameon=False)
    savefig(fig, 'quench_sol')


def plot_AC_solution():  # pragma: no cover
    from pySDC.projects.TOMS.AllenCahn_monitor import monitor

    my_setup_mpl()
    if JOURNAL == 'JSC_beamer':
        raise NotImplementedError
        fig, ax = plt.subplots(figsize=figsize_by_journal(JOURNAL, 0.5, 0.9))
    else:
        fig, axs = plt.subplots(1, 2, figsize=figsize_by_journal(JOURNAL, 1.0, 0.45))

    stats, _, _ = run_AC(Tend=0.032, hook_class=monitor)

    u = get_sorted(stats, type='u')

    computed_radius = get_sorted(stats, type='computed_radius')
    exact_radius = get_sorted(stats, type='exact_radius')
    axs[1].plot([me[0] for me in computed_radius], [me[1] for me in computed_radius], ls='-', label='numerical')
    axs[1].plot([me[0] for me in exact_radius], [me[1] for me in exact_radius], ls='--', color='black', label='exact')
    axs[1].axvline(0.025, ls=':', label=r'$t=0.025$', color='grey')
    axs[1].set_title('Radius over time')
    axs[1].set_xlabel('$t$')
    axs[1].legend(frameon=False)

    im = axs[0].imshow(u[0][1], extent=(-0.5, 0.5, -0.5, 0.5))
    fig.colorbar(im)
    axs[0].set_title(r'$u_0$')
    axs[0].set_xlabel('$x$')
    axs[0].set_ylabel('$y$')
    savefig(fig, 'AC_sol')


def plot_vdp_solution():  # pragma: no cover
    """
    Plot the solution of van der Pol problem over time to illustrate the varying time scales.

    Returns:
        None
    """
    from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

    my_setup_mpl()
    if JOURNAL == 'JSC_beamer':
        fig, ax = plt.subplots(figsize=figsize_by_journal(JOURNAL, 0.5, 0.9))
    else:
        fig, ax = plt.subplots(figsize=figsize_by_journal(JOURNAL, 1.0, 0.33))

    custom_description = {'convergence_controllers': {Adaptivity: {'e_tol': 1e-7}}}

    stats, _, _ = run_vdp(custom_description=custom_description, Tend=28.6)

    u = get_sorted(stats, type='u')
    ax.plot([me[0] for me in u], [me[1][0] for me in u], color='black')
    ax.set_ylabel(r'$u$')
    ax.set_xlabel(r'$t$')
    savefig(fig, 'vdp_sol')


def work_precision():  # pragma: no cover
    from pySDC.projects.Resilience.work_precision import (
        all_problems,
        single_problem,
        ODEs,
        get_fig,
        execute_configurations,
        save_fig,
        get_configs,
        MPI,
        vdp_stiffness_plot,
    )

    all_params = {
        'record': False,
        'work_key': 't',
        'precision_key': 'e_global_rel',
        'plotting': True,
        'base_path': 'data/paper',
    }

    for mode in ['compare_strategies', 'parallel_efficiency', 'RK_comp']:
        all_problems(**all_params, mode=mode)

    # # Quench stuff
    # fig, axs = get_fig(x=3, y=1, figsize=figsize_by_journal('Springer_Numerical_Algorithms', 1, 0.47))
    # quench_params = {
    #     **all_params,
    #     'problem': run_quench,
    #     'decorate': True,
    #     'configurations': get_configs('step_size_limiting', run_quench),
    #     'num_procs': 1,
    #     'runs': 1,
    #     'comm_world': MPI.COMM_WORLD,
    #     'mode': 'step_size_limiting',
    # }
    # quench_params.pop('base_path', None)
    # execute_configurations(**{**quench_params, 'work_key': 'k_SDC', 'precision_key': 'k_Newton'}, ax=axs[2])
    # execute_configurations(**{**quench_params, 'work_key': 'param', 'precision_key': 'restart'}, ax=axs[1])
    # execute_configurations(**{**quench_params, 'work_key': 't', 'precision_key': 'e_global_rel'}, ax=axs[0])
    # axs[1].set_yscale('linear')
    # # axs[2].set_yscale('linear')
    # axs[2].set_xscale('linear')
    # axs[1].set_xlabel(r'$e_\mathrm{tol}$')
    # # axs[0].set_xticks([1e0, 3e0], [r'$10^{0}$', r'$3\times 10^{0}$'], minor=False)

    # for ax in axs:
    #     ax.set_title(ax.get_ylabel())
    #     ax.set_ylabel('')
    # fig.suptitle('Quench')

    # axs[1].set_yticks([4.0, 6.0, 8.0, 10.0, 12.0], minor=False)

    # save_fig(
    #     fig=fig,
    #     name=f'{run_quench.__name__}',
    #     work_key='step-size',
    #     precision_key='limiting',
    #     legend=True,
    #     base_path=all_params["base_path"],
    # )
    # End Quench stuff

    # vdp_stiffness_plot(base_path='data/paper')


def make_plots_for_TIME_X_website():  # pragma: no cover
    global JOURNAL, BASE_PATH
    JOURNAL = 'JSC_beamer'
    BASE_PATH = 'data/paper/time-x_website'

    fig, ax = plt.subplots(figsize=figsize_by_journal(JOURNAL, 0.5, 2.0 / 3.0))
    plot_recovery_rate_recoverable_only(get_stats(run_vdp), fig, ax)
    savefig(fig, 'recovery_rate', format='png')

    from pySDC.projects.Resilience.work_precision import vdp_stiffness_plot

    vdp_stiffness_plot(base_path=BASE_PATH, format='png')


def make_plots_for_SIAM_CSE23():  # pragma: no cover
    """
    Make plots for the SIAM talk
    """
    global JOURNAL, BASE_PATH
    JOURNAL = 'JSC_beamer'
    BASE_PATH = 'data/paper/SIAMCSE23'

    fig, ax = plt.subplots(figsize=figsize_by_journal(JOURNAL, 0.5, 3.0 / 4.0))
    plot_recovery_rate_recoverable_only(get_stats(run_vdp), fig, ax)
    savefig(fig, 'recovery_rate')

    plot_adaptivity_stuff()
    compare_recovery_rate_problems()
    plot_vdp_solution()


def make_plots_for_paper():  # pragma: no cover
    """
    Make plots that are supposed to go in the paper.
    """
    global JOURNAL, BASE_PATH
    JOURNAL = 'Springer_Numerical_Algorithms'
    BASE_PATH = 'data/paper'

    plot_adaptivity_stuff()

    work_precision()

    plot_vdp_solution()
    plot_AC_solution()
    plot_quench_solution()

    plot_recovery_rate(get_stats(run_vdp))
    plot_fault_vdp(0)
    plot_fault_vdp(13)
    compare_recovery_rate_problems(num_procs=1, strategy_type='SDC')


def make_plots_for_notes():  # pragma: no cover
    """
    Make plots for the notes for the website / GitHub
    """
    global JOURNAL, BASE_PATH
    JOURNAL = 'Springer_Numerical_Algorithms'
    BASE_PATH = 'notes/Lorenz'

    analyse_resilience(run_Lorenz, format='png')
    analyse_resilience(run_quench, format='png')


if __name__ == "__main__":
    # make_plots_for_notes()
    # make_plots_for_SIAM_CSE23()
    # make_plots_for_TIME_X_website()
    make_plots_for_paper()
