# script to make pretty plots for papers or talks
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pySDC.projects.Resilience.fault_stats import (
    FaultStats,
    BaseStrategy,
    AdaptivityStrategy,
    IterateStrategy,
    HotRodStrategy,
    run_Lorenz,
    run_Schroedinger,
    run_vdp,
    run_quench,
)
from pySDC.helpers.plot_helper import setup_mpl, figsize_by_journal
from pySDC.helpers.stats_helper import get_sorted


cm = 1 / 2.5
TEXTWIDTH = 11.9446244611 * cm
JOURNAL = 'Springer_Numerical_Algorithms'
BASE_PATH = 'data/paper'


def get_stats(problem, path='data/stats-jusuf'):
    """
    Create a FaultStats object for a given problem to use for the plots.
    Note that the statistics need to be already generated somewhere else, this function will only load them.

    Args:
        problem (function): A problem to run
        path (str): Path to the associated stats for the problem

    Returns:
        FaultStats: Object to analyse resilience statistics from
    """
    if problem in [run_Lorenz, run_vdp]:
        mode = 'combination'
    else:
        mode = 'random'

    recovery_thresh_abs = {
        run_quench: 5e-3,
    }

    strategies = [BaseStrategy(), AdaptivityStrategy(), IterateStrategy()]
    if JOURNAL not in ['JSC_beamer']:
        strategies += [HotRodStrategy()]

    stats_analyser = FaultStats(
        prob=problem,
        strategies=strategies,
        faults=[False, True],
        reload=True,
        recovery_thresh=1.1,
        recovery_thresh_abs=recovery_thresh_abs.get(problem, 0),
        num_procs=1,
        mode=mode,
        stats_path=path,
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

    # special = stats_analyser.get_mask(strategy=strategy, key='bit', val=10, op='eq')
    # stats_analyser.print_faults(special)

    # Adaptivity: 19, ...
    # stats_analyser.scrutinize(strategy, run=19, faults=True)

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
        'recovered', 'bit', False, op=stats_analyser.rec_rate, args={'ylabel': 'recovery rate'}, ax=axs[0]
    )
    plot_recovery_rate_recoverable_only(stats_analyser, fig, axs[1], ylabel='')
    axs[1].get_legend().remove()
    axs[0].set_title('All faults')
    axs[1].set_title('Only recoverable faults')
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
        )


def compare_recovery_rate_problems():  # pragma: no cover
    """
    Compare the recovery rate for vdP, Lorenz and Schroedinger problems.
    Only faults that can be recovered are shown.

    Returns:
        None
    """
    stats = [
        get_stats(run_vdp),
        get_stats(run_Lorenz),
        get_stats(run_Schroedinger),
        get_stats(run_quench),
    ]
    titles = ['Van der Pol', 'Lorenz attractor', r'Schr\"odinger', 'Quench']

    my_setup_mpl()
    fig, axs = plt.subplots(2, 2, figsize=figsize_by_journal(JOURNAL, 1, 0.8), sharey=True)
    [
        plot_recovery_rate_recoverable_only(stats[i], fig, axs.flatten()[i], ylabel='', title=titles[i])
        for i in range(len(stats))
    ]

    for ax in axs.flatten():
        ax.get_legend().remove()

    axs[1, 1].legend(frameon=False)
    axs[1, 0].set_ylabel('recovery rate')
    axs[0, 0].set_ylabel('recovery rate')

    savefig(fig, 'compare_equations')


def plot_efficiency_polar_vdp(problem, path='data/stats'):  # pragma: no cover
    stats_analyser = get_stats(problem, path)
    fig, ax = plt.subplots(
        subplot_kw={'projection': 'polar'}, figsize=figsize_by_journal(JOURNAL, 0.7, 0.5), layout='constrained'
    )
    theta, norms = plot_efficiency_polar_single(stats_analyser, ax)

    labels = ['fail rate', 'extra iterations\nfor recovery', 'iterations for solution']
    ax.set_xticks(theta[:-1], [f'{labels[i]}\nmax={norms[i]:.2f}' for i in range(len(labels))])
    ax.set_rlabel_position(90)

    fig.legend(frameon=False, loc='outside right', ncols=1)
    savefig(fig, 'efficiency', tight_layout=False)


def plot_efficiency_polar_other():  # pragma: no cover
    problems = [run_Lorenz, run_Schroedinger, run_quench]
    paths = ['./data/stats/', './data/stats-jusuf', './data/stats-jusuf']
    titles = ['Lorenz attractor', r'Schr\"odinger', 'Quench']

    fig, axs = plt.subplots(
        1, 3, subplot_kw={'projection': 'polar'}, figsize=figsize_by_journal(JOURNAL, 0.7, 0.5), layout='constrained'
    )

    for i in range(len(problems)):
        stats_analyser = get_stats(problems[i], paths[i])
        ax = axs[i]
        theta, norms = plot_efficiency_polar_single(stats_analyser, ax)

        labels = ['fail rate', 'extra iterations\nfor recovery', 'iterations for solution']
        ax.set_rlabel_position(90)
        # ax.set_xticks(theta[:-1], [f'max={norms[i]:.2f}' for i in range(len(labels))])
        ax.set_xticks(theta[:-1], ['' for i in range(len(labels))])
        ax.set_title(titles[i])

    handles, labels = fig.get_axes()[0].get_legend_handles_labels()
    fig.legend(handles=handles, labels=labels, frameon=False, loc='outside lower center', ncols=4)
    savefig(fig, 'efficiency_other', tight_layout=False)


def plot_efficiency_polar_single(stats_analyser, ax):  # pragma: no cover
    """
    Plot the recovery rate and the computational cost in a polar plot.

    Shown are three axes, where lower is better in all cases.
    First is the fail rate, which is averaged across all faults, not just ones that can be fixed.
    Then, there is the number of iterations, which we use as a measure for how expensive the scheme is to run.
    And finally, there is an axis of how many extra iterations we need in case a fault is fixed by the resilience
    scheme.

    All quantities are plotted relative to their maximum.

    Args:
        problem (function): A problem to run
        path (str): Path to the associated stats for the problem

    Returns:
        None
    """
    # TODO: fix docs
    mask = stats_analyser.get_mask()  # get empty mask, potentially put in some other mask later

    my_setup_mpl()

    res = {}
    for strategy in stats_analyser.strategies:
        dat = stats_analyser.load(strategy=strategy, faults=True)
        dat_no_faults = stats_analyser.load(strategy=strategy, faults=False)

        mask = stats_analyser.get_fixable_faults_only(strategy=strategy)
        fail_rate = 1.0 - stats_analyser.rec_rate(dat, dat_no_faults, 'recovered', mask)
        iterations_no_faults = np.mean(dat_no_faults['total_iteration'])

        detected = stats_analyser.get_mask(strategy=strategy, key='total_iteration', op='gt', val=iterations_no_faults)
        rec_mask = stats_analyser.get_mask(strategy=strategy, key='recovered', op='eq', val=True, old_mask=detected)
        if rec_mask.any():
            extra_iterations = np.mean(dat['total_iteration'][rec_mask]) - iterations_no_faults
        else:
            extra_iterations = 0

        res[strategy.name] = [fail_rate, extra_iterations, iterations_no_faults]

    # normalize
    # for strategy in stats_analyser.strategies:
    norms = [max([res[k][i] for k in res.keys()]) for i in range(len(res['base']))]
    norms[1] = norms[2]  # use same norm for all iterations
    res_norm = res.copy()
    for k in res_norm.keys():
        for i in range(3):
            res_norm[k][i] /= norms[i]

    theta = np.array([30, 150, 270, 30]) * 2 * np.pi / 360
    for s in stats_analyser.strategies:
        ax.plot(theta, res_norm[s.name] + [res_norm[s.name][0]], label=s.label, color=s.color, marker=s.marker)
    return theta, norms


def plot_adaptivity_stuff():  # pragma: no cover
    """
    Plot the solution for a van der Pol problem as well as the local error and cost associated with the base scheme and
    adaptivity in k and dt in order to demonstrate that adaptivity is useful.

    Returns:
        None
    """
    from pySDC.implementations.convergence_controller_classes.estimate_embedded_error import EstimateEmbeddedError
    from pySDC.implementations.hooks.log_errors import LogLocalErrorPostStep
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
        k = get_sorted(stats, type='k')
        iter_ax.plot(
            [me[0] for me in k], np.cumsum([me[1] for me in k]), **strategy.style, markevery=markevery, **kwargs
        )
        ax.set_yscale('log')
        ax.set_ylabel('local error')
        iter_ax.set_ylabel(r'SDC iterations')

    force_params = {'convergence_controllers': {EstimateEmbeddedError: {}}}
    # force_params = {'convergence_controllers': {EstimateEmbeddedError: {}}, 'step_params': {'maxiter': 5}, 'level_params': {'dt': 4e-2}}
    for strategy in [BaseStrategy, AdaptivityStrategy, IterateStrategy]:
        stats, _, _ = stats_analyser.single_run(
            strategy=strategy(), force_params=force_params, hook_class=[LogLocalErrorPostStep, LogData]
        )
        plot_error(stats, axs[1], axs[2], strategy())

        if strategy == BaseStrategy:
            u = get_sorted(stats, type='u', recomputed=False)
            axs[0].plot([me[0] for me in u], [me[1][0] for me in u], color='black', label=r'$u$')
            # axs[0].plot([me[0] for me in u], [me[1][1] for me in u], color='black', ls='--', label=r'$u_t$')
            # axs[0].legend(frameon=False)

    axs[2].set_xlabel(r'$t$')
    axs[0].set_ylabel('solution')
    axs[2].legend(frameon=JOURNAL == 'JSC_beamer')
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

    custom_description = strategy.get_custom_description(run_quench)

    stats, controller, _ = run_quench(custom_description=custom_description, Tend=strategy.get_Tend(run_quench))

    prob = controller.MS[0].levels[0].prob

    u = get_sorted(stats, type='u')

    ax.plot([me[0] for me in u], [max(me[1]) for me in u], color='black', label='$T$')
    ax.axhline(prob.u_thresh, label='$T_\mathrm{thresh}$', ls='--', color='grey', zorder=-1)
    ax.axhline(prob.u_max, label='$T_\mathrm{max}$', ls=':', color='grey', zorder=-1)

    ax.set_xlabel(r'$t$')
    ax.legend(frameon=False)
    savefig(fig, 'quench_sol')


def plot_Lorenz_solution():  # pragma: no cover
    """
    Plot the solution of Lorenz attractor problem over time

    Returns:
        None
    """
    from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity
    from pySDC.projects.Resilience.strategies import AdaptivityStrategy
    from pySDC.implementations.hooks.log_errors import LogGlobalErrorPostRun

    strategy = AdaptivityStrategy()

    my_setup_mpl()

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # if JOURNAL == 'JSC_beamer':
    #    fig, ax = plt.subplots(figsize=figsize_by_journal(JOURNAL, 0.5, 0.9))
    # else:
    #    fig, ax = plt.subplots(figsize=figsize_by_journal(JOURNAL, 1.0, 0.33))

    custom_description = strategy.get_custom_description(run_Lorenz, 1)
    custom_description['convergence_controllers'] = {Adaptivity: {'e_tol': 1e-10}}

    stats, _, _ = run_Lorenz(
        custom_description=custom_description,
        Tend=strategy.get_Tend(run_Lorenz, 1) * 20,
        hook_class=LogGlobalErrorPostRun,
    )

    u = get_sorted(stats, type='u')
    e = get_sorted(stats, type='e_global_post_run')[-1]
    print(u[-1], e)
    ax.plot([me[1][0] for me in u], [me[1][1] for me in u], [me[1][2] for me in u])

    ##################
    from pySDC.projects.Resilience.strategies import DIRKStrategy, ERKStrategy, IterateStrategy
    from pySDC.implementations.convergence_controller_classes.adaptivity import AdaptivityRK

    strategy = ERKStrategy()
    custom_description = strategy.get_custom_description(run_Lorenz, 1)
    custom_description['convergence_controllers'] = {Adaptivity: {'e_tol': 1e-10}}
    stats, _, _ = run_Lorenz(
        custom_description=custom_description,
        Tend=strategy.get_Tend(run_Lorenz, 1) * 20,
        hook_class=LogGlobalErrorPostRun,
    )

    u = get_sorted(stats, type='u')
    e = get_sorted(stats, type='e_global_post_run')[-1]
    print(u[-1], e)
    ax.plot([me[1][0] for me in u], [me[1][1] for me in u], [me[1][2] for me in u], ls='--')
    ################
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    savefig(fig, 'lorenz_sol')


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


def work_precision():
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

    for mode in ['compare_strategies', 'parallel_efficiency']:
        all_problems(**all_params, mode=mode)

    # Quench stuff
    fig, axs = get_fig(x=3, y=1, figsize=figsize_by_journal('Springer_Numerical_Algorithms', 1, 0.47))
    quench_params = {
        **all_params,
        'problem': run_quench,
        'decorate': True,
        'configurations': get_configs('step_size_limiting', run_quench),
        'num_procs': 1,
        'runs': 1,
        'comm_world': MPI.COMM_WORLD,
    }
    quench_params.pop('base_path', None)
    execute_configurations(**{**quench_params, 'work_key': 'k_SDC', 'precision_key': 'k_Newton'}, ax=axs[2])
    execute_configurations(**{**quench_params, 'work_key': 'param', 'precision_key': 'restart'}, ax=axs[1])
    execute_configurations(**{**quench_params, 'work_key': 't', 'precision_key': 'e_global_rel'}, ax=axs[0])
    axs[1].set_yscale('linear')
    axs[2].set_yscale('linear')
    axs[2].set_xscale('linear')
    axs[1].set_xlabel(r'$e_\mathrm{tol}$')

    for ax in axs:
        ax.set_title(ax.get_ylabel())
        ax.set_ylabel('')
    fig.suptitle('Quench')
    save_fig(
        fig=fig,
        name=f'{run_quench.__name__}',
        work_key='step-size',
        precision_key='limiting',
        legend=True,
        base_path=all_params["base_path"],
    )

    vdp_stiffness_plot(base_path='data/paper')


def make_plots_for_TIME_X_website():
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

    plot_vdp_solution()
    plot_quench_solution()
    plot_recovery_rate(get_stats(run_vdp))
    plot_fault_vdp(0)
    plot_fault_vdp(13)
    plot_adaptivity_stuff()
    compare_recovery_rate_problems()
    work_precision()


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
