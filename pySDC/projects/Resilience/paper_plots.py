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
    run_leaky_superconductor,
)
from pySDC.helpers.plot_helper import setup_mpl, figsize_by_journal
from pySDC.helpers.stats_helper import get_sorted


cm = 1 / 2.5
TEXTWIDTH = 11.9446244611 * cm
JOURNAL = 'Springer_Numerical_Algorithms'
BASE_PATH = 'data/paper'


def get_stats(problem, path='data/stats'):
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
        run_leaky_superconductor: 5e-5,
    }

    strategies = [BaseStrategy(), AdaptivityStrategy(), IterateStrategy()]
    if JOURNAL not in ['JSC_beamer']:
        strategies += [HotRodStrategy()]

    return FaultStats(
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


def my_setup_mpl(**kwargs):
    setup_mpl(reset=True, font_size=8)
    mpl.rcParams.update({'lines.markersize': 6})


def savefig(fig, name, format='pdf'):  # pragma: no cover
    """
    Save a figure to some predefined location.

    Args:
        fig (Matplotlib.Figure): The figure of the plot
        name (str): The name of the plot
    Returns:
        None
    """
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
    plot_recovery_rate_recoverable_only(stats_analyser, fig, axs[1], ylabel='', xlabel='')
    axs[1].get_legend().remove()
    axs[0].set_title('All faults')
    axs[1].set_title('Only recoverable faults')
    savefig(fig, 'recovery_rate_compared', **kwargs)


def plot_recovery_rate_recoverable_only(stats_analyser, fig, ax, **kwargs):  # pragma no cover
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


def compare_recovery_rate_problems():  # pragma no cover
    """
    Compare the recovery rate for vdP, Lorenz and Schroedinger problems.
    Only faults that can be recovered are shown.

    Returns:
        None
    """
    stats = [
        get_stats(run_vdp),
        get_stats(run_Lorenz),
        get_stats(run_Schroedinger, 'data/stats-jusuf'),
        get_stats(run_leaky_superconductor, 'data/stats-jusuf'),
    ]
    titles = ['Van der Pol', 'Lorenz attractor', r'Schr\"odinger', 'Quench']

    my_setup_mpl()
    fig, axs = plt.subplots(2, 2, figsize=figsize_by_journal(JOURNAL, 1, 0.7), sharey=True)
    [
        plot_recovery_rate_recoverable_only(stats[i], fig, axs.flatten()[i], ylabel='', xlabel='', title=titles[i])
        for i in range(len(stats))
    ]

    for ax in axs.flatten():
        ax.get_legend().remove()

    axs[1, 1].legend(frameon=False)
    axs[1, 0].set_xlabel('bit')
    axs[1, 0].set_ylabel('recovery rate')

    savefig(fig, 'compare_equations')


def plot_efficiency_polar(problem, path='data/stats'):  # pragma no cover
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

    stats_analyser = get_stats(problem, path)
    mask = stats_analyser.get_mask()  # get empty mask, potentially put in some other mask later

    my_setup_mpl()
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(7 * cm, 7 * cm))

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

    labels = ['fail rate', 'extra iterations\nfor recovery', 'iterations for solution']
    ax.set_xticks(theta[:-1], [f'{labels[i]}\nmax={norms[i]:.2f}' for i in range(len(labels))])
    ax.set_rlabel_position(90)

    ax.legend(frameon=True, loc='lower right')
    savefig(fig, 'efficiency')


def plot_adaptivity_stuff():  # pragma no cover
    """
    Plot the solution for a van der Pol problem as well as the local error and cost associated with the base scheme and
    adaptivity in k and dt in order to demonstrate that adaptivity is useful.

    Returns:
        None
    """
    from pySDC.implementations.convergence_controller_classes.estimate_embedded_error import EstimateEmbeddedErrorNonMPI

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
        e = get_sorted(stats, type='error_embedded_estimate', recomputed=False)
        ax.plot([me[0] for me in e], [me[1] for me in e], markevery=15, **strategy.style, **kwargs)
        k = get_sorted(stats, type='k')
        iter_ax.plot([me[0] for me in k], np.cumsum([me[1] for me in k]), **strategy.style, markevery=15, **kwargs)
        ax.set_yscale('log')
        ax.set_ylabel('local error')
        iter_ax.set_ylabel(r'iterations')

    force_params = {'convergence_controllers': {EstimateEmbeddedErrorNonMPI: {}}}
    for strategy in [BaseStrategy, AdaptivityStrategy, IterateStrategy]:
        stats, _, _ = stats_analyser.single_run(strategy=strategy(), force_params=force_params)
        plot_error(stats, axs[1], axs[2], strategy())

        if strategy == AdaptivityStrategy:
            u = get_sorted(stats, type='u')
            axs[0].plot([me[0] for me in u], [me[1][0] for me in u], color='black', label=r'$u$')
            axs[0].plot([me[0] for me in u], [me[1][1] for me in u], color='black', ls='--', label=r'$u_t$')
            axs[0].legend(frameon=False)

    axs[1].set_ylim(bottom=1e-9)
    axs[2].set_xlabel(r'$t$')
    axs[0].set_ylabel('solution')
    axs[2].legend(frameon=JOURNAL == 'JSC_beamer')
    savefig(fig, 'adaptivity')


def plot_fault_vdp(bit=0):  # pragma no cover
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
    fig, ax = plt.subplots(1, 1, figsize=(TEXTWIDTH * 3.0 / 4.0, 5 * cm))
    colors = ['blue', 'red', 'magenta']
    ls = ['--', '-']
    markers = ['*', '.', 'y']
    do_faults = [False, True]
    superscripts = ['*', '']
    subscripts = ['', 't', '']

    run = 779 + 12 * bit

    for i in range(len(do_faults)):
        stats, controller, Tend = stats_analyser.single_run(
            strategy=BaseStrategy(),
            run=run,
            faults=do_faults[i],
            hook_class=[LogData],
        )
        u = get_sorted(stats, type='u')
        faults = get_sorted(stats, type='bitflip')
        for j in range(len(u[0][1])):
            ax.plot(
                [me[0] for me in u],
                [me[1][j] for me in u],
                ls=ls[i],
                color=colors[j],
                label=rf'$u^{{{superscripts[i]}}}_{{{subscripts[j]}}}$',
                marker=markers[0],
                markevery=15,
            )
        for idx in range(len(faults)):
            ax.axvline(faults[idx][0], color='black', label='Fault', ls=':')
            print(
                f'Fault at t={faults[idx][0]:.2e}, iter={faults[idx][1][1]}, node={faults[idx][1][2]}, space={faults[idx][1][3]}, bit={faults[idx][1][4]}'
            )
            ax.set_title(f'Fault in bit {faults[idx][1][4]}')

    ax.legend(frameon=False)
    ax.set_xlabel(r'$t$')
    savefig(fig, f'fault_bit_{bit}')


def plot_vdp_solution():  # pragma no cover
    """
    Plot the solution of van der Pol problem over time to illustrate the varying time scales.

    Returns:
        None
    """
    from pySDC.implementations.convergence_controller_classes.adaptivity import Adaptivity

    my_setup_mpl()
    fig, ax = plt.subplots(figsize=(9 * cm, 8 * cm))

    custom_description = {'convergence_controllers': {Adaptivity: {'e_tol': 1e-7}}}
    problem_params = {}

    stats, _, _ = run_vdp(custom_description=custom_description, custom_problem_params=problem_params, Tend=28.6)

    u = get_sorted(stats, type='u')
    ax.plot([me[0] for me in u], [me[1][0] for me in u], color='black')
    ax.set_ylabel(r'$u$')
    ax.set_xlabel(r'$t$')
    savefig(fig, 'vdp_sol')


def make_plots_for_SIAM_CSE23():  # pragma no cover
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


def make_plots_for_paper():  # pragma no cover
    """
    Make plots that are supposed to go in the paper.
    """
    global JOURNAL, BASE_PATH
    JOURNAL = 'Springer_Numerical_Algorithms'
    BASE_PATH = 'data/paper'

    plot_recovery_rate(get_stats(run_vdp))
    plot_fault_vdp(0)
    plot_fault_vdp(13)
    plot_adaptivity_stuff()
    plot_efficiency_polar(run_vdp)
    compare_recovery_rate_problems()


def make_plots_for_notes():  # pragma no cover
    """
    Make plots for the notes for the website / GitHub
    """
    global JOURNAL, BASE_PATH
    JOURNAL = 'Springer_Numerical_Algorithms'
    BASE_PATH = 'notes/Lorenz'

    analyse_resilience(run_Lorenz, format='png')


if __name__ == "__main__":
    make_plots_for_notes()
    make_plots_for_SIAM_CSE23()
    make_plots_for_paper()
