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
)
from pySDC.helpers.plot_helper import setup_mpl


cm = 1 / 2.5


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

    return FaultStats(
        prob=problem,
        strategies=[BaseStrategy(), AdaptivityStrategy(), IterateStrategy(), HotRodStrategy()],
        faults=[False, True],
        reload=True,
        recovery_thresh=1.1,
        num_procs=1,
        mode=mode,
        stats_path=path,
    )


def savefig(fig, name, format='pdf', base_path='data/paper'):  # pragma: no cover
    """
    Save a figure to some predefined location.

    Args:
        fig (Matplotlib.Figure): The figure of the plot
        name (str): The name of the plot
    Returns:
        None
    """
    fig.tight_layout()
    path = f'{base_path}/{name}.{format}'
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
    fig, ax = plt.subplots(figsize=(16 * cm, 7 * cm))
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
    fig, axs = plt.subplots(1, 2, figsize=(16 * cm, 7 * cm), sharex=True, sharey=True)
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
    vdp_stats = get_stats(run_vdp)
    lorenz_stats = get_stats(run_Lorenz)
    schroedinger_stats = get_stats(run_Schroedinger, 'data/stats-jusuf/')
    titles = ['Van der Pol', 'Lorenz attractor', r'Schr\"odinger']

    fig, axs = plt.subplots(1, 3, figsize=(16 * cm, 5.5 * cm), sharex=False, sharey=True)

    plot_recovery_rate_recoverable_only(vdp_stats, fig, axs[0], ylabel='recovery rate')
    plot_recovery_rate_recoverable_only(lorenz_stats, fig, axs[1], ylabel='', xlabel='')
    plot_recovery_rate_recoverable_only(schroedinger_stats, fig, axs[2], ylabel='', xlabel='')

    for i in range(len(axs)):
        axs[i].set_title(titles[i])

    for ax in axs[1:]:
        ax.get_legend().remove()

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

    setup_mpl(reset=True, font_size=8)
    mpl.rcParams.update({'lines.markersize': 8})

    stats_analyser = get_stats(problem, path)

    mask = stats_analyser.get_mask()

    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8 * cm, 8 * cm))

    res = {}
    for strategy in stats_analyser.strategies:
        dat = stats_analyser.load(strategy=strategy, faults=True)
        dat_no_faults = stats_analyser.load(strategy=strategy, faults=False)

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
        ax.plot(theta, res_norm[s.name] + [res_norm[s.name][0]], label=s.name, color=s.color, marker=s.marker)

    labels = ['fail rate', 'extra iterations\nfor recovery', 'iterations for solution']
    ax.set_xticks(theta[:-1], [f'{labels[i]}\nmax={norms[i]:.2f}' for i in range(len(labels))])
    ax.set_rlabel_position(90)

    ax.legend(frameon=True, loc='lower right')
    savefig(fig, 'efficiency')


if __name__ == "__main__":
    plot_efficiency_polar(run_vdp)
    compare_recovery_rate_problems()
    analyse_resilience(run_Lorenz, format='png', base_path='notes/Lorenz')
