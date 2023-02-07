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


cm = 1 / 2.5


def get_stats(problem, path='data/stats'):
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


if __name__ == "__main__":
    compare_recovery_rate_problems()
    analyse_resilience(run_Lorenz, format='png', base_path='notes/Lorenz')
