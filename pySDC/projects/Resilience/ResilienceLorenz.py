# Script for making some plots about resilience in the Lorenz problem
import numpy as np
import matplotlib.pyplot as plt

from pySDC.projects.Resilience.fault_stats import (
    FaultStats,
    BaseStrategy,
    AdaptivityStrategy,
    IterateStrategy,
    HotRodStrategy,
    run_Lorenz,
)


cm = 1 / 2.54


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
    path = f'notes/Lorenz/{name}.{format}'
    fig.savefig(path, bbox_inches='tight', transparent=True, dpi=200)
    print(f'saved "{path}"')


def analyse_resilience():  # pragma: no cover
    """
    Generate some stats for resilience / load them if already available and make some plots.

    Returns:
        None
    """

    stats_analyser = FaultStats(
        prob=run_Lorenz,
        strategies=[BaseStrategy(), AdaptivityStrategy(), IterateStrategy(), HotRodStrategy()],
        faults=[False, True],
        reload=True,
        recovery_thresh=1.1,
        num_procs=1,
        mode='combination',
    )
    stats_analyser.run_stats_generation(runs=5000, step=50)

    compare_strategies(stats_analyser)
    plot_recovery_rate(stats_analyser)


def compare_strategies(stats_analyser):  # pragma: no cover
    """
    Make a plot showing local error and iteration number of time for all strategies

    Args:
        stats_analyser (FaultStats): Fault stats object, which contains some stats

    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(16 * cm, 7 * cm))
    stats_analyser.compare_strategies(ax=ax)
    savefig(fig, 'compare_strategies', format='png')


def plot_recovery_rate(stats_analyser):  # pragma: no cover
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
    not_crashed = None
    for i in range(len(stats_analyser.strategies)):
        fixable = stats_analyser.get_fixable_faults_only(strategy=stats_analyser.strategies[i])

        stats_analyser.plot_things_per_things(
            'recovered',
            'bit',
            False,
            op=stats_analyser.rec_rate,
            mask=fixable,
            args={'ylabel': '', 'xlabel': ''},
            ax=axs[1],
            fig=fig,
            strategies=[stats_analyser.strategies[i]],
        )
    axs[1].get_legend().remove()
    axs[0].set_title('All faults')
    axs[1].set_title('Only recoverable faults')
    savefig(fig, 'recovery_rate_compared', format='png')


if __name__ == "__main__":
    analyse_resilience()
