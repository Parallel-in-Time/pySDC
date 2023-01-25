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


def savefig(fig, name, format='pdf'):
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


def analyse_resilience():
    """ """
    # TODO docs

    stats_analyser = FaultStats(
        prob=run_Lorenz,
        strategies=[BaseStrategy(), AdaptivityStrategy(), IterateStrategy(), HotRodStrategy()],
        faults=[False, True],
        reload=True,
        recovery_thresh=1.1,
        num_procs=1,
        mode='combination',
    )
    # stats_analyser.run_stats_generation(runs=5000, step=50)

    compare_strategies(stats_analyser)
    plot_recovery_rate(stats_analyser)
    stats_analyser.scrutinize(IterateStrategy(), 41)
    raise


def compare_strategies(stats_analyser):
    fig, ax = plt.subplots(figsize=(16 * cm, 7 * cm))
    stats_analyser.compare_strategies(ax=ax)
    savefig(fig, 'compare_strategies', format='png')


def plot_recovery_rate(stats_analyser):
    # TODO: docs
    mask = None
    fig, axs = plt.subplots(1, 2, figsize=(16 * cm, 7 * cm), sharex=True, sharey=True)
    stats_analyser.plot_things_per_things(
        'recovered', 'bit', False, op=stats_analyser.rec_rate, mask=mask, args={'ylabel': 'recovery rate'}, ax=axs[0]
    )
    not_crashed = None
    for i in range(len(stats_analyser.strategies)):
        fixable = stats_analyser.get_fixable_faults_only(strategy=stats_analyser.strategies[i])

        not_fixed = stats_analyser.get_mask(
            strategy=stats_analyser.strategies[i], key='recovered', op='eq', val=False, old_mask=fixable
        )

        if i == 2:
            dat = stats_analyser.load(strategy=stats_analyser.strategies[i])
            # print(dat['error'][not_fixed])
            print(stats_analyser.strategies[i].name)
            fixed = stats_analyser.get_mask(
                strategy=stats_analyser.strategies[i], key='recovered', op='eq', val=False, old_mask=fixable
            )
            stats_analyser.print_faults(fixed)

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
