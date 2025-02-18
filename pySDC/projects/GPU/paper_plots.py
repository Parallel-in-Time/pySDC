"""Make plots for publications"""

import matplotlib.pyplot as plt
from pySDC.helpers.plot_helper import setup_mpl, figsize_by_journal


def plot_scalings_separately(problem, journal='TUHH_thesis', **kwargs):  # pragma: no cover
    from pySDC.projects.GPU.analysis_scripts.parallel_scaling import (
        plot_scalings,
        GrayScottSpaceScalingGPU3D,
        GrayScottSpaceScalingCPU3D,
        RayleighBenardSpaceScalingCPU,
        RayleighBenardSpaceScalingGPU,
        PROJECT_PATH,
    )

    if problem == 'GS3D':
        configs = [
            GrayScottSpaceScalingCPU3D(),
            GrayScottSpaceScalingGPU3D(),
        ]
    elif problem == 'RBC':
        configs = [
            RayleighBenardSpaceScalingCPU(),
            RayleighBenardSpaceScalingGPU(),
        ]

    else:
        raise NotImplementedError

    ideal_lines = {
        ('GS3D', 'throughput'): {'x': [0.25, 400], 'y': [5e6, 8e9]},
        ('GS3D', 'time'): {'x': [0.25, 400], 'y': [80, 5e-2]},
    }

    fig, ax = plt.subplots(figsize=figsize_by_journal(journal, 1, 0.6))
    configs[1].plot_scaling_test(ax=ax, quantity='efficiency')
    ax.legend(frameon=False)
    path = f'{PROJECT_PATH}/plots/scaling_{problem}_efficiency.pdf'
    fig.savefig(path, bbox_inches='tight')
    print(f'Saved {path!r}', flush=True)

    for quantity in ['time']:
        fig, ax = plt.subplots(figsize=figsize_by_journal(journal, 1, 0.6))
        for config in configs:
            config.plot_scaling_test(ax=ax, quantity=quantity)
        if (problem, quantity) in ideal_lines.keys():
            ax.loglog(*ideal_lines[(problem, quantity)].values(), color='black', ls=':', label='ideal')
        ax.legend(frameon=False)
        path = f'{PROJECT_PATH}/plots/scaling_{problem}_{quantity}.pdf'
        fig.savefig(path, bbox_inches='tight')
        print(f'Saved {path!r}', flush=True)


def make_plots_for_thesis():  # pragma: no cover
    from pySDC.projects.GPU.analysis_scripts.plot_RBC_matrix import plot_DCT, plot_preconditioners, plot_ultraspherical

    # small plots with no simulations
    plot_DCT()
    plot_preconditioners()
    plot_ultraspherical()

    # plot space-time parallel scaling
    for problem in ['GS3D', 'RBC']:
        plot_scalings_separately(problem=problem)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--target', choices=['thesis'], type=str)
    args = parser.parse_args()

    if args.target == 'thesis':
        make_plots_for_thesis()
    else:
        raise NotImplementedError(f'Don\'t know how to make plots for target {args.target}')
