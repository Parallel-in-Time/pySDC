import pickle
import numpy as np
import matplotlib.pyplot as plt
from pySDC.projects.GPU.configs.base_config import get_config
from pySDC.projects.GPU.run_experiment import parse_args
from pySDC.helpers.stats_helper import get_sorted


def plot_step_size_single(args, ax, **plotting_params):  # pragma: no cover

    config = get_config(args)

    path = f'data/{config.get_path(ranks=[me -1 for me in args["procs"]])}-stats-whole-run.pickle'
    with open(path, 'rb') as file:
        stats = pickle.load(file)

    dt = get_sorted(stats, type='dt', recomputed=False)

    plotting_params = {
        **plotting_params,
    }
    ax.plot([me[0] for me in dt], [me[1] for me in dt], **plotting_params)
    ax.legend(frameon=False)
    ax.set_ylabel(r'$\Delta t$')
    ax.set_xlabel('$t$')


def plot_step_size(args):  # pragma: no cover
    fig, ax = plt.subplots()
    plot_step_size_single(args, ax)

    config = get_config(args)
    path = f'plots/{config.get_path(ranks=[me -1 for me in args["procs"]])}-dt.pdf'
    fig.savefig(path, bbox_inches='tight')
    plt.show()


def plot_step_size_GS(args):  # pragma: no cover
    fig, ax = plt.subplots()
    for config in ['GS_GoL', 'GS_dt']:
        plot_step_size_single({**args, 'config': config}, ax, label=config)

    path = 'plots/GrayScott-dt.pdf'
    fig.savefig(path, bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    args = parse_args()
    # plot_step_size(args)
    plot_step_size_GS(args)
